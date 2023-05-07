import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
from pytorch3d.ops.ball_query import ball_query
from pytorch3d.ops.utils import masked_gather
from pytorch3d.ops.sample_farthest_points import sample_farthest_points

DEVICE = torch.device("cuda:0")

class PointNet2(nn.Module):
    
    def __init__(self, num_class=3):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512, device=DEVICE)
        self.bn1 = nn.BatchNorm1d(512, device=DEVICE)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256, device=DEVICE)
        self.bn2 = nn.BatchNorm1d(256, device=DEVICE)
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, num_class, device=DEVICE)

    def forward(self, xyz):
        
        B, _, _ = xyz.shape
        
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        x = l3_points.view(B, 1024)
        x = self.drop1(self.bn1(self.fc1(x)))
        x = self.drop2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        
        #x = F.log_softmax(x, -1)

        return x

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1, stride=1, padding=0, device=DEVICE))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel, device=DEVICE))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # TODO: permute the points to get [B, N, C] shape
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        # TODO: permute again to get new_point as [B, C+D, nsample, npoint]
        #print(new_xyz.shape, new_points.shape)
        new_points = new_points.permute(0, 3, 2, 1)
        #print("New Points Shape - ")
        #print(new_points.shape)
        for i, conv in enumerate(self.mlp_convs):
            # TODO: apply conv and bn from self.mlp_convs and self.mlp_bns
            new_points = self.mlp_bns[i](conv(new_points.to(DEVICE)))
            #print(new_points.shape)
            
        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        #print(new_points.shape, new_xyz.shape)
        #print(" ------ ")
        return new_xyz, new_points

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    #print("Input Args -  ", npoint, radius, nsample, xyz.shape)
    B, N, C = xyz.shape
    S = npoint

    new_xyz = sample_farthest_points(xyz, K=npoint)[0] # [B, npoint, C]
    _, idx, grouped_xyz = ball_query(new_xyz, xyz, K=nsample, radius=radius)
    #grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    #print(xyz.shape, new_xyz.shape, idx.shape, grouped_xyz.shape)
    
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        #points = points.permute(0, 2, 1)
        #print(points.shape, idx.shape)
        grouped_points = masked_gather(points.to(DEVICE), idx.to(DEVICE)).to(DEVICE)
        #print(grouped_points.shape, points.shape, idx.shape, grouped_xyz_norm.shape)
        new_points = torch.cat([grouped_xyz_norm.to(DEVICE), grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz.to(DEVICE), points.to(DEVICE).view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points