import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda:0")
DEVICE = torch.device("cpu")


class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.tnet_block1 = nn.Sequential(
            nn.Conv1d(k, 64, 1, device=DEVICE),
            nn.BatchNorm1d(64, device=DEVICE),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1, device=DEVICE),
            nn.BatchNorm1d(128, device=DEVICE),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1, device=DEVICE),
            nn.BatchNorm1d(1024, device=DEVICE),
            nn.ReLU(True)
        )
        self.tnet_block2 = nn.Sequential(
            nn.Linear(1024, 512, device=DEVICE),
            nn.BatchNorm1d(512, device=DEVICE),
            nn.ReLU(True),
            nn.Linear(512, 256, device=DEVICE),
            nn.BatchNorm1d(256, device=DEVICE),
            nn.ReLU(True),
            nn.Linear(256, k*k, device=DEVICE)
        )

    def forward(self, input):
        # input points : B, N, 3
        #print(input.shape)
        B, N, _ = input.shape
        input = input.to(DEVICE)
        input = input.permute(0, 2, 1) # B, 3, N
        output = self.tnet_block1(input) # B, 1024, N
        output = nn.MaxPool1d(int(output.size(2)))(output) # B, 1024, 1
        output = nn.Flatten()(output) # B, 1024
        output = self.tnet_block2(output) # B, 9

        output = output.view(B, self.k, self.k).to(DEVICE)
        i_kk = torch.eye(self.k, dtype=torch.float).unsqueeze(0).to(DEVICE)
        output = output + i_kk

        return output

# ------ TO DO ------


class Pointnet(nn.Module):
    def __init__(self):
        super(Pointnet, self).__init__()

        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)

        self.mlp_block1 = nn.Sequential(
            nn.Conv1d(3, 64, 1, device=DEVICE),
            nn.BatchNorm1d(64, device=DEVICE),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1, device=DEVICE),
            nn.BatchNorm1d(64, device=DEVICE),
            nn.ReLU(True)
        )

        self.mlp_block2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, device=DEVICE),
            nn.BatchNorm1d(64, device=DEVICE),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1, device=DEVICE),
            nn.BatchNorm1d(128, device=DEVICE),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1, device=DEVICE),
            nn.BatchNorm1d(1024, device=DEVICE),
            nn.ReLU(True)
        )

        self.final_mlp_block = nn.Sequential(
            nn.Linear(1024, 512, device=DEVICE),
            nn.BatchNorm1d(512, device=DEVICE),
            nn.ReLU(True),
            nn.Linear(512, 256, device=DEVICE),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256, device=DEVICE),
            nn.ReLU(True)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        points = points.to(DEVICE)
        matrix_3 = self.input_transform(points) # B, 3, 3
        output = torch.bmm(points, matrix_3.to(DEVICE)) # B, N, 3
        output = output.permute(0, 2, 1) # B, 3, N
        output = self.mlp_block1(output) # B, 64, N

        output = output.permute(0, 2, 1) # B, N, 64
        matrix_64 = self.feature_transform(output) # B, 64, 64
        output = torch.bmm(output, matrix_64) # B, N, 64

        output = output.permute(0, 2, 1) # B, 64, N
        output = self.mlp_block2(output) # B, 1024, N

        output = nn.MaxPool1d(int(output.size(2)))(output) # B, 1024, 1
        output = nn.Flatten()(output) # B, 1024

        output = self.final_mlp_block(output) # B, 256

        return output

# ------ TO DO ------


class seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(seg_model, self).__init__()
        self.classes = num_seg_classes

        self.mlp_block1 = nn.Sequential(
            nn.Conv1d(3, 64, 1, device=DEVICE),
            nn.BatchNorm1d(64, device=DEVICE),
            nn.ReLU(True)
        )
        self.mlp_block2 = nn.Sequential(
            nn.Conv1d(64, 128, 1, device=DEVICE),
            nn.BatchNorm1d(128, device=DEVICE),
            nn.ReLU(True)
        )
        self.mlp_block3 = nn.Sequential(
            nn.Conv1d(128, 128, 1, device=DEVICE),
            nn.BatchNorm1d(128, device=DEVICE),
            nn.ReLU(True)
        )
        self.mlp_block4 = nn.Sequential(
            nn.Conv1d(128, 128, 1, device=DEVICE),
            nn.BatchNorm1d(128, device=DEVICE),
            nn.ReLU(True)
        )
        self.mlp_block5 = nn.Sequential(
            nn.Conv1d(128, 512, 1, device=DEVICE),
            nn.BatchNorm1d(512, device=DEVICE),
            nn.ReLU(True)
        )
        self.mlp_block6 = nn.Sequential(
            nn.Conv1d(512, 1024, 1, device=DEVICE),
            nn.BatchNorm1d(1024, device=DEVICE),
            nn.ReLU(True)
        )

        self.mlp1 = nn.Sequential(
            nn.Conv1d(1664, 256, 1, device=DEVICE),
            nn.BatchNorm1d(256, device=DEVICE),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 1, device=DEVICE),
            nn.BatchNorm1d(256, device=DEVICE),
            nn.ReLU(True),
            nn.Conv1d(256, 128, 1, device=DEVICE),
            nn.BatchNorm1d(128, device=DEVICE),
            nn.ReLU(True),
            nn.Conv1d(128, num_seg_classes, 1, device=DEVICE)
        )

    def forward(self, points, labels):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        labels: tensor of size (B, N)
        output: tensor of size (B, N, num_seg_classes)
        '''
        B, N, _ = points.shape
        points = points.to(DEVICE)
        #matrix_3 = self.cls.input_transform(points) # B, 3, 3
        #output = torch.bmm(points, matrix_3.to(DEVICE)) # B, N, 3
        points = points.permute(0, 2, 1) # B, 3, N

        seg_feat1 = self.mlp_block1(points) # B, 64, N
        seg_feat2 = self.mlp_block2(seg_feat1) # B, 128, N
        seg_feat3 = self.mlp_block3(seg_feat2) # B, 128, N
        seg_feat4 = self.mlp_block4(seg_feat3) # B, 128, N
        seg_feat5 = self.mlp_block5(seg_feat4) # B, 512, N
        seg_feat6 = self.mlp_block6(seg_feat5) # B, 1024, N

        output = nn.MaxPool1d(seg_feat6.size(2))(seg_feat6) # B, 1024, 1
        embeddings = nn.Flatten()(output) # B, 1024

        embeddings = embeddings.unsqueeze(2).repeat(1, 1, N) # B, 1024, N
        #label_one_hot = F.one_hot(labels, num_classes=self.classes)
        seg_feats = torch.cat((seg_feat4, seg_feat5, embeddings), dim=1) # B, 1664, N

        seg_output = self.mlp1(seg_feats) # B, num_seg_classes, N
        seg_output = seg_output.permute(0, 2, 1) # B, N, num_seg_classes

        return seg_output


def pointNetModel(
    cfg,
    pretrain=False,
    pretrained_model="/data/Data/pretrain_models/resnet50-19c8e357.pth",
    last_layer_stride=2,
):
    pointnet = Pointnet()
    if pretrain and pretrained_model != "":
        pointnet.load_model(pretrain=pretrained_model)
    else:
        print("Choose to train from scratch")
    return pointnet


if __name__ == "__main__":
    pnet = Pointnet()
    B = 64
    N = 2000
    x = torch.rand(B, N, 3)
    result = pnet(x)
    import pdb
    pdb.set_trace()
