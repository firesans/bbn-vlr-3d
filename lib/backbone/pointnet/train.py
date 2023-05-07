import numpy as np
import argparse
import torch
import time
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pointnet import Pointnet
from utils import save_checkpoint, create_dir
from pointnet2 import PointNet2

from torch_geometric.datasets import ModelNet, ShapeNet
from torch_geometric.loader import DataLoader
#from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import scatter
import torch_geometric.transforms as T

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train(train_dataloader, model, opt, epoch, args, writer):
    
    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        #print(batch.y)
        s = batch.y.shape[0]
        point_clouds, labels = batch.pos.view(s, -1, 3), batch.y
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)

        # ------ TO DO: Forward Pass ------
        if (args.task == "cls"):
            predictions = model(point_clouds)
            
        # Compute Loss
        criterion = torch.nn.CrossEntropyLoss()
        #print(predictions.shape, labels.shape)
        loss = criterion(predictions, labels)
        epoch_loss += loss

        # Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar('train_loss', loss.item(), step+i)

    return epoch_loss

def test(test_dataloader, model, epoch, args, writer):
    
    model.eval()

    # Evaluation in Classification Task
    if (args.task == "cls"):
        correct_obj = 0
        num_obj = 0
        for batch in test_dataloader:
            s = batch.y.shape[0]
            point_clouds, labels = batch.pos.view(s, -1, 3), batch.y
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                pred_labels = model(point_clouds)
                pred_labels = torch.argmax(pred_labels, dim=-1)
                
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_obj / num_obj           

    writer.add_scalar("test_acc", accuracy, epoch)
    return accuracy


def main(args):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """
    print(torch.cuda.is_available())
    # Create Directories
    create_dir(args.checkpoint_dir)
    create_dir('./logs')

    # Tensorboard Logger
    writer = SummaryWriter('./logs/{0}'.format(args.task+"_"+args.exp_name))

    # ------ TO DO: Initialize Model ------
    if args.task == "cls":
        model = Pointnet(num_classes=10)
    
    # Load Checkpoint 
    if args.load_checkpoint:
        model_path = "{}/{}.pt".format(args.checkpoint_dir,args.load_checkpoint)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
        print ("successfully loaded checkpoint from {}".format(model_path))

    # Optimizer
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(2048)
    
    train_dataset = ModelNet(root="ModelNet10", name='10', train=True, transform=transform, pre_transform=pre_transform)
    val_dataset = ModelNet(root="ModelNet10", name='10', train=False, transform=transform, pre_transform=pre_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)

    print ("successfully loaded data")

    best_acc = -1

    print ("======== start training for {} task ========".format(args.task))
    print ("(check tensorboard for plots of experiment logs/{})".format(args.task+"_"+args.exp_name))
    
    for epoch in range(args.num_epochs):

        # Train
        stime = time.time()
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)
        etime = time.time() - stime
        print("Time :", etime)
        
        # Test
        current_acc = test(test_dataloader, model, epoch, args, writer)

        print ("epoch: {}   train loss: {:.4f}   test accuracy: {:.4f}".format(epoch, train_epoch_loss, current_acc))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            print ("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=False)

        # Save Best Model Checkpoint
        if (current_acc >= best_acc):
            best_acc = current_acc
            print ("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=True)

    print ("======== training completes ========")


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model & Data hyper-parameters
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (default 0.001)')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=10)

    parser.add_argument('--load_checkpoint', type=str, default='')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir+"/"+args.task+"_vlr" # checkpoint directory is task specific

    main(args)
