# %matplotlib notebook
import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from model.self_pose6 import SelfPose
from model.pose_VQVAE import PoseVQVAE
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import copy
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

from torchvision import transforms
import dataset.transform as trsf
from base import SetType
from dataset import AMASS_TEST as AMASS
from dataset import Mocap6 as Mocap
from utils import config

def visualization(pred_pose1, pred_pose2, gt_pose, p_loss1, p_loss2, inference_time):
    # assume that batch size is 1
    pred_pose1 = pred_pose1[0]
    pred_pose2 = pred_pose2[0]
    gt_pose = gt_pose[0]
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle(str(p_loss1) + '_' + str(p_loss2) + '_' + str(inference_time), fontsize=16)
    
    linewidth = 4
    kinetic_chain = [[1, 0], [1, 2, 3, 4], [1, 5, 6, 7], [1, 8, 9, 10], [1, 11, 12, 13]]
    colors = ['black', 'red', 'blue', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    for i, (chain, color) in enumerate(zip(kinetic_chain, colors)):
        ax1.plot3D(pred_pose1[chain, 0], pred_pose1[chain, 1], pred_pose1[chain, 2], linewidth=linewidth, color=color)
        ax2.plot3D(pred_pose2[chain, 0], pred_pose2[chain, 1], pred_pose2[chain, 2], linewidth=linewidth, color=color)
        ax3.plot3D(gt_pose[chain, 0], gt_pose[chain, 1], gt_pose[chain, 2], linewidth=linewidth, color=color)
    
    ax1.scatter(pred_pose1[:, 0], pred_pose1[:, 1], pred_pose1[:, 2])
    ax2.scatter(pred_pose2[:, 0], pred_pose2[:, 1], pred_pose2[:, 2])
    ax3.scatter(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2])
    
    ax1.set_xlabel('SelfPose', fontsize=10, rotation=100)
    ax2.set_xlabel('SelfPose+VQVAE', fontsize=10, rotation=100)
    ax3.set_xlabel('GT', fontsize=10, rotation=100)
    
    plt.show()


def main():
    # ------------------- Data loader -------------------
    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])
    
    # let's load data from training and validation set as example
    test_data = Mocap(
        config.dataset.test,
        SetType.TEST,
        transform=data_transform)
    test_loader = DataLoader(
        test_data,
        batch_size=config.val_data_loader.batch_size,
        shuffle=False)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SelfPose()
    model.load_state_dict(torch.load("./ckpts/model6/epoch6_iter15000.pth"))
    model.to(device)
    model.eval()
        
    VQVAE = PoseVQVAE(down_t=1, fourier_features='positional')
    VQVAE.load_state_dict(torch.load("./ckpts/model_vqvae1/2_epoch_0.0006308839614052485.pth"))
    VQVAE.to(device)
    VQVAE.eval()
    
    pose_loss = torch.nn.MSELoss()
    
    for image, _, gt_pose, _, _ in test_loader:
        image = image.to(device).float()
        gt_pose = gt_pose.to(device).float()
        
        start = time.time()
        pred_pose1 = model.inference(image)
        pred_pose1 = pred_pose1.detach().view(-1, 14, 3)
        pred_pose2, _, _ = VQVAE(pred_pose1)
        inference_time = time.time() - start
        pred_pose1 = pred_pose1.detach().cpu().view(-1, 14, 3)
        pred_pose2 = pred_pose2.detach().cpu().view(-1, 14, 3)
        gt_pose = gt_pose.detach().cpu().view(-1, 14, 3)
        
        p_loss1 = pose_loss(pred_pose1, gt_pose)
        p_loss2 = pose_loss(pred_pose2, gt_pose)
                
        visualization(pred_pose1.numpy(), pred_pose2.numpy(), gt_pose.numpy(), p_loss1.numpy(), p_loss2.numpy(), inference_time)
        
if __name__ == "__main__":
    main()
