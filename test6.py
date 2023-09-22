import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from model.self_pose6 import SelfPose
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import copy
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import glob
from PIL import Image
import io

from torchvision import transforms
from base import SetType
import dataset.transform as trsf
from dataset import Mocap6 as Mocap
from utils import config

def visualization(pred_pose, gt_pose, p_loss, inference_time):
    # assume that batch size is 1
    pred_pose = pred_pose[0]
    gt_pose = gt_pose[0]
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle(str(p_loss) + '_' + str(inference_time), fontsize=16)
    
    linewidth = 4
    kinetic_chain = [[1, 0], [1, 2, 3, 4], [1, 5, 6, 7], [1, 8, 9, 10], [1, 11, 12, 13]]
    colors = ['black', 'red', 'blue', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for i, (chain, color) in enumerate(zip(kinetic_chain, colors)):
        ax1.plot3D(pred_pose[chain, 0], pred_pose[chain, 1], pred_pose[chain, 2], linewidth=linewidth, color=color)
        ax2.plot3D(gt_pose[chain, 0], gt_pose[chain, 1], gt_pose[chain, 2], linewidth=linewidth, color=color)
    
    ax1.scatter(pred_pose[:, 0], pred_pose[:, 1], pred_pose[:, 2])
    ax2.scatter(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2])
    
    ax1.set_xlabel('Pred', fontsize=10, rotation=100)
    ax2.set_xlabel('GT', fontsize=10, rotation=100)
    
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
        
    model = SelfPose()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("./ckpts/model6/epoch6_iter15000.pth"))
    model.to(device)
    model.eval()
    
    pose_loss = nn.MSELoss()
    
    for image, p2d, gt_pose, action, gt_hm in test_loader:
        image = image.to(device).float()
        gt_pose = gt_pose.to(device).float()
        gt_hm = gt_hm.to(device).float()
        
        start = time.time()
        pred_pose = model.inference(image)
        inference_time = time.time() - start
        pred_pose = pred_pose.detach().cpu().view(-1, 14, 3)
        gt_pose = gt_pose.detach().cpu().view(-1, 14, 3)
        
        p_loss = pose_loss(pred_pose, gt_pose)
                
        visualization(pred_pose.numpy(), gt_pose.numpy(), p_loss.numpy(), inference_time)

if __name__ == "__main__":
    main()
