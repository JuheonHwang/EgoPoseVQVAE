# %matplotlib notebook
import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from model.self_pose2 import SelfPose
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
# import imgaug.augmenters as iaa
from torch.optim import lr_scheduler
import logging

from torchvision import transforms
from base import SetType
import dataset.transform as trsf
from dataset import Mocap2 as Mocap
from utils import config2 as config
from utils import ConsoleLogger
from utils import evaluate, io


def train_model(model, dataloaders, device, logger,
                pose_loss, heatmap_loss, heatmap_loss2,l1_loss,
                cos_loss, optimizer, scheduler, num_epochs=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            sizes = {"train":0, "val":0}
            i = 0
            for image, p2d, gt_pose, action, gt_hm in dataloaders[phase]:
                image = image.to(device).float()
                gt_pose = gt_pose.to(device).float()
                gt_hm = gt_hm.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in trainf
                with torch.set_grad_enabled(phase == 'train'):
                    pred_pose, hm, pred_hm = model(image)
                    pred_pose = pred_pose.view(-1,16,3)
                    gt_pose = gt_pose.view(-1,16,3)
#                     _, preds = torch.max(outputs, 1)
                    p_loss = pose_loss(pred_pose, gt_pose)
#                     cosine_loss = torch.mean(1 - cos_loss(pred_pose, gt_pose))
#                     l1_loss_pose = l1_loss(pred_pose, gt_pose)
                    hmp_loss = heatmap_loss(hm, gt_hm)
                    hmp_emb_loss = heatmap_loss2(pred_hm, hm)

                    loss = p_loss + (hmp_loss/50) + (hmp_emb_loss/50) #+ (l1_loss_pose) + (0.1*cosine_loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':

                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * image.size(0)
                sizes[phase] += image.size(0)
                if i % 10 == 0:
                    logger.info(f"Epoch: {epoch} Phase {phase}: loss = {loss.item()} Step = {i}")
                    logger.info(f"l2 pose {p_loss.item()} hmp loss {hmp_loss.item()} hmp2 {hmp_emb_loss.item()} LR: {scheduler.get_last_lr()}")# "l1 loss",l1_loss_pose.item(), "cos loss",cosine_loss.item())
                i += 1
                if i % 5000 == 0:
                    torch.save(model.state_dict(), f"./ckpts/model2/epoch{epoch}_iter{i}.pth")
#                 running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
 

        print()
        torch.save(model.state_dict(), f"./ckpts/model2/{epoch}_epoch_{epoch_loss}.pth")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights

#     model.load_state_dict(best_model_wts)
    return model


def main():
    # ------------------- Data loader -------------------
    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    # let's load data from training and validation set as example
    data = Mocap(
        config.dataset.train,
        SetType.TRAIN,
        transform=data_transform)
    train_loader = DataLoader(
        data,
        batch_size=config.data_loader.batch_size,
        shuffle=config.data_loader.shuffle)
    
    val_data = Mocap(
        config.dataset.val,
        SetType.VAL,
        transform=data_transform)
    valid_loader = DataLoader(
        val_data,
        batch_size=config.val_data_loader.batch_size,
        shuffle=config.val_data_loader.shuffle)
        
    model = SelfPose()
    dataloaders = {"train": train_loader, "val": valid_loader}
    pose_criterion = nn.MSELoss()
    hmp_loss = nn.MSELoss()
    hmp_loss2 = nn.MSELoss()
    l1_loss = nn.L1Loss()
    cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0001) 
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.95)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.basicConfig(filename="./train_logs/model2.log", format='%(message)s',
                        level=logging.INFO, filemode='w')
    logger = logging.getLogger('train_logger')
    print("loaded", device)

    best = train_model(model, dataloaders, device, logger,
                       pose_criterion, hmp_loss, hmp_loss2, l1_loss,
                       cos_sim, optimizer_ft, scheduler, 7)


if __name__ == "__main__":
    main()
