# %matplotlib notebook
import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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
# import imgaug.augmenters as iaa
from torch.optim import lr_scheduler
import logging

from torchvision import transforms
from base import SetType
from dataset import AMASS
from utils import ConsoleLogger



def train_model(model, dataloader, device, logger,
                pose_loss, optimizer, scheduler, num_epochs=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        model.train()

        running_loss = 0.0
        # Iterate over data.
        size = 0
        i = 0
        for gt_pose in dataloader:
            gt_pose = gt_pose.to(device).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in trainf
            with torch.set_grad_enabled(True):
                pred_pose, loss_commit, perplexity = model(gt_pose)
                
                pred_pose = pred_pose.view(-1,14,3)
                gt_pose = gt_pose.view(-1,14,3)

                p_loss = pose_loss(pred_pose, gt_pose)

                loss = p_loss + 0.02 * loss_commit
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * gt_pose.size(0)
                size += gt_pose.size(0)
                
                if i % 10 == 0:
                    logger.info(f"Epoch: {epoch} : loss = {loss.item()} Step = {i}")
                    logger.info(f"l2 pose {p_loss.item()} commit {loss_commit.item()} LR: {scheduler.get_last_lr()}")
                i += 1
                if i % 5000 == 0:
                    torch.save(model.state_dict(), f"./ckpts/model_vqvae1/epoch{epoch}_iter{i}.pth")
            
        scheduler.step()
        epoch_loss = running_loss / size
        print('Loss: {:.4f}'.format(epoch_loss))
 
        torch.save(model.state_dict(), f"./ckpts/model_vqvae1/{epoch}_epoch_{epoch_loss}.pth")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights

#     model.load_state_dict(best_model_wts)
    return model


def main():
    # ------------------- Data loader -------------------

    # let's load data from training and validation set as example
    data = AMASS(data_path='/hdd4/cmk/dataset/AMASS/AMASS')

    train_loader = DataLoader(
        data,
        batch_size=100,
        shuffle=True)
        
    model = PoseVQVAE(down_t=1, fourier_features='positional')
        
    resume_path = None
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ckpt)
            
    ##### ---- Optimizer & Scheduler ---- #####
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.99), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50000, 400000], gamma=0.05)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    LossFn = torch.nn.MSELoss()
    
    logging.basicConfig(filename="./train_logs/vqvae1.log", format='%(message)s',
                        level=logging.INFO, filemode='w')
    logger = logging.getLogger('train_logger')
    print("loaded", device)

    best = train_model(model, train_loader, device, logger,
                       LossFn, optimizer, scheduler, 3)


if __name__ == "__main__":
    main()
