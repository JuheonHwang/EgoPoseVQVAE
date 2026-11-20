"""Train the pose VQ-VAE on AMASS, e.g.

    python train_vqvae.py --config configs/vqvae1.yml --amass-root /path/to/AMASS
"""
import argparse
import os
import time

from utils.runtime import init_gpu
init_gpu()  # CUDA_VISIBLE_DEVICES must be set before torch is imported

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from dataset import AMASS  # noqa: E402
from model.pose_VQVAE import build_pose_vqvae  # noqa: E402
from utils import read_config  # noqa: E402
from utils.runtime import get_device, override_output, setup_logging  # noqa: E402


def train_model(model, dataloader, device, logger, config, pose_loss,
                optimizer, scheduler):
    since = time.time()
    num_epochs = config.train.epochs
    n_joints = len(config.skel)
    ckpt_dir = config.output.ckpt_dir

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()

        running_loss = 0.0
        size = 0
        i = 0
        for gt_pose in dataloader:
            gt_pose = gt_pose.to(device).float()

            optimizer.zero_grad()

            pred_pose, loss_commit, perplexity = model(gt_pose)

            pred_pose = pred_pose.view(-1, n_joints, 3)
            gt_pose = gt_pose.view(-1, n_joints, 3)

            p_loss = pose_loss(pred_pose, gt_pose)

            loss = p_loss + config.train.commit_weight * loss_commit
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * gt_pose.size(0)
            size += gt_pose.size(0)

            if i % 10 == 0:
                logger.info(f"Epoch: {epoch} : loss = {loss.item()} Step = {i}")
                logger.info(f"l2 pose {p_loss.item()} commit {loss_commit.item()} LR: {scheduler.get_last_lr()}")
            i += 1
            if i % config.train.save_every == 0:
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"epoch{epoch}_iter{i}.pth"))

        scheduler.step()
        epoch_loss = running_loss / size
        print('Loss: {:.4f}'.format(epoch_loss))

        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, f"{epoch}_epoch_{epoch_loss}.pth"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train the pose VQ-VAE on AMASS')
    parser.add_argument('--config', required=True, help='configs/vqvaeN.yml')
    parser.add_argument('--amass-root', help='AMASS root; overrides amass.root')
    parser.add_argument('--smpl-root', help='SMPL model directory; overrides amass.smpl_model_root')
    parser.add_argument('--gpu', help='GPU id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epochs', type=int, help='overrides train.epochs')
    parser.add_argument('--batch-size', type=int, help='overrides data_loader.batch_size')
    parser.add_argument('--resume', help='state dict to initialise the model from')
    parser.add_argument('--output-dir', help='write checkpoints to <dir>/ckpts/<last part of output.ckpt_dir>/ and the log to '
                        '<dir>/<name of output.log_file> instead of the output.* paths of the config')
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_config(args.config, **{
        'amass.root': args.amass_root,
        'amass.smpl_model_root': args.smpl_root,
        'train.epochs': args.epochs,
        'data_loader.batch_size': args.batch_size})
    override_output(config, args.output_dir)

    os.makedirs(config.output.ckpt_dir, exist_ok=True)
    logger = setup_logging(config.output.log_file)

    data = AMASS(config.amass.root, split='train',
                 smpl_model_root=config.amass.smpl_model_root)
    train_loader = DataLoader(
        data,
        batch_size=config.data_loader.batch_size,
        shuffle=config.data_loader.shuffle)

    model = build_pose_vqvae(config)
    device = get_device()
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr,
                            betas=tuple(config.train.betas),
                            weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.train.milestones, gamma=config.train.lr_gamma)

    model.to(device)
    print("loaded", device)

    train_model(model, train_loader, device, logger, config, nn.MSELoss(),
                optimizer, scheduler)


if __name__ == "__main__":
    main()
