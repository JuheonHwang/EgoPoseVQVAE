"""Train a SelfPose (EgoPose) model, e.g.

    python train.py --config configs/model1.yml --data-root /path/to/xr-egopose

Every experiment (models 1-6) is described by its YAML file in configs/.
"""
import argparse
import os
import time

from utils.runtime import init_gpu
init_gpu()  # CUDA_VISIBLE_DEVICES must be set before torch is imported

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.optim import lr_scheduler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

import dataset.transform as trsf  # noqa: E402
from base import SetType  # noqa: E402
from dataset import Mocap  # noqa: E402
from model.self_pose import build_self_pose  # noqa: E402
from utils import config, load_config  # noqa: E402
from utils.runtime import get_device, override_output, setup_logging  # noqa: E402


def compute_loss(model, batch, device, mse):
    """Forward one batch.

    Returns the total loss and a list of (name, value) loss terms for logging.
    """

    image, _, gt_pose, _, gt_hm, *_ = batch
    image = image.to(device).float()
    gt_pose = gt_pose.to(device).float()
    gt_hm = gt_hm.to(device).float()

    n_joints = len(config.skel)
    pred_pose, hm, pred_hm, *_ = model(image)
    pred_pose = pred_pose.view(-1, n_joints, 3)
    gt_pose = gt_pose.view(-1, n_joints, 3)

    p_loss = mse(pred_pose, gt_pose)
    hmp_loss = mse(hm, gt_hm)
    hmp_emb_loss = mse(pred_hm, hm)
    hm_div = config.train.heatmap_div

    if not config.model.rotation:
        loss = p_loss + (hmp_loss / hm_div) + (hmp_emb_loss / hm_div)
        return loss, [('l2 pose', p_loss), ('hmp loss', hmp_loss),
                      ('hmp2', hmp_emb_loss)]

    # NOTE: this reproduces experiments 4 and 5, where the "rotation" loss was
    # computed between the predicted and ground-truth *pose* (not between
    # pred_rot and gt_rot, which are never used). Probably unintended.
    r_loss = mse(pred_pose, gt_pose)
    loss = (p_loss + (r_loss / config.train.rot_div)
            + (hmp_loss / hm_div) + (hmp_emb_loss / hm_div))
    return loss, [('l2 pose', p_loss), ('rot', r_loss), ('hmp loss', hmp_loss),
                  ('hmp2', hmp_emb_loss)]


def train_model(model, dataloaders, device, logger, mse, optimizer,
                scheduler, num_epochs):
    since = time.time()
    save_every = config.train.save_every
    ckpt_dir = config.output.ckpt_dir

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            size = 0
            i = 0
            for batch in dataloaders[phase]:
                batch_size = batch[0].size(0)

                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss, terms = compute_loss(model, batch, device, mse)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_size
                size += batch_size
                if i % 10 == 0:
                    logger.info(f"Epoch: {epoch} Phase {phase}: loss = {loss.item()} Step = {i}")
                    terms = ' '.join(f"{name} {value.item()}" for name, value in terms)
                    logger.info(f"{terms} LR: {scheduler.get_last_lr()}")
                i += 1
                # NOTE: as in the original scripts this also runs in the
                # validation phase (same file name as the training one).
                if i % save_every == 0:
                    torch.save(model.state_dict(),
                               os.path.join(ckpt_dir, f"epoch{epoch}_iter{i}.pth"))

            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / size
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        print()
        # epoch_loss is the one of the last phase, i.e. the validation loss
        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, f"{epoch}_epoch_{epoch_loss}.pth"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a SelfPose model')
    parser.add_argument('--config', required=True, help='configs/modelN.yml')
    parser.add_argument('--data-root', help='dataset root (TrainSet/ValSet/TestSet); overrides dataset.root')
    parser.add_argument('--gpu', help='GPU id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epochs', type=int, help='overrides train.epochs')
    parser.add_argument('--batch-size', type=int, help='overrides data_loader.batch_size')
    parser.add_argument('--output-dir', help='write checkpoints to <dir>/ckpts/<last part of output.ckpt_dir>/ and the log to '
                        '<dir>/<name of output.log_file> instead of the output.* paths of the config')
    return parser.parse_args()


def main():
    args = parse_args()
    load_config(args.config, **{
        'dataset.root': args.data_root,
        'train.epochs': args.epochs,
        'data_loader.batch_size': args.batch_size})
    override_output(config, args.output_dir)

    os.makedirs(config.output.ckpt_dir, exist_ok=True)
    logger = setup_logging(config.output.log_file)

    # ------------------- Data loader -------------------
    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    data = Mocap(config.dataset.train, SetType.TRAIN, transform=data_transform)
    train_loader = DataLoader(
        data,
        batch_size=config.data_loader.batch_size,
        shuffle=config.data_loader.shuffle)

    val_data = Mocap(config.dataset.val, SetType.VAL, transform=data_transform)
    valid_loader = DataLoader(
        val_data,
        batch_size=config.val_data_loader.batch_size,
        shuffle=config.val_data_loader.shuffle)
    dataloaders = {"train": train_loader, "val": valid_loader}

    # ------------------- Model, loss, optimizer -------------------
    model = build_self_pose(config)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.train.lr_gamma)
    device = get_device()
    model.to(device)
    print("loaded", device)

    train_model(model, dataloaders, device, logger, mse, optimizer,
                scheduler, config.train.epochs)


if __name__ == "__main__":
    main()
