"""Visualise SelfPose predictions against the ground truth, e.g.

    python test.py --config configs/model6.yml --checkpoint ckpts/model6/epoch6_iter15000.pth
"""
import argparse
import time

from utils.runtime import init_gpu
init_gpu()  # CUDA_VISIBLE_DEVICES must be set before torch is imported

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

import dataset.transform as trsf  # noqa: E402
from base import SetType  # noqa: E402
from dataset import Mocap  # noqa: E402
from model.self_pose import build_self_pose  # noqa: E402
from utils import config, load_config  # noqa: E402
from utils.runtime import get_device  # noqa: E402
from utils.visualize import show_poses  # noqa: E402

SET_TYPES = {'train': SetType.TRAIN, 'val': SetType.VAL, 'test': SetType.TEST}


def parse_args():
    parser = argparse.ArgumentParser(description='Test a SelfPose model')
    parser.add_argument('--config', required=True, help='configs/modelN.yml')
    parser.add_argument('--checkpoint', required=True, help='state dict written by train.py')
    parser.add_argument('--data-root', help='dataset root (TrainSet/ValSet/TestSet); overrides dataset.root')
    parser.add_argument('--split', choices=sorted(SET_TYPES), default='test',
                        help='split to evaluate (default: test)')
    parser.add_argument('--gpu', help='GPU id(s) for CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def main():
    args = parse_args()
    load_config(args.config, **{'dataset.root': args.data_root})

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    test_data = Mocap(config.dataset[args.split], SET_TYPES[args.split],
                      transform=data_transform)
    test_loader = DataLoader(
        test_data,
        batch_size=config.val_data_loader.batch_size,
        shuffle=False)

    device = get_device()
    model = build_self_pose(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    pose_loss = nn.MSELoss()
    n_joints = len(config.skel)

    for image, _, gt_pose, *_ in test_loader:
        image = image.to(device).float()
        gt_pose = gt_pose.to(device).float()

        start = time.time()
        with torch.no_grad():
            pred_pose = model.inference(image)
        inference_time = time.time() - start
        pred_pose = pred_pose.cpu().view(-1, n_joints, 3)
        gt_pose = gt_pose.cpu().view(-1, n_joints, 3)

        p_loss = pose_loss(pred_pose, gt_pose)

        show_poses([pred_pose.numpy(), gt_pose.numpy()], ['Pred', 'GT'],
                   config.skeleton, str(p_loss.numpy()) + '_' + str(inference_time))


if __name__ == "__main__":
    main()
