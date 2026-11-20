"""Visualise VQ-VAE reconstructions of AMASS or xR-EgoPose ground-truth poses, e.g.

    python test_vqvae.py --config configs/vqvae1.yml --checkpoint CKPT --data amass
    python test_vqvae.py --config configs/vqvae1.yml --checkpoint CKPT --data xr-egopose
"""
import argparse
import time

from utils.runtime import init_gpu
init_gpu()  # CUDA_VISIBLE_DEVICES must be set before torch is imported

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

import dataset.transform as trsf  # noqa: E402
from base import SetType  # noqa: E402
from dataset import AMASS, Mocap  # noqa: E402
from model.pose_VQVAE import build_pose_vqvae  # noqa: E402
from utils import config, load_config, read_config  # noqa: E402
from utils.runtime import get_device  # noqa: E402
from utils.visualize import show_poses  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Test the pose VQ-VAE')
    parser.add_argument('--config', required=True, help='configs/vqvaeN.yml')
    parser.add_argument('--checkpoint', required=True, help='state dict written by train_vqvae.py')
    parser.add_argument('--data', required=True, choices=['amass', 'xr-egopose'],
                        help='poses to reconstruct: AMASS test split or xR-EgoPose test set')
    parser.add_argument('--amass-root', help='AMASS root; overrides amass.root')
    parser.add_argument('--smpl-root', help='SMPL model directory; overrides amass.smpl_model_root')
    parser.add_argument('--egopose-config', default='configs/model6.yml',
                        help='EgoPose config (skeleton / normalisation) used with --data xr-egopose '
                        '(default: configs/model6.yml)')
    parser.add_argument('--data-root', help='xR-EgoPose root (TestSet/); overrides dataset.root of --egopose-config')
    parser.add_argument('--gpu', help='GPU id(s) for CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def main():
    args = parse_args()
    vq_config = read_config(args.config, **{
        'amass.root': args.amass_root,
        'amass.smpl_model_root': args.smpl_root})

    if args.data == 'amass':
        test_data = AMASS(vq_config.amass.root, split='test',
                          smpl_model_root=vq_config.amass.smpl_model_root)
        test_loader = DataLoader(
            test_data,
            batch_size=vq_config.test_data_loader.batch_size,
            shuffle=vq_config.test_data_loader.shuffle)
        skeleton = vq_config.skeleton
    else:
        # the global config must hold the EgoPose configuration (read by the
        # dataset and its transforms)
        load_config(args.egopose_config, **{'dataset.root': args.data_root})
        data_transform = transforms.Compose([
            trsf.ImageTrsf(),
            trsf.Joints3DTrsf(),
            trsf.ToTensor()])
        test_data = Mocap(config.dataset.test, SetType.TEST, transform=data_transform)
        test_loader = DataLoader(
            test_data,
            batch_size=config.val_data_loader.batch_size,
            shuffle=False)
        skeleton = config.skeleton

    device = get_device()
    model = build_pose_vqvae(vq_config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    pose_loss = torch.nn.MSELoss()
    n_joints = len(vq_config.skel)

    for batch in test_loader:
        gt_pose = (batch if args.data == 'amass' else batch[2]).to(device).float()

        start = time.time()
        with torch.no_grad():
            pred_pose, _, _ = model(gt_pose)
        inference_time = time.time() - start
        pred_pose = pred_pose.cpu().view(-1, n_joints, 3)
        gt_pose = gt_pose.cpu().view(-1, n_joints, 3)

        p_loss = pose_loss(pred_pose, gt_pose)

        show_poses([pred_pose.numpy(), gt_pose.numpy()], ['Pred', 'GT'],
                   skeleton, str(p_loss.numpy()) + '_' + str(inference_time))


if __name__ == "__main__":
    main()
