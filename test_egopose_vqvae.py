"""Compare SelfPose predictions with and without the VQ-VAE refinement, e.g.

    python test_egopose_vqvae.py --config configs/model6.yml --checkpoint EGOPOSE_CKPT \\
        --vqvae-config configs/vqvae1.yml --vqvae-checkpoint VQVAE_CKPT
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
from dataset import Mocap  # noqa: E402
from model.pose_VQVAE import build_pose_vqvae  # noqa: E402
from model.self_pose import build_self_pose  # noqa: E402
from utils import config, load_config, read_config  # noqa: E402
from utils.runtime import get_device  # noqa: E402
from utils.visualize import show_poses  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Test SelfPose + VQ-VAE')
    parser.add_argument('--config', required=True, help='EgoPose config, e.g. configs/model6.yml')
    parser.add_argument('--checkpoint', required=True, help='SelfPose state dict written by train.py')
    parser.add_argument('--vqvae-config', required=True, help='configs/vqvaeN.yml')
    parser.add_argument('--vqvae-checkpoint', required=True, help='state dict written by train_vqvae.py')
    parser.add_argument('--data-root', help='xR-EgoPose root (TestSet/); overrides dataset.root')
    parser.add_argument('--gpu', help='GPU id(s) for CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def main():
    args = parse_args()
    # the global config holds the EgoPose configuration (dataset + transforms
    # read it); the VQ-VAE one is read separately
    load_config(args.config, **{'dataset.root': args.data_root})
    vq_config = read_config(args.vqvae_config)

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    test_data = Mocap(config.dataset.test, SetType.TEST, transform=data_transform)
    test_loader = DataLoader(
        test_data,
        batch_size=config.val_data_loader.batch_size,
        shuffle=False)

    device = get_device()

    model = build_self_pose(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    vqvae = build_pose_vqvae(vq_config)
    vqvae.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device))
    vqvae.to(device)
    vqvae.eval()

    pose_loss = torch.nn.MSELoss()
    n_joints = len(config.skel)

    for image, _, gt_pose, *_ in test_loader:
        image = image.to(device).float()
        gt_pose = gt_pose.to(device).float()

        start = time.time()
        with torch.no_grad():
            pred_pose1 = model.inference(image).view(-1, n_joints, 3)
            pred_pose2, _, _ = vqvae(pred_pose1)
        inference_time = time.time() - start
        pred_pose1 = pred_pose1.cpu()
        pred_pose2 = pred_pose2.cpu().view(-1, n_joints, 3)
        gt_pose = gt_pose.cpu().view(-1, n_joints, 3)

        p_loss1 = pose_loss(pred_pose1, gt_pose)
        p_loss2 = pose_loss(pred_pose2, gt_pose)

        show_poses([pred_pose1.numpy(), pred_pose2.numpy(), gt_pose.numpy()],
                   ['SelfPose', 'SelfPose+VQVAE', 'GT'], config.skeleton,
                   str(p_loss1.numpy()) + '_' + str(p_loss2.numpy()) + '_' + str(inference_time))


if __name__ == "__main__":
    main()
