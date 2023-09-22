"""
From Pose2Mesh_RELEASE by hongsukchoi
data/AMASS/dataset.py
"""


import os
import os.path as osp
import numpy as np
import glob
import copy
import torch

from dataset.smpl import SMPL


class AMASS_TEST(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.debug = False

        self.data_path = data_path # '/hdd4/cmk/dataset/AMASS/AMASS'
        
        self.rearrange_indices = [10, 8, 11, 12, 13, 14, 15, 16, 4, 5, 6, 1, 2, 3]
        
        # SMPL joint set
        self.mesh_model = SMPL()
        self.smpl_root_joint_idx = self.mesh_model.root_joint_idx
        self.face_kps_vertex = self.mesh_model.face_kps_vertex
        self.smpl_vertex_num = 6890
        self.smpl_joint_num = 24
        self.smpl_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.smpl_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))

        # h36m skeleton
        self.human36_joint_num = 17
        self.human36_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        # self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        self.human36_root_joint_idx = self.human36_joints_name.index('Neck')
        self.joint_regressor_h36m = self.mesh_model.joint_regressor_h36m


        self.input_joint_name = 'human36'
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)

        self.datalist = self.load_data()

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def load_data(self):
        print('Load annotations of AMASS')
        sub_dataset_list = glob.glob(f'{self.data_path}/*')

        datalist = []
        for sub in sub_dataset_list:
            sub_name = sub.split('/')[-1]

            sampling_ratio = self.get_subsampling_ratio(sub_name.lower())
            seq_name_list = glob.glob(f'{sub}/*')
            for seq in seq_name_list:
                file_list = glob.glob(f'{seq}/*_poses.npz')
                for file in file_list:
                    # data load
                    data = np.load(file)
                    poses = data['poses']  # (frame_num, 156)
                    dmpls = data['dmpls'] # (frame_num, 8)
                    trans = data['trans']  # (frame_num, 3)
                    betas = data['betas']  # (16,)
                    gender = data['gender']  # male

                    for frame_idx in range(len(poses)):
                        if frame_idx % sampling_ratio != 0:
                            continue
                        # get vertex and joint coordinates
                        pose = poses[frame_idx:frame_idx+1, :72]
                        beta = betas[None, :10]
                        
                        smpl_param = {'pose': pose, 'shape': beta}
                        datalist.append({'smpl_param': smpl_param})

                if self.debug:
                    break

        return datalist

    def get_subsampling_ratio(self, dataset_name):
        if dataset_name == 'cmu':
            return 60  # 120 -> 10 fps
        elif dataset_name == 'mpi_mosh':
            return 60
        elif dataset_name == 'bmlrub':
            return 60
        elif dataset_name == 'bmlmovi':
            return 60
        else:
            return 60

    def get_smpl_coord(self, smpl_param):
        pose, shape = smpl_param['pose'], smpl_param['shape']
        # smpl_pose = torch.FloatTensor(pose).view(-1, 3)
        # smpl_shape = torch.FloatTensor(shape).view(1, -1)  # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_pose = torch.FloatTensor(pose)
        smpl_shape = torch.FloatTensor(shape)  # smpl parameters (pose: 72 dimension, shape: 10 dimension)

        # get mesh and joint coordinates / we can change the neutral to male or female
        smpl_mesh_coord, _ = self.mesh_model.layer['neutral'](smpl_pose, smpl_shape)

        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);

        # # meter -> milimeter # We use meter in our pose-VQVAE
        # smpl_mesh_coord *= 1000;
        # smpl_joint_coord *= 1000;

        return smpl_mesh_coord

    def get_joints_from_mesh(self, mesh):
        joint_coord = np.dot(self.joint_regressor_h36m, mesh)
        return joint_coord

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])

        # get smpl mesh, joints
        smpl_param = data['smpl_param']
        mesh_cam = self.get_smpl_coord(smpl_param)

        # regress coco joints
        joint_h36m = self.get_joints_from_mesh(mesh_cam)

        # root relative camera coordinate
        joint_h36m = joint_h36m - joint_h36m[self.human36_root_joint_idx, :]
        
        # 'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist' -> 'Head', 'Neck', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle'
        return joint_h36m[self.rearrange_indices, :]
        
