# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Data processing where Images and associated 3D joint positions
(optionally also joint rotations) are loaded, together with the
2D heatmaps used to supervise SelfPose.

Behaviour is driven by ``config.dataset`` (read when the dataset is built,
so utils.load_config() must have been called):

    center_crop   False: resize the full frame to 368x368
                  True : crop the central square first, then resize
    heatmap_size  side of the (square) ground truth heatmaps (51 or 47)
    rotation      also return the joint rotations as 6th element

@author: Denis Tome'

"""
import math
import os
from skimage import io as sio
import numpy as np
from base import BaseDataset
from utils import io, config
from PIL import Image


# 7x7 gaussian (sigma = 1) used to draw the joint heatmaps
_GAUSSIAN = np.array([1.23409802e-04, 1.50343915e-03, 6.73794700e-03, 1.11089963e-02,
                      6.73794700e-03, 1.50343915e-03, 1.23409802e-04, 1.50343915e-03,
                      1.83156393e-02, 8.20849985e-02, 1.35335281e-01, 8.20849985e-02,
                      1.83156393e-02, 1.50343915e-03, 6.73794700e-03, 8.20849985e-02,
                      3.67879450e-01, 6.06530666e-01, 3.67879450e-01, 8.20849985e-02,
                      6.73794700e-03, 1.11089963e-02, 1.35335281e-01, 6.06530666e-01,
                      1.00000000e+00, 6.06530666e-01, 1.35335281e-01, 1.11089963e-02,
                      6.73794700e-03, 8.20849985e-02, 3.67879450e-01, 6.06530666e-01,
                      3.67879450e-01, 8.20849985e-02, 6.73794700e-03, 1.50343915e-03,
                      1.83156393e-02, 8.20849985e-02, 1.35335281e-01, 8.20849985e-02,
                      1.83156393e-02, 1.50343915e-03, 1.23409802e-04, 1.50343915e-03,
                      6.73794700e-03, 1.11089963e-02, 6.73794700e-03, 1.50343915e-03,
                      1.23409802e-04]).reshape(7,7)


class Mocap(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    CM_TO_M = 100

    def __init__(self, db_path, set_type, transform=None,
                 center_crop=None, heatmap_size=None, rotation=None):
        """Init class

        Keyword Arguments (None -> take the value from config.dataset):
            center_crop {bool} -- crop the central square of the frame
            heatmap_size {int} -- side of the ground truth heatmaps
            rotation {bool} -- also return joint rotations
        """

        super().__init__(db_path, set_type, transform=transform)

        ds_cfg = config.dataset
        self.center_crop = ds_cfg.get('center_crop', False) if center_crop is None else center_crop
        self.heatmap_size = ds_cfg.get('heatmap_size', 51) if heatmap_size is None else heatmap_size
        self.rotation = ds_cfg.get('rotation', False) if rotation is None else rotation

    def index_db(self):

        return self._index_dir(self.path)

    def _index_dir(self, path):
        """Recursively add paths to the set of
        indexed files

        Arguments:
            path {str} -- folder path

        Returns:
            dict -- indexed files per root dir
        """

        indexed_paths = dict()
        sub_dirs, _ = io.get_subdirs(path)
        if set(self.ROOT_DIRS) <= set(sub_dirs):

            # get files from subdirs
            n_frames = -1

            # let's extract the rgba and json data per frame
            for sub_dir in self.ROOT_DIRS:
                d_path = os.path.join(path, sub_dir)
                _, paths = io.get_files(d_path)

                if n_frames < 0:
                    n_frames = len(paths)
                else:
                    if len(paths) != n_frames:
                        self.logger.error(
                            'Frames info in {} not matching other passes'.format(d_path))

                encoded = [p.encode('utf8') for p in paths]
                indexed_paths.update({sub_dir: encoded})

            return indexed_paths

        # initialize indexed_paths
        for sub_dir in self.ROOT_DIRS:
            indexed_paths.update({sub_dir: []})

        # check subdirs of path and merge info
        for sub_dir in sub_dirs:
            indexed = self._index_dir(os.path.join(path, sub_dir))

            for r_dir in self.ROOT_DIRS:
                indexed_paths[r_dir].extend(indexed[r_dir])

        return indexed_paths

    def _process_points(self, data):
        """Filter joints to select only a sub-set for
        training/evaluation

        Arguments:
            data {dict} -- data dictionary with frame info

        Returns:
            np.ndarray -- 2D joint positions, format (J x 2)
            np.ndarray -- 3D joint positions, format (J x 3)
            np.ndarray -- joint rotations, format (J x 3); only if self.rotation
        """

        p2d_orig = np.array(data['pts2d_fisheye']).T
        p3d_orig = np.array(data['pts3d_fisheye']).T
        joint_names = {j['name'].replace('mixamorig:', ''): jid
                       for jid, j in enumerate(data['joints'])}

        # ------------------- Filter joints -------------------

        p2d = np.empty([len(config.skel), 2], dtype=p2d_orig.dtype)
        p3d = np.empty([len(config.skel), 3], dtype=p2d_orig.dtype)
        if self.rotation:
            p_rot = np.empty([len(config.skel), 3], dtype=p2d_orig.dtype)

        for jid, j in enumerate(config.skel.keys()):
            p2d[jid] = p2d_orig[joint_names[j]]
            p3d[jid] = p3d_orig[joint_names[j]]
            if self.rotation:
                p_rot[jid] = data['joints'][joint_names[j]]["rot"]

        p3d /= self.CM_TO_M

        if self.rotation:
            return p2d, p3d, p_rot
        return p2d, p3d

    def _draw_gaussian(self, img, pt, sigma):
        tmpSize = int(math.ceil(3 * sigma))
        ul = [int(math.floor(pt[0] - tmpSize)), int(math.floor(pt[1] - tmpSize))]
        br = [int(math.floor(pt[0] + tmpSize)), int(math.floor(pt[1] + tmpSize))]

        if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
            return img

        g = _GAUSSIAN

        g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
        g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]

        img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
        img_y = [max(0, ul[1]), min(br[1], img.shape[0])]

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img

    @staticmethod
    def _center_square(width, height):
        """Side of the central square of a frame and its (half) margins."""

        new_length = width * (width <= height) + height * (height <= width)
        return new_length, (width - new_length) / 2, (height - new_length) / 2

    def _get_heatmap(self, p2d, shape, joints=15):
        """Draw one heatmap per joint.

        NOTE: ``p2d`` is converted to heatmap coordinates IN PLACE (the caller
        passes a view of the 2D joints, which therefore end up in heatmap
        coordinates in the returned sample). This is intentional: it is the
        behaviour of the original code and must be kept.
        """

        new_shape = (self.heatmap_size, self.heatmap_size)

        if self.center_crop:
            width, height = shape
            new_length, half_reduced_width, half_reduced_height = \
                self._center_square(width, height)

            p2d[:, 0] = ((p2d[:, 0] - half_reduced_width + 1) / new_length) * new_shape[0]
            p2d[:, 1] = ((p2d[:, 1] - half_reduced_height + 1) / new_length) * new_shape[1]
        else:
            p2d[:, 0] = (p2d[:, 0] / shape[0]) * new_shape[0]
            p2d[:, 1] = (p2d[:, 1] / shape[1]) * new_shape[1]

        p2d = p2d.astype(int)

        heatmap = np.zeros((joints, new_shape[0], new_shape[1]))
        for i in range(joints):
            heatmap[i] = self._draw_gaussian(heatmap[i], (p2d[i][1], p2d[i][0]), 1)

        return heatmap

    def __getitem__(self, index):

        # load image
        img_path = self.index['rgba'][index].decode('utf8')
        img = sio.imread(img_path)
        img = Image.fromarray(img)

        # read joint positions
        json_path = self.index['json'][index].decode('utf8')
        data = io.read_json(json_path)
        if self.rotation:
            p2d, p3d, p_rot = self._process_points(data)
        else:
            p2d, p3d = self._process_points(data)

        # get heatmap
        heatmap = self._get_heatmap(p2d[1:, ...], img.size, joints=len(p2d) - 1)

        if self.center_crop:
            width, height = img.size
            new_length, _, _ = self._center_square(width, height)

            left = (width - new_length) / 2
            top = (height - new_length) / 2
            right = (width + new_length) / 2
            bottom = (height + new_length) / 2

            img = img.crop((left, top, right, bottom))

        img = np.array(img.resize((368, 368), Image.BICUBIC)).astype(np.float32)
        img /= 255.0

        # get action name
        action = data['action']

        if self.transform:
            img = self.transform({'image': img})['image']
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = self.transform({'joints2D': p2d})['joints2D']
            heatmap = self.transform({'heatmap': heatmap})['heatmap']
            if self.rotation:
                p_rot = self.transform({'jointsRot': p_rot})['jointsRot']

        if self.rotation:
            return img, p2d, p3d, action, heatmap, p_rot
        return img, p2d, p3d, action, heatmap

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])
