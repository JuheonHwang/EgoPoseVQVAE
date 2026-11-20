# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Experiment configuration.

``config`` is a module-level (initially empty) EasyDict that is filled **in
place** by :func:`load_config`, so ``from utils import config`` references held
by other modules stay valid after the script loads its YAML file.

Original author: Denis Tome'

"""
import copy
import os

import yaml
from easydict import EasyDict as edict

__all__ = ['config', 'load_config', 'read_config', 'SKELETONS']


# --------------------------------------------------------------------------
# Skeletons. The ORDER of the joints defines the joint ids (``jid``) and must
# not change: it determines tensor layout and the meaning of saved checkpoints.
# Entries are (name, parent, col).
# --------------------------------------------------------------------------

_HEAD_NECK = [
    ('Head', 'Neck', 5),
    ('Neck', 'Spine2', 5),
]

_ARMS = [
    ('LeftArm', 'LeftShoulder', 3),
    ('LeftForeArm', 'LeftArm', 3),
    ('LeftHand', 'LeftForeArm', 4),
    ('RightArm', 'RightShoulder', 1),
    ('RightForeArm', 'RightArm', 1),
    ('RightHand', 'RightForeArm', 2),
]

_LEGS_NO_TOES = [
    ('LeftUpLeg', 'Hips', 8),
    ('LeftLeg', 'LeftUpLeg', 8),
    ('LeftFoot', 'LeftLeg', 9),
]
_LEGS_NO_TOES_R = [
    ('RightUpLeg', 'Hips', 6),
    ('RightLeg', 'RightUpLeg', 6),
    ('RightFoot', 'RightLeg', 7),
]

_LEFT_TOE = ('LeftToeBase', 'LeftFoot', 9)
_RIGHT_TOE = ('RightToeBase', 'RightFoot', 7)

SKELETONS = {
    # 14 joints: no toes (SelfPose experiments 5 and 6, VQ-VAE pose layout)
    '14': _HEAD_NECK + _ARMS + _LEGS_NO_TOES + _LEGS_NO_TOES_R,
    # 16 joints: joints used in SelfPose / xR-EgoPose (experiments 1, 2, 4)
    '16': (_HEAD_NECK + _ARMS + _LEGS_NO_TOES + [_LEFT_TOE]
           + _LEGS_NO_TOES_R + [_RIGHT_TOE]),
    # 22 joints: full xR-EgoPose skeleton (experiment 3). Note that Head comes
    # first and Hips second, so jid('Hips') == 1 here (jid('Neck') == 1 in the
    # 14/16 joint skeletons).
    '22': [
        ('Head', 'Neck', 5),
        ('Hips', None, 0),
        ('Spine', 'Hips', 0),
        ('Spine1', 'Spine', 0),
        ('Spine2', 'Spine1', 0),
        ('Neck', 'Spine2', 5),
        ('LeftShoulder', 'Spine2', 0),
        ('LeftArm', 'LeftShoulder', 3),
        ('LeftForeArm', 'LeftArm', 3),
        ('LeftHand', 'LeftForeArm', 4),
        ('RightShoulder', 'Spine2', 0),
        ('RightArm', 'RightShoulder', 1),
        ('RightForeArm', 'RightArm', 1),
        ('RightHand', 'RightForeArm', 2),
    ] + _LEGS_NO_TOES + [_LEFT_TOE] + _LEGS_NO_TOES_R + [_RIGHT_TOE],
}


def set_skeleton(name):
    """Create skeleton definition

    Arguments:
        name {str} -- key of SKELETONS ('14', '16' or '22')

    Returns:
        dict -- hierarchy {joint name: {'parent', 'col', 'jid'}}
    """

    name = str(name)
    if name not in SKELETONS:
        raise ValueError('Unknown skeleton {!r}, choose one of {}'.format(
            name, sorted(SKELETONS)))

    joints = {}
    for jid, (jname, parent, col) in enumerate(SKELETONS[name]):
        joints[jname] = {'parent': parent, 'col': col, 'jid': jid}

    return joints


def set_action():
    """Clastering specialized actions in groups

    Returns:
        dict -- action names mapping
    """

    _NAMES = [
        'Gesticuling', 'Reacting', 'Greeting',
        'Talking', 'UpperStretching', 'Gaming',
        'LowerStretching', 'Patting', 'Walking', 'All'
    ]

    _ACTION = {
        'anim_Clip1': 8, 'Opening_A_Lid': 0, 'Dribble': 5, 'Boxing': 5,
        'Standing_Arguing__1_': 3, 'Happy': 3, 'Plotting': 3, 'Counting': 4,
        'Standing_Arguing': 0, 'Standing_2H_Cast_Spell_01': 4, 'Shooting_Gun': 5,
        'Two_Hand_Spell_Casting': 0, 'Shaking_Hands_2': 2, 'Hands_Forward_Gesture': 2,
        'Rifle_Punch': 1, 'Baseball_Umpire': 5, 'Angry_Gesture': 0, 'Waving_Gesture': 0,
        'Taunt_Gesture': 0, 'Golf_Putt_Failure': 5, 'Rejected': 1, 'Shake_Fist': 2,
        'Revealing_Dice': 5, 'Golf_Putt_Failure__1_': 5, 'No': 3, 'Angry_Point': 1,
        'Agreeing': 3, 'Sitting_Thumbs_Up': 6, 'Standing_Thumbs_Up': 4, 'Patting': 7,
        'Petting': 7, 'Petting_Animal': 7, 'Taking_Punch': 0,
        'Standing_1H_Magic_Attack_01': 4, 'Talking': 3, 'Standing_Greeting': 2,
        'Happy_Hand_Gesture': 0, 'Dismissing_Gesture': 1, 'Strong_Gesture': 1,
        'Pointing_Gesture': 1, 'Golf_Putt_Victory': 5, 'Pointing': 0,
        'Thinking': 4, 'Loser': 1, 'Reaching_Out': 3, 'Crazy_Gesture': 0,
        'Golf_Putt_Victory__1_': 5, 'Insult': 3, 'Arm_Gesture': 0,
        'Beckoning': 1, 'Charge': 5, 'Weight_Shift_Gesture': 8,
        'Pain_Gesture': 1, 'Fist_Pump': 0, 'Terrified': 1, 'Surprised': 1,
        'Clapping': 1, 'Rallying': 1, 'Hand_Raising': 0, 'Sitting_Disapproval': 6,
        'Quick_Formal_Bow': 2, 'Counting__1_': 0, 'Tpose_Take_001': 4,
        'upper_stretching': 4, 'lower_stretching': 6, 'walking': 8
    }

    action_map = {}
    for k, v in _ACTION.items():
        action_map[k] = _NAMES[v]

    return action_map


class _Config(edict):
    """EasyDict that fails with a helpful message while still empty."""

    def __getattr__(self, name):
        # only called when normal attribute lookup failed
        if not name.startswith('__') and not dict.__len__(self):
            raise AttributeError(
                "config.{} accessed before the configuration was loaded; "
                "call utils.load_config('configs/<name>.yml') first"
                .format(name))
        raise AttributeError(name)


def _reset(conf):
    """Empty a config in place. EasyDict mirrors keys in __dict__: clear both
    to avoid stale attributes. (A module-level function on purpose: EasyDict
    copies the public methods of subclasses into its keys.)"""

    dict.clear(conf)
    conf.__dict__.clear()


# Filled in place by load_config(); starts empty.
config = _Config()


def _set_dotted(conf, dotted, value):
    keys = dotted.split('.')
    node = conf
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def read_config(path, **overrides):
    """Read an experiment YAML file and return it as a *fresh* EasyDict.

    The global ``config`` is not touched, so this can be used to read a second
    configuration (e.g. the VQ-VAE one next to an EgoPose one).

    Arguments:
        path {str} -- YAML file, e.g. configs/model6.yml

    Keyword Arguments:
        overrides -- values replacing entries of the YAML file before they are
            resolved. Keys are dotted paths, e.g.
            ``read_config(p, **{'dataset.root': '/data/xr'})``.
            ``root`` is a shortcut for ``dataset.root`` (or ``amass.root`` for
            VQ-VAE configs). Overrides whose value is None are ignored, so
            unset CLI arguments can be passed straight through.

    Returns:
        EasyDict -- the resolved configuration
    """

    with open(path) as fin:
        conf = yaml.safe_load(fin)

    for key, value in overrides.items():
        if value is None:
            continue
        if key == 'root':
            key = 'dataset.root' if 'dataset' in conf else 'amass.root'
        _set_dotted(conf, key, value)

    ds = conf.get('dataset')
    if ds is not None:
        root = ds['root']
        for split in ('train', 'val', 'test'):
            if split in ds:
                ds[split] = os.path.join(root, ds[split])

    if 'skeleton' in conf:
        conf['skeleton'] = str(conf['skeleton'])
        conf['skel'] = set_skeleton(conf['skeleton'])
        norm = conf.get('transforms', {}).get('norm')
        if norm is not None and norm not in conf['skel']:
            raise ValueError('transforms.norm {!r} is not a joint of skeleton '
                             '{!r}'.format(norm, conf['skeleton']))

    conf['actions'] = set_action()

    return edict(conf)


def load_config(path, **overrides):
    """Load an experiment YAML file into the global ``config`` (in place).

    Same arguments as :func:`read_config`.

    Returns:
        EasyDict -- the (same) global config object
    """

    conf = read_config(path, **overrides)

    _reset(config)
    config.update(copy.deepcopy(conf))

    return config
