"""SelfPose: heatmap encoder/decoder followed by a pose + heatmap regressor.

One implementation covers all experiments:

    variant='full'   ResNet up to layer4, 2048-d features, deep heatmap decoders
                     (input 368x368 -> regressor fc1 in_features 15488).
    variant='light'  ResNet up to layer3 (layer4 is still *constructed* so that
                     the state_dict keeps its keys, but it is never run),
                     1024-d features and shallower decoders
                     (fc1 in_features 12800).
    rotation=True    adds a rotation head to the regressor (full variant only).

Attribute names (encoder, decoder, regressor, fc1, pose_fc1, hm_decoder, ...)
are part of the checkpoint format and must not be renamed.
"""
import torch.nn as nn

from .resnet import ResNet, BasicBlock, Bottleneck


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class HMEncoder(ResNet):
    """ResNet feature extractor.

    The light variant stops after layer3. layer4/avgpool/fc/regression_head are
    still created by ResNet.__init__ so state_dict keys are the same for both
    variants.
    """

    def __init__(self, backbone="resnet50", variant="full"):
        if backbone == "resnet18":
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif backbone == "resnet34":
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif backbone == "resnet50":
            block = Bottleneck
            layers = [3, 4, 6, 3]
        else:
            raise NotImplementedError

        super().__init__(block, layers)
        self.use_layer4 = variant == "full"
        # channels of the feature map this encoder outputs
        self.out_channels = (512 if self.use_layer4 else 256) * block.expansion

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.use_layer4:
            x = self.layer4(x)
        return x


class HMDecoder(nn.Module):
    """Full heatmap decoder (2 deconvs, 3 double convs)."""

    def __init__(self, feature_size=2048, out_channels=15):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(feature_size, feature_size // 2, kernel_size=3, stride=2)
        self.conv1 = DoubleConv(feature_size // 2, feature_size // 2)
        self.deconv2 = nn.ConvTranspose2d(feature_size // 2, feature_size // 4, kernel_size=3, stride=2)
        self.conv2 = DoubleConv(feature_size // 4, feature_size // 8)
        self.conv3 = DoubleConv(feature_size // 8, out_channels)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LightHMDecoder(nn.Module):
    """Light heatmap decoder used after the encoder (1 deconv, 1 double conv)."""

    def __init__(self, feature_size=1024, out_channels=15):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(feature_size, feature_size // 4, kernel_size=3, stride=2)
        self.conv1 = DoubleConv(feature_size // 4, out_channels)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.conv1(x)
        return x


class LightHMDecoder2(nn.Module):
    """Light heatmap decoder used inside the regressor (kernel 5 first deconv)."""

    def __init__(self, feature_size=1024, out_channels=15):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(feature_size, feature_size // 2, kernel_size=5, stride=2)
        self.conv1 = DoubleConv(feature_size // 2, feature_size // 2)
        self.deconv2 = nn.ConvTranspose2d(feature_size // 2, feature_size // 4, kernel_size=3, stride=2)
        self.conv2 = DoubleConv(feature_size // 4, out_channels)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv2(x)
        return x


class Regressor(nn.Module):
    """Maps the predicted heatmaps to a pose (and optionally rotations) and
    reconstructs the heatmaps from the latent code (pred_hm)."""

    def __init__(self, joints=16, variant="full", rotation=False):
        super().__init__()
        if variant not in ("full", "light"):
            raise ValueError("variant must be 'full' or 'light', got {!r}".format(variant))
        if rotation and variant != "full":
            raise NotImplementedError("the rotation head only exists for variant='full'")

        self.variant = variant
        self.rotation = rotation

        self.conv1 = DoubleConv(joints - 1, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 128, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l_relu = nn.LeakyReLU(0.2)

        if variant == "full":
            self.fc1 = nn.Linear(15488, 2048)
            self.fc2 = nn.Linear(2048, 512)
            self.fc3 = nn.Linear(512, 50)

            self.pose_fc1 = nn.Linear(50, 32)
            self.pose_fc2 = nn.Linear(32, 32)
            self.pose_fc3 = nn.Linear(32, joints * 3)
            self._pose_fcs = ("pose_fc1", "pose_fc2", "pose_fc3")

            if rotation:
                self.rotation_fc1 = nn.Linear(50, 32)
                self.rotation_fc2 = nn.Linear(32, 32)
                self.rotation_fc3 = nn.Linear(32, joints * 3)

            self.hm_fc1 = nn.Linear(50, 512)
            self.hm_fc2 = nn.Linear(512, 2048)
            self.hm_fc3 = nn.Linear(2048, 18432)
            self._hm_size = 12

            self.hm_decoder = HMDecoder(128, out_channels=joints - 1)
        else:
            self.fc1 = nn.Linear(12800, 1024)
            self.fc2 = nn.Linear(1024, 256)
            self.fc3 = nn.Linear(256, 32)

            self.pose_fc1 = nn.Linear(32, 32)
            self.pose_fc2 = nn.Linear(32, joints * 3)
            self._pose_fcs = ("pose_fc1", "pose_fc2")

            self.hm_fc1 = nn.Linear(32, 512)
            self.hm_fc2 = nn.Linear(512, 2048)
            self.hm_fc3 = nn.Linear(2048, 12800)
            self._hm_size = 10

            self.hm_decoder = LightHMDecoder2(128, out_channels=joints - 1)

    def _latent(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        return x

    def _pose(self, z):
        # NOTE: the original code also applies LeakyReLU after the last pose fc
        pose = z
        for name in self._pose_fcs:
            pose = self.l_relu(getattr(self, name)(pose))
        return pose

    def forward(self, x):
        z = self._latent(x)
        pose = self._pose(z)

        outputs = [pose]

        hm = self.l_relu(self.hm_fc1(z))
        hm = self.l_relu(self.hm_fc2(hm))
        hm = self.l_relu(self.hm_fc3(hm))
        hm = hm.view(hm.size(0), 128, self._hm_size, self._hm_size)
        outputs.append(self.hm_decoder(hm))

        if self.rotation:
            # no activation after the last rotation fc
            rot = self.l_relu(self.rotation_fc1(z))
            rot = self.l_relu(self.rotation_fc2(rot))
            outputs.append(self.rotation_fc3(rot))

        return tuple(outputs)

    def inference(self, x):
        return self._pose(self._latent(x))


class SelfPose(nn.Module):
    """
    Arguments:
        joints {int} -- number of skeleton joints (16, 14, 22, ...)
        variant {str} -- 'full' or 'light'
        rotation {bool} -- add the rotation head (full variant only)
        backbone {str} -- resnet18 | resnet34 | resnet50

    forward(x)    -> (pose, hm, pred_hm) or (pose, hm, pred_hm, rot)
    inference(x)  -> pose
    """

    def __init__(self, joints=16, variant="full", rotation=False, backbone="resnet50"):
        super().__init__()
        if variant not in ("full", "light"):
            raise ValueError("variant must be 'full' or 'light', got {!r}".format(variant))

        self.rotation = rotation
        self.encoder = HMEncoder(backbone, variant)
        feature_size = self.encoder.out_channels
        if variant == "full":
            self.decoder = HMDecoder(feature_size, joints - 1)
        else:
            self.decoder = LightHMDecoder(feature_size, joints - 1)
        self.regressor = Regressor(joints, variant, rotation)

    def forward(self, x):
        x = self.encoder(x)
        hm = self.decoder(x)
        pose, pred_hm, *rot = self.regressor(hm)

        return (pose, hm, pred_hm, *rot)

    def inference(self, x):
        x = self.encoder(x)
        hm = self.decoder(x)
        return self.regressor.inference(hm)


def build_self_pose(config):
    """Build SelfPose from a loaded config (see utils.load_config)."""

    model_cfg = config.model
    return SelfPose(
        joints=len(config.skel),
        variant=model_cfg.get('variant', 'full'),
        rotation=model_cfg.get('rotation', False),
        backbone=model_cfg.get('backbone', 'resnet50'))
