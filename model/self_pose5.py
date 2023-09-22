from .resnet import ResNet, BasicBlock, Bottleneck
import torch
import torch.nn as nn


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
    def __init__(self, backbone="resnet50", pretrained=True):
        if backbone == "resnet18":
            block = BasicBlock
            layers = [2, 2, 2, 2]
            
        elif backbone == "resnet34":
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif backbone =="resnet50":
            block = Bottleneck
            layers = [3, 4, 6, 3]
        else:
            raise NotImplementedError
            
        super().__init__(block, layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    

class HMDecoder(nn.Module):
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
    
    
class Regressor(nn.Module):
    def __init__(self, joints=16):
        super().__init__()
        self.conv1 = DoubleConv(joints-1,64)
        self.conv2 = DoubleConv(64,128)
        self.conv3 = DoubleConv(128, 128, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l_relu = nn.LeakyReLU(0.2)
        
        self.fc1 = nn.Linear(15488, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 50)
        
        self.pose_fc1 = nn.Linear(50, 32)
        self.pose_fc2 = nn.Linear(32, 32)
        self.pose_fc3 = nn.Linear(32, joints * 3)
        
        self.rotation_fc1 = nn.Linear(50, 32)
        self.rotation_fc2 = nn.Linear(32, 32)
        self.rotation_fc3 = nn.Linear(32, joints * 3)
        
        self.hm_fc1 = nn.Linear(50, 512)
        self.hm_fc2 = nn.Linear(512, 2048)
        self.hm_fc3 = nn.Linear(2048, 18432)
        
        self.hm_decoder = HMDecoder(128, out_channels=joints-1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        
        pose = self.l_relu(self.pose_fc1(x))
        pose = self.l_relu(self.pose_fc2(pose))
        pose = self.l_relu(self.pose_fc3(pose))
        
        rotation = self.l_relu(self.rotation_fc1(x))
        rotation = self.l_relu(self.rotation_fc2(rotation))
        rotation = self.rotation_fc3(rotation)
        
        hm = self.l_relu(self.hm_fc1(x))
        hm = self.l_relu(self.hm_fc2(hm))
        hm = self.l_relu(self.hm_fc3(hm))
        hm = hm.view(hm.size(0),128,12,12)
        hm = self.hm_decoder(hm)
        
        return pose, hm, rotation
    
    def inference(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        
        pose = self.l_relu(self.pose_fc1(x))
        pose = self.l_relu(self.pose_fc2(pose))
        pose = self.l_relu(self.pose_fc3(pose))
        return pose
        
class SelfPose(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, feature_size=2048, joints=14):
        super().__init__()
        self.encoder = HMEncoder(backbone)
        self.decoder = HMDecoder(feature_size, joints-1)
        self.regressor = Regressor(joints)
    
    def forward(self, x):
        x = self.encoder(x)
        hm = self.decoder(x)
        pose, pred_hm, rot = self.regressor(hm)
        
        return pose, hm, pred_hm, rot
        
    def inference(self, x):
        x = self.encoder(x)
        hm = self.decoder(x)
        pose = self.regressor.inference(hm)
        return pose
        
    
