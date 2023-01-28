import torch.nn as nn
import torchvision.models as models
import numpy as np

from components import load_pretrained_weights


class LeakyReluBottleneck(models.resnet.Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(LeakyReluBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation,
                                                  norm_layer)
        self.relu = nn.LeakyReLU(inplace=True)


class LeakyReluBasicBlock(models.resnet.BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(LeakyReluBasicBlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation,
                                                  norm_layer)
        self.relu = nn.LeakyReLU(inplace=True)


class ResNetWithoutPool(models.ResNet):
    def __init__(self, block, layers):
        super(ResNetWithoutPool, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def build_backbone(num_layers, pretrained=False):
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: LeakyReluBasicBlock, 50: LeakyReluBottleneck}[num_layers]
    model = ResNetWithoutPool(block_type, blocks)

    if pretrained:
        loaded = load_pretrained_weights('resnet{}'.format(num_layers), map_location='cpu')
        model.load_state_dict(loaded)
    return model


class DispEncoder(nn.Module):
    """
    Resnet without maxpool
    """
    def __init__(self, num_layers: int, pre_trained=True):
        super(DispEncoder, self).__init__()
        # make backbone
        backbone = build_backbone(num_layers, pre_trained)
        # blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        ])
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, x):
        # from shallow to deep
        features = [(x - 0.45) / 0.225]
        for block in self.blocks:
            features.append(block(features[-1]))
        return features[1:]
