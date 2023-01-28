import torch
import torch.nn as nn

from .layers import UpConv3x3, Conv3x3


class DispDecoder(nn.Module):
    def __init__(self, num_ch_enc):
        """
        Initialize a disp decoder which have four output scales
        :param num_ch_enc: number of channels of encoder
        """
        super(DispDecoder, self).__init__()
        # set parameters
        self.num_ch_enc = num_ch_enc
        # components
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        # 4
        self.reduce4 = nn.Conv2d(self.num_ch_enc[4], 512, 1, bias=False)
        self.conv4 = Conv3x3(512, 512)
        self.up_conv4 = UpConv3x3(512, 512)
        self.disp_conv4 = nn.Conv2d(512, 1, 3, padding=1, padding_mode='reflect')
        # 3
        self.reduce3 = nn.Conv2d(self.num_ch_enc[3], 256, 1, bias=False)
        self.conv3 = Conv3x3(512, 256)
        self.up_conv3 = UpConv3x3(256 * 2, 256)
        self.disp_conv3 = nn.Conv2d(256, 1, 3, padding=1, padding_mode='reflect')
        # 2
        self.reduce2 = nn.Conv2d(self.num_ch_enc[2], 128, 1, bias=False)
        self.conv2 = Conv3x3(256, 128)
        self.up_conv2 = UpConv3x3(128 * 2, 128)
        self.disp_conv2 = nn.Conv2d(128, 1, 3, padding=1, padding_mode='reflect')
        # 1
        self.reduce1 = nn.Conv2d(self.num_ch_enc[1], 64, 1, bias=False)
        self.conv1 = Conv3x3(128, 64)
        self.up_conv1 = UpConv3x3(64 * 2, 64)
        self.disp_conv1 = nn.Conv2d(64, 1, 3, padding=1, padding_mode='reflect')
        # 0
        self.reduce0 = nn.Conv2d(self.num_ch_enc[0], 64, 1, bias=False)
        self.conv0 = Conv3x3(64, 64)
        self.up_conv0 = UpConv3x3(64 * 2, 64)
        self.disp_conv0 = nn.Conv2d(64, 1, 3, padding=1, padding_mode='reflect')

    def forward(self, in_features: list, frame_idx: int = 0):
        """
        Forward step
        :param in_features: features from shallow to deep
        :param frame_idx: index of frame
        :return:
        """
        assert isinstance(in_features, list)
        # get features
        f0, f1, f2, f3, f4 = in_features
        # forward
        # 4
        x4 = self.reduce4(f4)
        x4 = self.conv4(x4)
        x4 = self.leaky_relu(x4)
        x4 = self.up_conv4(x4)
        disp4 = torch.sigmoid(self.disp_conv4(x4))
        # 3
        s3 = self.reduce3(f3)
        x3 = self.conv3(x4)
        x3 = torch.cat([x3, s3], dim=1)
        x3 = self.leaky_relu(x3)
        x3 = self.up_conv3(x3)
        disp3 = torch.sigmoid(self.disp_conv3(x3))
        # 2
        s2 = self.reduce2(f2)
        x2 = self.conv2(x3)
        x2 = torch.cat([x2, s2], dim=1)
        x2 = self.leaky_relu(x2)
        x2 = self.up_conv2(x2)
        disp2 = torch.sigmoid(self.disp_conv2(x2))
        # 1
        s1 = self.reduce1(f1)
        x1 = self.conv1(x2)
        x1 = torch.cat([x1, s1], dim=1)
        x1 = self.leaky_relu(x1)
        x1 = self.up_conv1(x1)
        disp1 = torch.sigmoid(self.disp_conv1(x1))
        # 0
        s0 = self.reduce0(f0)
        x0 = self.conv0(x1)
        x0 = torch.cat([x0, s0], dim=1)
        x0 = self.leaky_relu(x0)
        x0 = self.up_conv0(x0)
        disp0 = torch.sigmoid(self.disp_conv0(x0))
        # pack and return
        outputs = {
            ('disp', frame_idx, 0): disp0,
            ('disp', frame_idx, 1): disp1,
            ('disp', frame_idx, 2): disp2,
            ('disp', frame_idx, 3): disp3,
            ('disp', frame_idx, 4): disp4
        }
        return outputs
