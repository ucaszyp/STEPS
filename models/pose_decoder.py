from __future__ import absolute_import, division, print_function
import torch.nn as nn


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, stride=1):
        super(PoseDecoder, self).__init__()

        self.reduce = nn.Conv2d(num_ch_enc[-1], 256, 1)
        self.conv1 = nn.Conv2d(256, 256, 3, stride, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, stride, 1)
        self.axis_conv = nn.Conv2d(256, 3, 1)
        self.trans_conv = nn.Conv2d(256, 3, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_features):
        f = input_features[-1]
        out = self.relu(self.reduce(f))
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        axis_out = self.axis_conv(out).mean(3).mean(2) * 0.01
        trans_out = self.trans_conv(out).mean(3).mean(2) * 0.01
        axisangle = axis_out.view(-1, 1, 1, 3)
        translation = trans_out.view(-1, 1, 1, 3)
        return axisangle, translation
