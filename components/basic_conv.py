import torch.nn as nn

from .util import get_norm_layer, get_non_linear


class Conv1x1(nn.Module):
    """
    Convolution layer with 1 kernel size, followed by non_linear layer
    """
    def __init__(self, in_channels, out_channels, bias=True, non_linear='elu'):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        if non_linear == 'elu':
            self.non_linear = nn.ELU(inplace=True)
        elif non_linear == 'relu':
            self.non_linear = nn.ReLU(inplace=True)
        elif non_linear == 'leaky_relu':
            self.non_linear = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('Not supported non linear function: {}.'.format(non_linear))

    def forward(self, x):
        out = self.conv(x)
        out = self.non_linear(out)
        return out


class Conv3x3(nn.Module):
    """
    Convolution layer with 3 kernel size, followed by non_linear layer
    """
    def __init__(self, in_channels, out_channels, padding_mode='reflect', bias=True, non_linear='elu'):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode=padding_mode, bias=bias)
        if non_linear == 'elu':
            self.non_linear = nn.ELU(inplace=True)
        elif non_linear == 'relu':
            self.non_linear = nn.ReLU(inplace=True)
        elif non_linear == 'leaky_relu':
            self.non_linear = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('Not supported non linear function: {}.'.format(non_linear))

    def forward(self, x):
        out = self.conv(x)
        out = self.non_linear(out)
        return out


class BasicConv(nn.Module):
    """
    Convolution layers with normalization and non_linear
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode='reflect',
                 norm_layer='batch_norm', non_linear='elu'):
        super(BasicConv, self).__init__()
        # conv
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False,
                              padding_mode=padding_mode)
        # norm
        self.norm = get_norm_layer(norm_layer, out_channels)
        # non_linear
        self.non_linear = get_non_linear(non_linear)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.non_linear(out)
        return out
