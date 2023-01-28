from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Backproject(nn.Module):
    def __init__(self, batch_size, height, width):
        super(Backproject, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = torch.from_numpy(self.id_coords)
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = torch.cat([self.pix_coords, self.ones], 1)

    def forward(self, depth, inv_K):

        #print(inv_K)
        #print(self.pix_coords)
        #inv_K= inv_K.to(torch.float32)
        #print(inv_K)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.cuda())
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones.cuda()], 1)
        return cam_points


class Project(nn.Module):
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # print("=========Project forward===========")
        # print('********K************')
        # print(K)
        # print(K.shape)    
        # print('********T************')
        # print(T)
        # print(T.shape)

        P = torch.matmul(K, T)[:, :3, :]
        
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class DeConv3x3(nn.Module):
    """
    Use transposed convolution to up sample (scale_factor = 2.0)
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(DeConv3x3, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                          output_padding=0, bias=bias)
        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))
        self.non_linear = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.up_conv(x)
        out = self.pad(out)
        out = self.non_linear(out)
        return out


class UpConv3x3(nn.Module):
    """
    Use bilinear followed by conv
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(UpConv3x3, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels, bias=bias)
        self.non_linear = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0, mode='nearest')
        out = self.conv(out)
        out = self.non_linear(out)
        return out


class Conv3x3(nn.Module):
    """
    Convolution layer with 3 kernel size, followed by non_linear layer
    """
    def __init__(self, in_channels, out_channels, padding_mode='reflect', bias=True):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return out
