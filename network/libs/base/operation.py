#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-06-24 22:08
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : operation.py
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class SparseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, eps=1e-5):
        super(SparseConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.eps = eps

    def forward(self, x, mask):
        _, c, _, _ = x.size()
        x = x * mask
        x = self.conv(x)
        _mask = mask.repeat(1, c, 1, 1)
        # print('_mask shape:', _mask.shape)
        _mask = self.conv(_mask) + self.eps
        x = x / _mask
        mask = self.pool(mask)
        return x, mask


class SparseGuidedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, eps=1e-5):
        super(SparseGuidedConv2d, self).__init__()

        from network.libs.pac.pac import PacConv2d
        self.conv = PacConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.eps = eps

    def forward(self, x, mask, guide):
        _, c, _, _ = x.size()
        x = x * mask
        _x = x

        # precompute kernel
        kernel, output_mask = self.conv.compute_kernel(guide, mask=None)

        x = self.conv(x, guide, kernel_size=kernel, mask=output_mask)
        mask_rpt = mask.repeat(1, c, 1, 1)
        mask_conv = self.conv(mask_rpt.float(), guide, kernel_size=kernel, mask=output_mask) + self.eps

        x = x / mask_conv
        x[mask_rpt] = _x[mask_rpt]
        return x


class BatchNorm2d_Relu(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, activation_type='leaky_relu'):
        super(BatchNorm2d_Relu, self).__init__()
        self.batchnorm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                        track_running_stats=track_running_stats)

        if activation_type.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation_type.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError

    def forward(self, input):
        return self.activation(self.batchnorm(input))


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
        ?? original implementaion doesn't have relu activation function?
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36),
                 normlayer=BatchNorm2d_Relu):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   normlayer(inner_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            normlayer(inner_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            normlayer(inner_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            normlayer(inner_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            normlayer(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            normlayer(out_features),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = th.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle

if __name__ == '__main__':
    x = th.randn(1, 2, 9, 9)
    x3 = th.randn(1, 1, 9, 9)
    x3[x3 < 0] = 0
    mask = x3 > 0
    mask = mask.float()

    print(x3, mask)

    conv = SparseConv2d(1, 1, 3, 1, 1)

    y, maskout = conv(x3, mask)

    print('y:', y)

    print('mask1:', mask)
    print('mask2:', maskout)
