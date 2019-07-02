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
