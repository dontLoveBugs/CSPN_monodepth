#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-19 19:41
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : CSPN_ours.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class AffinityPropagate(nn.Module):

    def __init__(self, prop_time):
        super(AffinityPropagate, self).__init__()
        self.times = prop_time

    def forward(self, x, guided, sparse_depth=None):
        """
        :param x:        Feature maps, N,C,H,W
        :param guided:   guided Filter, N, K^2-1, H, W, K is kernel size
        :return:         returned feature map, N, C, H, W
        """

        B, C, H, W = guided.size()
        K = int(math.sqrt(C + 1))

        # 归一化
        guided = F.softmax(guided, dim=1)

        kernel = torch.zeros(B, C + 1, H, W, device=guided.device)
        kernel[:, 0:C // 2, :, :] = guided[:, 0:C // 2, :, :]
        kernel[:, C // 2 + 1:C + 1, :, :] = guided[:, C // 2:C, :, :]

        kernel = kernel.unsqueeze(dim=1).reshape(B, 1, K, K, H, W)

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()
            _x = x

        for _ in range(self.times):
            from network.libs.base.pac import conv2d
            x = conv2d(x, kernel, kernel_size=K, stride=1, padding=K // 2, dilation=1)

            if sparse_depth is not None:
                no_sparse_mask = 1 - sparse_mask
                x = sparse_mask * _x + no_sparse_mask * x
        return x
