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


class AffinityPropagate_prediction(nn.Module):
    """
    Args:
        affinity: Tensor, in [BatchSize, kernel_size**2-1, Height, Width] layout, the affinity matrix
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout, any kind of feature map, e.g. depth map, segmentation
        times: int, the iteration times

    Outputs:
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout,  feature after refined with affinity propagation

    """

    def __init__(self, kernel_size=3):
        super(AffinityPropagate_prediction, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.center_point = (kernel_size ** 2 - 1) // 2

    def forward(self, affinity, feature, times=24):
        b, c, h, w = affinity.shape

        # the the abs sum of all affinity matrix
        affinity_abs = affinity.abs()
        affinity_sum = affinity_abs.sum(dim=1, keepdim=True)

        affinity_norm = torch.div(affinity, affinity_sum)
        center = 1 - affinity_norm.sum(dim=1, keepdim=True)

        c_h, c_w = self.kernel_size // 2, self.kernel_size // 2

        for it in range(times):

            # corresponding to 8 directions and center point, we pad feature with 0, then we can move by index
            feature_pad = F.pad(feature, pad=(self.pad, self.pad, self.pad, self.pad))

            index = 0
            for pad_h in range(self.kernel_size):
                for pad_w in range(self.kernel_size):
                    if index == 0:
                        feature = feature_pad[:, :, pad_h:(h + pad_h), pad_w:(w + pad_w)] * affinity_norm[:, index:index + 1]
                    else:

                        if pad_h == c_h and pad_w == c_w:
                            feature += feature_pad[:, :, pad_h:(h + pad_h), pad_w:(w + pad_w)] * center
                            continue

                        feature += feature_pad[:, :, pad_h:(h + pad_h), pad_w:(w + pad_w)] * affinity_norm[:, index:index + 1]
                    index += 1
        return feature


class AffinityPropagate_completion(nn.Module):
    """
    Args:
        affinity: Tensor, in [BatchSize, kernel_size**2-1, Height, Width] layout, the affinity matrix
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout, any kind of feature map, e.g. depth map, segmentation
        times: int, the iteration times

    Outputs:
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout,  feature after refined with affinity propagation

    """

    def __init__(self, kernel_size=3, ):
        super(AffinityPropagate_completion, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.center_point = (kernel_size ** 2 - 1) // 2

    def forward(self, affinity, feature, sparse_depth, times=24):
        b, c, h, w = affinity.shape

        # the the abs sum of all affinity matrix
        affinity_abs = affinity.abs()
        affinity_sum = affinity_abs.sum(dim=1, keepdim=True)

        affinity_norm = torch.div(affinity_abs, affinity_sum)

        sparse_mask = sparse_depth.sign()
        feature = (1 - sparse_mask) * feature.clone() + sparse_mask * sparse_depth

        for it in range(times):

            # corresponding to 8 directions and center point, we pad feature with 1, then we can move by index
            feature_pad = F.pad(feature, pad=(self.pad, self.pad, self.pad, self.pad))

            index = 0
            for pad_h in range(self.kernel_size):
                for pad_w in range(self.kernel_size):
                    if index == 0:
                        feature = feature_pad[:, :, pad_h:(h + pad_h), pad_w:(w + pad_w)] * affinity_norm[:, index:index + 1]
                    else:
                        feature += feature_pad[:, :, pad_h:(h + pad_h), pad_w:(w + pad_w)] * affinity_norm[:, index:index + 1]
                    index += 1

            feature = (1 - sparse_mask) * feature.clone() + sparse_mask * sparse_depth
        return feature


class AffinityPropagate3D(nn.Module):
    """
    Args:
        affinity: Tensor, in [BatchSize, kernel_size**2-1, Height, Width] layout, the affinity matrix
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout, any kind of feature map, e.g. feat map, segmentation
        times: int, the iteration times

    Outputs:
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout,  feature after refined with affinity propagation

    """

    def __init__(self, kernel_size=3, ):
        super(AffinityPropagate3D, self).__init__()

    def forward(self, affinity, feature, times=24):
        b, c, d, h, w = affinity.shape

        # the the abs sum of all affinity matrix
        affinity_sum = 0
        for i in range(c):
            affinity_sum += affinity[:, i:i + 1, ].abs()

        # get the mask that sum of all affinity matrix >= 1
        mask_need_norm = affinity_sum.ge(1).float()
        affinity_norm = []

        # normalize the affinity matrix by sum
        for i in range(c):
            affinity_norm.append(torch.div(affinity[:, i:i + 1], affinity_sum))
        affinity_norm = torch.cat(affinity_norm, 1)

        # for the pixel satisfy sum < 1, use the original value, otherwise replaced by the normalized affinity matrix
        affinity01 = torch.zeros(b, c, d, h, w).type_as(affinity)
        for i in range(c):
            affinity01[:, i:i + 1] = torch.add(-mask_need_norm, 1) * affinity[:, i:i + 1] + \
                                     mask_need_norm * affinity_norm[:, i:i + 1]

        # according to Laplace diffusion theory, degree matrix = 1 - sum of affinity matrix
        degree_matrix = 1 - torch.sum(affinity01, dim=1, keepdim=True)

        # append the degree matrix to the last channel of affinity matrix
        # in [BatchSize, Channels+1, Height, Width] layout
        affinity01 = torch.cat((affinity01, degree_matrix), 1)

        for it in range(times):

            # corresponding to 8 directions and center point, we pad feature with 1, then we can move by index
            feature_pad = F.pad(feature, pad=(1, 1, 1, 1, 1, 1))

            index = 0
            for pad_d in [2, 0, 1]:
                for pad_h in [2, 0, 1]:
                    for pad_w in [2, 0, 1]:
                        if index == 0:
                            feature = feature_pad[:, :, pad_d:(d + pad_d), pad_h:(h + pad_h),
                                      pad_w:(w + pad_w)] * affinity01[:, index:index + 1]
                        else:
                            feature += feature_pad[:, :, pad_d:(d + pad_d), pad_h:(h + pad_h),
                                       pad_w:(w + pad_w)] * affinity01[:, index:index + 1]
                        index += 1

        return feature

