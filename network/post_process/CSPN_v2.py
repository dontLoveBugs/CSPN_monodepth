# -*- coding: utf-8 -*-
"""
Written by youmi
Implementation of CSPN.
Reference:
    CSPN: https://github.com/XinJCheng/CSPN/blob/master/cspn.py
    SPN: https://github.com/Liusifei/pytorch_spn/blob/master/left_right_demo.py

Time:
    For 2D:
        As paper report, 4 iterations of CSPN on one 1024*768 image only takes 3.689ms
        For our implementation, it takes 2.934ms
        But for 24 iterations of our implementation, it takes 13.109ms
    For 3D:
        it mainly used to refine SPP module in PSMNet.
        SPP in PSMNet with the [BatchSize, 32, 256, 512] layout, after extending 4 pooling feature to 5D feature map,
            the layout becomes [BatchSize, 32, 4, 256, 512].
        In our implementation and Time consumption testing,
            input feature in [1, 32, 4, 256, 512] layout,
            affinity in [1, 26, 4, 256, 512] layout,
            it takes 714.010ms

        However, if we just concatenate 4 pooling feature,
            the layout becomes [BatchSize, 32*4, 256, 512].
        In our implementation and Time consumption testing,
            input feature in [1, 32*4, 256, 512] layout,
            affinity in [1, 8, 256, 512] layout,
            it takes 191.608ms
FrameWork: PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffinityPropagate(nn.Module):
    """
    Args:
        affinity: Tensor, in [BatchSize, kernel_size**2-1, Height, Width] layout, the affinity matrix
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout, any kind of feature map, e.g. depth map, segmentation
        times: int, the iteration times

    Outputs:
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout,  feature after refined with affinity propagation

    """

    def __init__(self, kernel_size=3, ):
        super(AffinityPropagate, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.center_point = (kernel_size ** 2 - 1) // 2

    def forward(self, affinity, feature, times=24):
        b, c, h, w = affinity.shape

        # the the abs sum of all affinity matrix
        affinity_abs = affinity.abs()
        affinity_sum = affinity_abs.sum(dim=1, keepdim=True)

        affinity_norm = torch.div(affinity_abs, affinity_sum)

        for it in range(times):

            # corresponding to 8 directions and center point, we pad feature with 1, then we can move by index
            feature_pad = F.pad(feature, pad=(self.pad, self.pad, self.pad, self.pad))

            index = 0
            for pad_h in range(self.kernel_size):
                for pad_w in range(self.kernel_size):
                    if index == 0:
                        feature = feature_pad[:, :, pad_h:(h + pad_h), pad_w:(w + pad_w)] * affinity_norm[:,
                                                                                            index:index + 1]
                    else:
                        feature += feature_pad[:, :, pad_h:(h + pad_h), pad_w:(w + pad_w)] * affinity_norm[:,
                                                                                             index:index + 1]
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


if __name__ == '__main__':
    import time
    import os
    import sys

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    kernel_size = 3
    # -------------------- 2D --------------------- #
    model = AffinityPropagate()
    model.cuda()
    # B, C, H, W = 1, kernel_size**2-1, 256, 512
    B, C, H, W = 1, 9, 1024, 768
    affinity = torch.rand(B, C, H, W).cuda()
    affinity.requires_grad = True
    feature = torch.rand(B, 1, H, W).cuda()
    feature.requires_grad = True

    # -------------------- 3D --------------------- #
    # model = AffinityPropagate3D()
    # model.cuda()
    # B, C, D, H, W = 1, kernel_size**3-1, 4, 256, 512
    # affinity = torch.rand(B, C, D, H, W).cuda()
    # affinity.requires_grad = True
    # feature = torch.rand(B, 32, D, H, W).cuda()
    # feature.requires_grad = True

    # --------------------------------------------- #

    loss = model(affinity, feature).mean()
    print(loss)
    loss.backward()

    sum_time = 0
    for i in range(20):
        start_time = time.time()
        model(affinity, feature)
        sum_time += (time.time() - start_time)
        print('time = %.7f' % (time.time() - start_time))
    print('average time = {:.7f}'.format(sum_time / 20))
