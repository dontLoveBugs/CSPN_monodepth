#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-19 13:29
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : base_model.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import sys


"""
 The methods for evaluation packaged by nn.Module
"""


class EvaluationModule(nn.Module):

    def __init__(self, depth_coefficients=None):
        super(EvaluationModule, self).__init__()
        self.dc = depth_coefficients

    def forward(self, pred, target):
        pred = pred[0]  # pred is a list [pred, others(usually, multi scale pred)]
        h, w = target.size(2), target.size(3)

        # print('# eval 0')

        if self.dc:
            pred = F.softmax(pred, dim=1)
            pred = self.dc.serialize(pred)

        # print('# eval 1')
        # print('# eval:', pred.shape, target.shape)

        if h != pred.size(2) or w != pred.size(3):
            output = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)
        else:
            output = pred

        valid_mask = target > 0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        mse = (torch.pow(abs_diff, 2)).mean()
        rmse = torch.sqrt(mse)
        mae = abs_diff.mean()
        lg10 = (torch.log10(output) - torch.log10(target)).abs().mean()
        absrel = (abs_diff / target).mean()

        maxRatio = torch.max(output / target, target / output)
        delta1 = (maxRatio < 1.25).float().mean()
        delta2 = (maxRatio < 1.25 ** 2).float().mean()
        delta3 = (maxRatio < 1.25 ** 3).float().mean()
        # data_time = 0
        # gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        irmse = torch.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        imae = abs_inv_diff.mean()

        # return irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, silog, delta1, delta2, delta3
        out = torch.tensor([irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3],
                           dtype=torch.float, device=torch.cuda.current_device())

        # print('# eval 2')
        return out

