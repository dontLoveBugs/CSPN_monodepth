#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-28 20:39
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : utils.py
"""


import torch


class DepthCoefﬁcient(object):

    def __init__(self, min=1, max=90, N=80):
        self.d_min = min
        self.d_max = max
        self.discretization = N
        self.interval = (self.d_max - self.d_min) / self.discretization

    def discretize(self, d):
        valid = d > 0.1
        c = d
        c[valid] = (d[valid] - self.d_min) // self.interval
        c[1 - valid] = -1

        # Depth Coefﬁcients for Depth Completion
        # b, _, h, w = d.size()
        # c = torch.zeros(b, self.discretization, h, w)
        return c  # c is the maximun coeffient and it is interger.

    def undiscretize(self, d):
        valid = d > -1
        d0 = d
        d0[valid] = self.interval * d[valid].float() + self.d_min
        d0[1-valid] = 0

        return d0

    def serialize(self, c):
        # TODO: 精细化

        # c = torch.argmax(c, dim=1)
        # print('c = ', c)
        # d = self.interval * c.float() + self.d_min

        b, k, h, w = c.size()
        d = torch.zeros((b, 1, h, w), device=c.device).float()

        for i in range(k):
            d += c[:, i, :, :] * (self.interval * float(i) + self.d_min)

        # mask0 = torch.argmax(c, dim=1, keepdim=True)
        #
        # valid1 = mask0 - 1 > -1
        # valid2 = mask0 + 1 < self.discretization
        #
        # mask1, mask2 = mask0, mask0
        # mask1[valid1] = mask1[valid1] - 1
        # mask2[valid2] = mask2[valid2] + 1
        #
        # p0 = c.gather(dim=1, index=mask0)
        # p1 = c.gather(dim=1, index=mask1)
        # p2 = c.gather(dim=1, index=mask2)
        #
        # d = (self.interval * mask0 + self.d_min) * p0
        # d += (self.interval * mask1 + self.d_min) * p1
        # d += (self.interval * mask2 + self.d_min) * p2
        #
        # d /= (p0 + p1 + p2)
        #
        # # print('d = ', d)
        # d = d.unsqueeze(1)
        return d  # d is the depth and it float.
