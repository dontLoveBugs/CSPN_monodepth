#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-26 17:04
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : image_processor.py
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_filter(input, mean=False, direction='x', spilt=True):

    assert input.dim() == 4

    if mean:
        input = torch.mean(input, 1, True)

    if direction.lower() == 'x':
        weight_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            weight_x = weight_x.cuda(torch.cuda.current_device())

        grad_x = F.conv2d(input=input, weight=weight_x, padding=1)

        return grad_x
    elif direction.lower() == 'y':
        weight_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            weight_y = weight_y.cuda(torch.cuda.current_device())

        grad_y = F.conv2d(input=input, weight=weight_y, padding=1)

        return grad_y

    elif direction.lower() == 'xy':
        weight_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            weight_x = weight_x.cuda(torch.cuda.current_device())

        grad_x = F.conv2d(input=input, weight=weight_x, padding=1)

        weight_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            weight_y = weight_y.cuda(torch.cuda.current_device())

        grad_y = F.conv2d(input=input, weight=weight_y, padding=1)

        if spilt:
            return grad_x, grad_y

        grad = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
        return grad
    else:
        raise NotImplementedError

