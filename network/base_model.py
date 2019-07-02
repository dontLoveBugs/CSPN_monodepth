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

sys.path.append('../')

from network.libs.inplace_abn import InPlaceABNSync

affine_par = True
import functools

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)

        return x1, x2, x3, x4


"""
    load pretrained model
"""


def resnet101(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3])

    if pretrained:
        # saved_state_dict = torch.load('./network/pretrained_models/resnet101-imagenet.pth')
        saved_state_dict = torch.load('./pretrained_models/resnet101-imagenet.pth')
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

        model.load_state_dict(new_params)

    return model


class BasicModel(nn.Module):

    def __init__(self, pretrained=True):
        super(BasicModel, self).__init__()
        self.backbone = resnet101(pretrained=pretrained)

        self.head = None

    def forward(self, *input):
        pass


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

        valid_mask = target > 0.1

        # convert from meters to mm
        output_mm = 1e3 * output[valid_mask]
        target_mm = 1e3 * target[valid_mask]

        abs_diff = (output_mm - target_mm).abs()

        mse = (torch.pow(abs_diff, 2)).mean()
        rmse = torch.sqrt(mse)
        mae = abs_diff.mean()
        lg10 = (torch.log10(output_mm) - torch.log10(target_mm)).abs().mean()
        absrel = (abs_diff / target_mm).mean()
        squared_rel = ((abs_diff / target_mm) ** 2).mean()

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        delta1 = (maxRatio < 1.25).float().mean()
        delta2 = (maxRatio < 1.25 ** 2).float().mean()
        delta3 = (maxRatio < 1.25 ** 3).float().mean()

        # silog uses meters
        err_log = torch.log(target[valid_mask]) - torch.log(output[valid_mask])
        normalized_squared_log = (err_log ** 2).mean()
        log_mean = err_log.mean()
        silog = torch.sqrt(normalized_squared_log - log_mean * log_mean) * 100

        # convert from meters to km
        inv_output_km = (1e-3 * output[valid_mask]) ** (-1)
        inv_target_km = (1e-3 * target[valid_mask]) ** (-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        irmse = torch.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        imae = abs_inv_diff.mean()

        # return irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, silog, delta1, delta2, delta3
        out = torch.tensor([irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, silog, delta1, delta2, delta3],
                           dtype=torch.float, device=torch.cuda.current_device())

        # print('# eval 2')
        return out
