#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-19 15:29
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""


def get_model(opt):
    if opt.dataset.lower() == 'kitti':
        raise NotImplementedError
    elif opt.dataset.lower() == 'nyu':
        if opt.modality.lower() == 'rgb':
            raise NotImplementedError
        elif opt.modality.lower() == 'rgbd':
            if opt.arch.lower() == 'unet':
                from network.unet_ours import resnet50
                return resnet50(pretrained=True)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


def get_train_params(opt, model):
    return model.parameters()