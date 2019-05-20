#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-19 15:29
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""


def get_model(args):
    if args.dataset.lower() == 'kitti':
        raise NotImplementedError
    elif args.dataset.lower() == 'nyu':
        if args.modality.lower() == 'rgb':
            from network.unet_cspn_nyu import resnet50_prediction
            return resnet50_prediction(pretrained=True)
        elif args.modality.lower() == 'rgbd':
            from network.unet_cspn_nyu import resnet50_completion
            return resnet50_completion(pretrained=True)
        else:
            raise NotImplementedError


def get_train_params(args, model):
    return model.parameters()