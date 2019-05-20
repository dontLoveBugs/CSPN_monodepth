#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-19 15:53
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""


def create_loader(args, mode='train'):
    if args.dataset.lower() == 'nyu':
        from dataloaders.nyu_dataloader import create_loader
        return create_loader(args, mode=mode)
    elif args.dataset.lower() == 'kitti':
        return NotImplementedError
    else:
        return NotImplementedError
