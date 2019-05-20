#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-19 16:44
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : utils.py
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import string


# update pretrained model params according to my model params
def load_model_dict(my_model, pretrained_dict):
    my_model_dict = my_model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_model_dict}
    # 2. overwrite entries in the existing state dict
    my_model_dict.update(pretrained_dict)

    return my_model_dict


def update_conv_spn_model(out_dict, in_dict):
    in_dict = {k: v for k, v in in_dict.items() if k in out_dict}
    return in_dict