#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-20 16:52
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : main.py
"""

import os
import random
import numpy as np
import torch
from torch.backends import cudnn

from optoins import parse_command
from network import get_model, get_train_params


def main():
    args = parse_command()
    print(args)

    # if setting gpu id, the using single GPU
    if args.gpu:
        print('Single GPU Mode.')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    cudnn.benchmark = True

    if torch.cuda.device_count() > 1:
        print('Multi-GPUs Mode.')
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
    else:
        print('Single GPU Mode.')
        print("Let's use GPU:", args.gpu)

    if args.restore:
        assert os.path.isfile(args.restore), \
            "=> no checkpoint found at '{}'".format(args.restore)
        print("=> loading checkpoint '{}'".format(args.restore))
        checkpoint = torch.load(args.restore)

        start_iter = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        optimizer = checkpoint['optimizer']

        model = get_model(args)
        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        del checkpoint  # clear memory
        # del model_dict
        torch.cuda.empty_cache()
    else:
        print("=> creating Model")
        model = get_model(args)

        print("=> model created.")
        start_iter = 1
        best_result = None

        # different modules have different learning rate
        train_params = get_train_params(args, model)
        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if torch.cuda.device_count() == 1:
        from libs.trainers import single_gpu_trainer
        trainer = single_gpu_trainer.trainer(args, model, optimizer, start_iter, best_result)
        trainer.train_eval()
    else:
        from libs.trainers import multi_gpu_trainer
        trainer = multi_gpu_trainer.trainer(args, model, optimizer, start_iter, best_result)
        trainer.train_eval()


if __name__ == '__main__':
    from torchvision.models import resnet18
    main()
