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


from network import get_model, get_train_params
from options import Options


def main():
    opt = Options()
    opt.parse_command()
    opt.print_items()

    # if setting gpu id, the using single GPU
    if opt.gpu:
        print('Single GPU Mode.')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set random seed
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    random.seed(opt.manual_seed)

    cudnn.benchmark = True

    if torch.cuda.device_count() > 1:
        print('Multi-GPUs Mode.')
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
    else:
        print('Single GPU Mode.')
        print("Let's use GPU:", opt.gpu)

    if opt.restore:
        assert os.path.isfile(opt.restore), \
            "=> no checkpoint found at '{}'".format(opt.restore)
        print("=> loading checkpoint '{}'".format(opt.restore))
        checkpoint = torch.load(opt.restore)

        start_iter = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        optimizer = checkpoint['optimizer']

        model = get_model(opt)
        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        del checkpoint  # clear memory
        # del model_dict
        torch.cuda.empty_cache()
    else:
        print("=> creating Model")
        model = get_model(opt)

        print("=> model created.")
        start_iter = 1
        best_result = None

        # different modules have different learning rate
        train_params = get_train_params(opt, model)
        optimizer = torch.optim.SGD(train_params, lr=opt.lr, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)

    if torch.cuda.device_count() == 1:
        from libs.trainers import single_gpu_trainer
        trainer = single_gpu_trainer.trainer(opt, model, optimizer, start_iter, best_result)
        trainer.train_eval()
    else:
        from libs.trainers import multi_gpu_trainer
        trainer = multi_gpu_trainer.trainer(opt, model, optimizer, start_iter, best_result)
        trainer.train_eval()


if __name__ == '__main__':
    main()
