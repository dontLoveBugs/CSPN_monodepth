# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/2 18:09
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from torch.optim.lr_scheduler import ReduceLROnPlateau
from libs.scheduler.scheduler import PolynomialLR, WarmUpLR


def get_schedular(optimizer, args):
    if args.scheduler.lower() == 'poly_lr':
        scheduler = PolynomialLR(optimizer, max_iter=args.max_iter, decay_iter=args.decay_iter, gamma=args.gamma)
    elif args.scheduler.lower() == 'reduce_lr':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.lr_patience)
    else:
        print('ERROR in get_schedular: not implement the scheduler named as ', args.schedular.lower())
        raise NotImplementedError
    return scheduler


def do_schedule(args, scheduler, it=None, len=None, metrics=None):
    if args.scheduler.lower() == 'poly_lr':
        scheduler.step()
        # print('test')
    elif args.scheduler.lower() == 'reduce_lr':
        if it is None or len is None or metrics is None:
            print('ERROR in do_schedule: it is None or len is None, metrics is None.')
            raise RuntimeError
        if it % len == 0:
            epoch = it // len
            scheduler.step(epoch=epoch, metrics=metrics)
    else:
        print('ERROR in do_schedule: not implement the scheduler named as ', args.schedular.lower())
        raise NotImplementedError
