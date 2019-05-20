#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-19 15:53
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""


def create_loader(args, mode='train'):
    import os
    from dataloaders.path import Path
    root_dir = Path.db_root_dir(args.dataset)

    if mode.lower() == 'train':
        traindir = os.path.join(root_dir, 'train')

        if os.path.exists(traindir):
            print('Train dataset "{}" is existed!'.format(traindir))
        else:
            print('Train dataset "{}" is not existed!'.format(traindir))
            exit(-1)

        from dataloaders.nyu_dataloader import nyu_dataloader
        train_set = nyu_dataloader.NYUDataset(traindir, type='train')
        import torch
        if torch.cuda.device_count() > 1:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                       shuffle=True, drop_last=True, num_workers=args.workers,
                                                       pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=args.workers, pin_memory=True)

        return train_loader
    elif mode.lower() == 'val':

        valdir = os.path.join(root_dir, 'val')
        if os.path.exists(valdir):
            print('Val dataset "{}" is existed!'.format(valdir))
        else:
            print('Val dataset "{}" is not existed!'.format(valdir))
            exit(-1)

        from dataloaders.nyu_dataloader import nyu_dataloader
        val_set = nyu_dataloader.NYUDataset(valdir, type='val')

        import torch
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,
                                                 shuffle=False, num_workers=args.workers, pin_memory=True)
        return val_loader
    else:
        raise NotImplementedError
