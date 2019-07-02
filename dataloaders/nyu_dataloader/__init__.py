#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-19 15:53
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""


def create_loader(args, mode='train'):
    # Data loading code
    print('=> creating ', mode, ' loader ...')
    import os
    from dataloaders.path import Path
    root_dir = Path.db_root_dir(args.dataset)

    # sparsifier is a class for generating random sparse depth input from the ground truth
    import numpy as np
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    from dataloaders.nyu_dataloader.dense_to_sparse import UniformSampling
    from dataloaders.nyu_dataloader.dense_to_sparse import SimulatedStereo
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args.num_samples, max_depth=max_depth)

    from dataloaders.nyu_dataloader.nyu_dataloader import NYUDataset

    import torch
    if mode.lower() == 'train':
        traindir = os.path.join(root_dir, 'train')

        if os.path.exists(traindir):
            print('Train dataset "{}" is existed!'.format(traindir))
        else:
            print('Train dataset "{}" is not existed!'.format(traindir))
            exit(-1)
        train_dataset = NYUDataset(traindir, type='train',
                                    modality=args.modality, sparsifier=sparsifier)
        # worker_init_fn ensures different sampling patterns for each data loading thread
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id: np.random.seed(work_id))

        return train_loader

    elif mode.lower() == 'val':
        valdir = os.path.join(root_dir, 'val')
        if os.path.exists(valdir):
            print('Val dataset "{}" is existed!'.format(valdir))
        else:
            print('Val dataset "{}" is not existed!'.format(valdir))
            exit(-1)
        val_dataset = NYUDataset(valdir, type='val',
                                 modality=args.modality, sparsifier=sparsifier)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

        return val_loader

    else:
        raise NotImplementedError
