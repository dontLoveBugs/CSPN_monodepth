# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/2 11:16
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


def parse_command():
    modality_names = ['rgb', 'rgbd', 'd']
    schedular_names = ['poly_lr', 'reduce_lr']

    import argparse
    parser = argparse.ArgumentParser(description='MonoDepth')

    # model parameters
    parser.add_argument('--arch', default='ccnet', type=str)
    parser.add_argument('--restore', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('--freeze', default=True, type=bool)

    # training parameters
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 4)')


    # criterion parameters
    parser.add_argument('--criterion', default='l1', type=str)
    parser.add_argument('--dsn', default=False, type=bool, help='if true, using DSN criteria')

    # lr scheduler parameters
    parser.add_argument('--scheduler', default='reduce_lr', type=str, choices=schedular_names)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate (default 0.0001)')

    parser.add_argument('--factor', default=0.1, type=float, help='factor in ReduceLROnPlateau.')
    parser.add_argument('--lr_patience', default=2, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--max_iter', default=300000, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--decay_iter', default=10, type=int,
                        help='decat iter in PolynomialLR.')
    parser.add_argument('--gamma', default=0.9, type=float, help='gamma in PolynomialLR, MultiStepLR, ExponentialLR.')

    # optimizer parameters
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    # dataset
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names)
    parser.add_argument('--jitter', type=float, default=0.1, help='color jitter for images')
    parser.add_argument('--dataset', default='kitti', type=str,
                        help='dataset used for training, kitti and nyu is available')
    parser.add_argument('--val_selection', type=bool, default=True)

    # others
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--gpu', default=None, type=str, help='if not none, use Single GPU')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pass

