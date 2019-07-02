# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/2 11:16
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


def parse_command():
    modality_names = ['rgb', 'rgbd', 'd']
    from dataloaders.nyu_dataloader.dense_to_sparse import UniformSampling, SimulatedStereo
    sparsifier_names = [x.name for x in [UniformSampling, SimulatedStereo]]
    schedular_names = ['poly_lr', 'reduce_lr']
    loss_type_names = ['none', 'ms', 'dsn', 'all']
    upsample_types = ['dgf', 'pac', 'djif', 'none']
    pretrained_choices = ['imagenet', 'vkitti']
    distance_types = ['si', 'sq', 'sl']

    import argparse
    parser = argparse.ArgumentParser(description='MonoDepth')

    # model parameters
    parser.add_argument('--arch', default='up', type=str)
    parser.add_argument('--restore', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('--pretrained', default='imagenet', type=str, choices=pretrained_choices,
                        help='pretrained model: vkitti, imagenet')
    parser.add_argument('--freeze', default=True, type=bool)
    parser.add_argument('--upt', default='none', choices=upsample_types,
                        help='upsample types, if none, do not upsample.')

    # training parameters
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 4)')

    # criterion parameters
    parser.add_argument('--criterion', default='l1', type=str)
    parser.add_argument('--loss_wrapper', default='none', type=str, choices=loss_type_names,
                        help='if true, using DSN criteria')
    parser.add_argument('--distance', default='si', choices=distance_types)

    # lr scheduler parameters
    parser.add_argument('--scheduler', default='reduce_lr', type=str, choices=schedular_names)
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default 0.0001)')

    parser.add_argument('--factor', default=0.2, type=float, help='factor in ReduceLROnPlateau.')
    parser.add_argument('--lr_patience', default=2, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--max_iter', default=200000, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--decay_iter', default=10, type=int,
                        help='decat iter in PolynomialLR.')
    parser.add_argument('--gamma', default=0.9, type=float, help='gamma in PolynomialLR, MultiStepLR, ExponentialLR.')

    # optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, choices=['adam', 'sgd'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    # dataset
    parser.add_argument('--dataset', default='nyu', type=str,
                        help='dataset used for training, kitti and nyu is available')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--jitter', type=float, default=0.1, help='color jitter for images')
    parser.add_argument('--val_selection', type=bool, default=True)
    parser.add_argument('--discretization', type=int, default=1, help='discretize depth using the given value.')

    # data sample strategy
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgbd', choices=modality_names)
    parser.add_argument('-s', '--num-samples', default=500, type=int, metavar='N',
                        help='number of sparse depth samples (default: 0)')
    parser.add_argument('--max-depth', default=-1.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
    parser.add_argument('--sparsifier', metavar='SPARSIFIER', default=UniformSampling.name, choices=sparsifier_names,
                        help='sparsifier: ' + ' | '.join(sparsifier_names) + ' (default: ' + UniformSampling.name + ')')

    # others
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--gpu', default=None, type=str, help='if not none, use Single GPU')
    parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()

    if args.modality == 'rgb' and args.num_samples != 0:
        print("number of samples is forced to be 0 when input modality is rgb")
        args.num_samples = 0
    if args.modality == 'rgb' and args.max_depth != 0.0:
        print("max depth is forced to be 0.0 when input modality is rgb/rgbd")
        args.max_depth = 0.0

    return args


# TODO:
class Options(object):

    def __init__(self):
        # model parameters
        self.arch = None
        self.restore = None
        self.pretrained = None

        self.upt = None

        # training parameters
        self.batch_size = None

        # criterion parameters
        self.criterion = None
        self.loss_wrapper = None
        self.distance = None

        # lr scheduler parameters
        self.scheduler = None
        self.lr = None

        self.factor = None
        self.lr_patience = None
        self.max_iter = None
        self.decay_iter = None
        self.gamma = None

        # optimizer paramters
        self.opt = None
        self.momentum = None
        self.weight_decay = None

        # dataset paramters
        self.dataset = None
        self.workers = None
        self.jitter = None
        self.val_selection = None
        self.discretization = None

        # data sample strategy
        self.modality = None
        self.num_samples = None
        self.max_depth = None
        self.sparsifier = None

        # others
        self.manual_seed = None
        self.gpu = None
        self.print_freq = None

    def parse_command(self):
        args = parse_command()

        self.arch = args.arch
        self.restore = args.restore
        self.pretrained = args.pretrained

        self.upt = args.upt

        # training parameters
        self.batch_size = args.batch_size

        # criterion parameters
        self.criterion = args.criterion
        self.loss_wrapper = args.loss_wrapper
        self.distance = args.distance

        # lr scheduler parameters
        self.scheduler = args.scheduler
        self.lr = args.lr

        self.factor = args.factor
        self.lr_patience = args.lr_patience
        self.max_iter = args.max_iter
        self.decay_iter = args.decay_iter
        self.gamma = args.gamma

        # optimizer paramters
        self.opt = args.opt
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay

        # dataset paramters
        self.dataset = args.dataset
        self.modality = args.modality
        self.num_samples = args.num_samples
        self.max_depth = args.max_depth
        self.sparsifier = args.sparsifier
        self.workers = args.workers
        self.jitter = args.jitter
        self.val_selection = args.val_selection
        self.discretization = args.discretization

        # others
        self.manual_seed = args.manual_seed
        self.gpu = args.gpu
        self.print_freq = args.print_freq

    def write_config(self, output_directory):
        import os
        config_txt = os.path.join(output_directory, 'options.txt')

        # write training parameters to config file
        if not os.path.exists(config_txt):
            with open(config_txt, 'w') as txtfile:
                out_str = self.__str__()
                txtfile.write(out_str)

    def print_items(self):
        print(self.__str__())

    def __str__(self):
        out_str = 'model parameters:\n'
        out_str += '  arch:' + str(self.arch) + '\n'
        out_str += '  restore model path:' + str(self.restore) + '\n'
        out_str += '  pretrained model type:' + str(self.pretrained) + '\n'
        out_str += '  upsample type:' + str(self.upt) + '\n'

        out_str += '\ntraining parameters:\n'
        out_str += '  batch size:' + str(self.batch_size) + '\n'

        out_str += '\ncriterion parameters:\n'
        out_str += '  criterion:' + str(self.criterion) + '\n'
        out_str += '  loss wrapper:' + str(self.loss_wrapper) + '\n'
        out_str += '  metric distance type:' + str(self.distance) + '\n'

        out_str += '\nlr scheduler parameters:\n'
        out_str += '  lr:' + str(self.lr) + '\n'
        out_str += '  scheduler:' + str(self.scheduler) + '\n'
        out_str += '  factor:' + str(self.factor) + '\n'
        out_str += '  lr patience:' + str(self.lr_patience) + '\n'
        out_str += '  max iter:' + str(self.max_iter) + '\n'
        out_str += '  decay iter:' + str(self.decay_iter) + '\n'
        out_str += '  gamma:' + str(self.gamma) + '\n'

        out_str += '\noptimizer parameters:\n'
        out_str += '  opt:' + str(self.opt) + '\n'
        out_str += '  momentum:' + str(self.momentum) + '\n'
        out_str += '  weight decay:' + str(self.weight_decay) + '\n'

        out_str += '\ndataset parameters:\n'
        out_str += '  dataset:' + str(self.dataset) + '\n'
        out_str += '  workers:' + str(self.workers) + '\n'
        out_str += '  jitter:' + str(self.jitter) + '\n'
        out_str += '  val selection:' + str(self.val_selection) + '\n'
        out_str += '  discretization:' + str(self.discretization) + '\n'

        out_str += '\ndata sample strategy:\n'
        out_str += '  modality:' + str(self.modality) + '\n'
        out_str += '  sparsifier:' + str(self.sparsifier) + '\n'
        out_str += '  num samples:' + str(self.num_samples) + '\n'
        out_str += '  max depth:' + str(self.max_depth) + '\n'

        out_str += '\nothers\n'
        out_str += '  manual seed:' + str(self.manual_seed) + '\n'
        out_str += '  gpu ids:' + str(self.gpu) + '\n'
        out_str += '  print freq:' + str(self.print_freq) + '\n'

        return out_str


if __name__ == '__main__':
    import torch

    # print(torch.cuda.current_device())

    opt = Options()
    opt.parse_command()
    import os

    opt.write_config(os.getcwd())

    # args = vars(args)
    # args = sorted(args.items(), key=lambda x:x[0])
    # print(args)

    # from dataloaders import create_loader
    #
    # #
    # train_loader = create_loader(args, mode='train')
    # val_loader = create_loader(args, mode='val')
    # test_loader = create_loader(args, mode='test')
    # #
    # # print('batch size:', args.batch_size)
    # # print('train nums:', len(train_loader))
    # # print('val nums:', len(val_loader))
    # # print('test nums:', len(test_loader))
    # #
    # print('...train loader ...')
    # import torch
    #
    # for i, data in enumerate(train_loader):
    #     img, depth = data
    #     depth = depth.to()
    #     print(img.shape, depth.shape)
    #     print(img)
    #     print(depth)
    #     print('max depth:', torch.max(depth), ' min depth:', torch.min(depth))
    #     break
    #
    # print('... val loader ...')
    # for i, data in enumerate(val_loader):
    #     img, depth = data
    #     print(img.shape, depth.shape)
    #     print('max depth:', torch.max(depth), ' min depth:', torch.min(depth))
    #     break
    #
    # print('... test loader ...')
    # for i, data in enumerate(test_loader):
    #     img = data
    #     print(img.shape)
    #     break
