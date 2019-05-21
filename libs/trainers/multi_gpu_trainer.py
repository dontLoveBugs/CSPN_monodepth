#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-20 18:33
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : multi_gpu_trainer.py
"""


import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders import create_loader
from libs import utils
from libs.criterion import get_criteria
from libs.metrics import AverageMeter, Result
from libs.scheduler import get_schedular, do_schedule
from network.encoding import DataParallelModel, DataParallelCriterion


class trainer(object):

    def __init__(self, args, model, optimizer, start_iter, best_result=None):
        self.opt = args
        self.original_model = model
        self.model = DataParallelModel(model).float().cuda()
        self.optimizer = optimizer
        self.scheduler = get_schedular(optimizer, args)
        self.criterion = DataParallelCriterion(get_criteria(args))

        self.output_directory = utils.get_save_path(args)
        self.best_txt = os.path.join(self.output_directory, 'best.txt')
        utils.write_config_file(args, self.output_directory)
        self.logger = utils.get_logger(self.output_directory)

        self.st_iter, self.ed_iter = start_iter, self.opt.max_iter

        self.train_loader = create_loader(self.opt, mode='train')
        self.eval_loader = create_loader(self.opt, mode='val')

        if best_result:
            self.best_result = best_result
        else:
            self.best_result = Result()
            self.best_result.set_to_worst()

        # train
        self.iter_save = len(self.train_loader)
        self.train_meter = AverageMeter()
        self.eval_meter = AverageMeter()
        self.metric = self.best_result.absrel
        self.result = Result()

        # batch size in each GPU
        self.ebt = self.opt.batch_size // torch.cuda.device_count()

    def train_iter(self, it):
        # Clear gradients (ready to accumulate)
        self.optimizer.zero_grad()

        end = time.time()

        try:
            input, target = next(loader_iter)
        except:
            loader_iter = iter(self.train_loader)
            input, target = next(loader_iter)

        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = self.model(input)  # @wx 注意输出

        loss = self.criterion(pred, target)
        loss.backward()  # compute gradient and do SGD step
        self.optimizer.step()

        gpu_time = time.time() - end

        # measure accuracy and record loss in each GPU
        for cuda_i in range(torch.cuda.device_count()):
            self.result.set_to_worst()
            _target = target[cuda_i * self.ebt: cuda_i * self.ebt + pred[cuda_i][0].size(0), ::].cuda(device=cuda_i)
            self.result.evaluate(pred[cuda_i][0], _target, loss.item())
            self.train_meter.update(self.result, gpu_time, data_time, pred[cuda_i][0].size(0))

        avg = self.train_meter.average()
        if it % self.opt.print_freq == 0:
            print('=> output: {}'.format(self.output_directory))
            print('Train Iter: [{0}/{1}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f}({average.loss:.5f}) '
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'REL={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                it, self.opt.max_iter, data_time=data_time,
                gpu_time=gpu_time, Loss=loss.item(), result=self.result, average=avg))

            self.logger.add_scalar('Train/Loss', avg.loss, it)
            self.logger.add_scalar('Train/RMSE', avg.rmse, it)
            self.logger.add_scalar('Train/rel', avg.absrel, it)
            self.logger.add_scalar('Train/Log10', avg.lg10, it)
            self.logger.add_scalar('Train/Delta1', avg.delta1, it)
            self.logger.add_scalar('Train/Delta2', avg.delta2, it)
            self.logger.add_scalar('Train/Delta3', avg.delta3, it)

    def eval(self, it):

        skip = len(self.eval_loader) // 9  # save images every skip iters
        self.eval_meter.reset()

        end = time.time()

        for i, (input, target) in enumerate(self.eval_loader):

            data_time = time.time() - end

            # compute output
            end = time.time()
            with torch.no_grad():
                pred = self.model(input)

            gpu_time = time.time() - end

            end = time.time()

            # measure accuracy and record loss
            # print(input.size(0))

            if input.size(0) == 1:
                self.result.set_to_worst()
                target = target.cuda()
                self.result.evaluate(pred[0][0], target)
                self.eval_meter.update(self.result, gpu_time, data_time, input.size(0))
            else:
                for cuda_i in range(torch.cuda.device_count()):
                    self.result.set_to_worst()
                    _target = target[cuda_i * self.ebt: cuda_i * self.ebt + pred[cuda_i][0].size(0), ::].cuda(device=cuda_i)
                    self.result.evaluate(pred[cuda_i][0], _target)
                    self.eval_meter.update(self.result, gpu_time, data_time, pred[cuda_i][0].size(0))

            if i % skip == 0:
                pred = pred[0][0]

                # save 8 images for visualization
                h, w = target.size(2), target.size(3)
                if h != pred.size(2) or w != pred.size(3):
                    pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)

                data = input[0]
                target = target[0]
                pred = pred[0]

            if self.opt.modality == 'd':
                img_merge = None
            else:
                if self.opt.modality == 'rgb':
                    rgb = data
                elif self.opt.modality == 'rgbd':
                    rgb = data[:3, :, :]
                    depth = data[3:, :, :]

                if i == 0:
                    if self.opt.modality == 'rgbd':
                        img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                    else:
                        img_merge = utils.merge_into_row(rgb, target, pred)

                elif (i < 8 * skip) and (i % skip == 0):
                    if self.opt.modality == 'rgbd':
                        row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                    else:
                        row = utils.merge_into_row(rgb, target, pred)
                    img_merge = utils.add_row(img_merge, row)
                elif i == 8 * skip:
                    filename = self.output_directory + '/comparison_' + str(it) + '.png'
                    utils.save_image(img_merge, filename)

            if (i + 1) % self.opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                      'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                      'REL={result.absrel:.2f}({average.absrel:.2f}) '
                      'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                      'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                      'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                      'Delta3={result.delta3:.3f}({average.delta3:.3f}) '.format(
                    i + 1, len(self.eval_loader), gpu_time=gpu_time, result=self.result,
                    average=self.eval_meter.average()))

        avg = self.eval_meter.average()

        self.logger.add_scalar('Test/RMSE', avg.rmse, it)
        self.logger.add_scalar('Test/rel', avg.absrel, it)
        self.logger.add_scalar('Test/Log10', avg.lg10, it)
        self.logger.add_scalar('Test/Delta1', avg.delta1, it)
        self.logger.add_scalar('Test/Delta2', avg.delta2, it)
        self.logger.add_scalar('Test/Delta3', avg.delta3, it)

        print('\n*\n'
              'RMSE={average.rmse:.3f}\n'
              'Rel={average.absrel:.3f}\n'
              'Log10={average.lg10:.3f}\n'
              'Delta1={average.delta1:.3f}\n'
              'Delta2={average.delta2:.3f}\n'
              'Delta3={average.delta3:.3f}\n'
              't_GPU={time:.3f}\n'.format(
            average=avg, time=avg.gpu_time))

    def train_eval(self):

        for it in tqdm(range(self.st_iter, self.ed_iter + 1), total=self.ed_iter - self.st_iter + 1,
                       leave=False, dynamic_ncols=True):
            self.model.train()
            self.train_iter(it)

            if it % self.iter_save == 0:
                self.model.eval()
                self.eval(it)

                self.metric = self.eval_meter.average().absrel
                train_avg = self.train_meter.average()
                eval_avg = self.eval_meter.average()

                self.logger.add_scalars('TrainVal/rmse',
                                        {'train_rmse': train_avg.rmse, 'test_rmse': eval_avg.rmse}, it)
                self.logger.add_scalars('TrainVal/rel',
                                        {'train_rel': train_avg.absrel, 'test_rmse': eval_avg.absrel}, it)
                self.logger.add_scalars('TrainVal/lg10',
                                        {'train_lg10': train_avg.lg10, 'test_rmse': eval_avg.lg10}, it)
                self.logger.add_scalars('TrainVal/Delta1',
                                        {'train_d1': train_avg.delta1, 'test_d1': eval_avg.delta1}, it)
                self.logger.add_scalars('TrainVal/Delta2',
                                        {'train_d2': train_avg.delta2, 'test_d2': eval_avg.delta2}, it)
                self.logger.add_scalars('TrainVal/Delta3',
                                        {'train_d3': train_avg.delta3, 'test_d3': eval_avg.delta3}, it)

                # save the change of learning_rate
                for i, param_group in enumerate(self.optimizer.param_groups):
                    old_lr = float(param_group['lr'])
                    self.logger.add_scalar('Lr/lr_' + str(i), old_lr, it)

                # remember best rmse and save checkpoint
                is_best = eval_avg.absrel < self.best_result.absrel
                if is_best:
                    self.best_result = eval_avg
                    with open(self.best_txt, 'w') as txtfile:
                        txtfile.write(
                            "Iter={}, rmse={:.3f}, rel={:.3f}, log10={:.3f}, d1={:.3f}, d2={:.3f}, dd31={:.3f}, "
                            "t_gpu={:.4f}".format(it, eval_avg.rmse, eval_avg.absrel, eval_avg.lg10,
                                                  eval_avg.delta1, eval_avg.delta2, eval_avg.delta3, eval_avg.gpu_time))

                # save checkpoint for each epoch
                utils.save_checkpoint({
                    'args': self.opt,
                    'epoch': it,
                    'state_dict': self.original_model.state_dict(),
                    'best_result': self.best_result,
                    'optimizer': self.optimizer,
                }, is_best, it, self.output_directory)

            # Update learning rate
            do_schedule(self.opt, self.scheduler, it=it, len=self.iter_save, metrics=self.metric)

        self.logger.close()
