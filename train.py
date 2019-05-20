# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 15:25
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import time
import torch
import random

from torch.backends import cudnn
from tqdm import tqdm

from libs.metrics import AverageMeter, Result
from libs import utils
from libs.criterion import get_criteria
from libs.scheduler import get_schedular, do_schedule
import os
import torch.nn.functional as F
import numpy as np

from args import parse_command
from network import get_model, get_train_params
from network.encoding import DataParallelModel, DataParallelCriterion


def main():
    args = parse_command()
    print(args)

    # if setting gpu id, the using single GPU
    if args.gpu:
        print('Single GPU Mode.')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    best_result = Result()
    best_result.set_to_worst()

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

    from dataloaders import create_loader
    train_loader = create_loader(args, mode='train')
    val_loader = create_loader(args, mode='val')

    if args.restore:
        assert os.path.isfile(args.restore), \
            "=> no checkpoint found at '{}'".format(args.restore)
        print("=> loading checkpoint '{}'".format(args.restore))
        checkpoint = torch.load(args.restore)

        start_iter = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        optimizer = checkpoint['optimizer']

        model_ori = get_model(args)
        model_ori.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        del checkpoint  # clear memory
        # del model_dict
        torch.cuda.empty_cache()
    else:
        print("=> creating Model")
        model_ori = get_model(args)

        print("=> model created.")
        start_iter = 1

        # different modules have different learning rate
        train_params = get_train_params(args, model_ori)

        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # using the encoding parallel model
    model = DataParallelModel(model_ori)
    model.float()
    model = model.cuda()

    # when training, use reduceLROnPlateau to reduce learning rate
    scheduler = get_schedular(optimizer, args)

    # define loss function and using the parallel model
    criterion = get_criteria(args)
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    # create directory path, record file, logger
    output_directory = utils.get_save_path(args)
    best_txt = os.path.join(output_directory, 'best.txt')
    utils.write_config_file(args, output_directory)
    logger = utils.get_logger(output_directory)

    # train
    iter_save = len(train_loader)
    # iter_save =
    average_meter = AverageMeter()
    metric = best_result.silog
    result = Result()
    ebt = args.batch_size // torch.cuda.device_count()  # bacth_size in each single GPU
    model.train()

    for it in tqdm(range(start_iter, args.max_iter + 1), total=args.max_iter, leave=False, dynamic_ncols=True):
        # for it in range(1, args.max_iter + 1):
        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        end = time.time()

        try:
            input, target = next(loader_iter)
        except:
            loader_iter = iter(train_loader)
            input, target = next(loader_iter)

        # input, target = input.cuda(), target.cuda()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)  # @wx 注意输出

        loss = criterion(pred, target)
        loss.backward()  # compute gradient and do SGD step
        optimizer.step()

        gpu_time = time.time() - end

        # measure accuracy and record loss in each GPU
        for cuda_i in range(torch.cuda.device_count()):
            result.set_to_worst()
            _target = target[cuda_i * ebt: cuda_i * ebt + pred[cuda_i][0].size(0), ::].cuda(device=cuda_i)
            result.evaluate(pred[cuda_i][0], _target)
            average_meter.update(result, gpu_time, data_time, pred[cuda_i][0].size(0))

        if it % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Iter: [{0}/{1}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f} '
                  'SILog={result.silog:.2f}({average.silog:.2f}) '
                  'sqErrorRel={result.squared_rel:.2f}({average.squared_rel:.2f}) '
                  'absErrorRel={result.absrel:.2f}({average.absrel:.2f}) '
                  'iRMSE={result.irmse:.2f}({average.irmse:.2f}) '.format(
                it, args.max_iter, data_time=data_time,
                gpu_time=gpu_time, Loss=loss.item(), result=result, average=average_meter.average()))

            logger.add_scalar('Train/Loss', loss.item(), it)
            logger.add_scalar('Train/SILog', result.silog, it)
            logger.add_scalar('Train/sqErrorRel', result.squared_rel, it)
            logger.add_scalar('Train/absErrorRel', result.absrel, it)
            logger.add_scalar('Train/iRMSE', result.irmse, it)

        if it % iter_save == 0:
            epoch = it // iter_save - 1
            result, img_merge = validate(args, val_loader, model, epoch=epoch, logger=logger,
                                         output_directory=output_directory)
            metric = result.rmse

            train_avg = average_meter.average()

            logger.add_scalars('TrainVal/SILog',
                               {'train_SILog': train_avg.silog, 'test_SILog': result.silog}, epoch)
            logger.add_scalars('TrainVal/sqErrorRel',
                               {'train_sqErrorRel': train_avg.squared_rel, 'test_sqErrorRel': result.squared_rel}, epoch)
            logger.add_scalars('TrainVal/absErrorRel',
                               {'train_absErrorRel': train_avg.absrel, 'test_absErrorRel': result.absrel}, epoch)
            logger.add_scalars('TrainVal/iRMSE',
                               {'train_iRMSE': train_avg.irmse, 'test_iRMSE': result.irmse}, epoch)
            average_meter.reset()

            # save the change of learning_rate
            for i, param_group in enumerate(optimizer.param_groups):
                old_lr = float(param_group['lr'])
                logger.add_scalar('Lr/lr_' + str(i), old_lr, it)

            # remember best rmse and save checkpoint
            is_best = result.silog < best_result.silog
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write(
                        "epoch={}, SILog={:.2f}, sqErrorRel={:.2f}, absErrorRel={:.2f}, iRMSE={:.2f}, t_gpu={:.4f}".
                        format(it, result.silog, result.squared_rel, result.absrel, result.irmse, result.gpu_time))
                if img_merge is not None:
                    img_filename = output_directory + '/comparison_best.png'
                    utils.save_image(img_merge, img_filename)

            # save checkpoint for each epoch
            utils.save_checkpoint({
                'args': args,
                'epoch': it,
                'state_dict': model_ori.state_dict(),
                'best_result': best_result,
                'optimizer': optimizer,
            }, is_best, it, output_directory)

            # change to train mode
            model.train()

        # Update learning rate
        do_schedule(args, scheduler, it=it, len=iter_save, metrics=metric)

    logger.close()


# validation
def validate(args, val_loader, model, epoch, logger, output_directory):
    average_meter = AverageMeter()

    model.eval()  # switch to evaluate mode

    skip = len(val_loader) // 9  # save images every skip iters
    result = Result()
    ebt = args.batch_size // torch.cuda.device_count()  # bacth_size in each single GPU

    for i, (input, target) in enumerate(val_loader):

        end = time.time()
        # input, target = input.cuda(), target.cuda()

        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)

        gpu_time = time.time() - end

        # measure accuracy and record loss
        # print(input.size(0))

        if input.size(0) == 1:
            result.set_to_worst()
            target = target.cuda()
            result.evaluate(pred[0][0], target)
            average_meter.update(result, gpu_time, data_time, input.size(0))
        else:
            for cuda_i in range(torch.cuda.device_count()):
                result.set_to_worst()
                _target = target[cuda_i * ebt: cuda_i * ebt + pred[cuda_i][0].size(0), ::].cuda(device=cuda_i)
                result.evaluate(pred[cuda_i][0], _target)
                average_meter.update(result, gpu_time, data_time, pred[cuda_i][0].size(0))

        if i % skip == 0:
            pred = pred[0][0]

            # save 8 images for visualization
            h, w = target.size(2), target.size(3)
            if h != pred.size(2) or w != pred.size(3):
                pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)

            data = input[0]
            target = target[0]
            pred = pred[0]

        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = data
            elif args.modality == 'rgbd':
                rgb = data[:3, :, :]
                depth = data[3:, :, :]

            if i == 0:
                if args.modality == 'rgbd':
                    img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    img_merge = utils.merge_into_row(rgb, target, pred)

            elif (i < 8 * skip) and (i % skip == 0):
                if args.modality == 'rgbd':
                    row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8 * skip:
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i + 1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'SILog={result.silog:.2f}({average.silog:.2f}) '
                  'sqErrorRel={result.squared_rel:.2f}({average.squared_rel:.2f}) '
                  'absErrorRel={result.absrel:.2f}({average.absrel:.2f}) '
                  'iRMSE={result.irmse:.3f}({average.irmse:.3f}) '.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
          'SILog={average.silog:.2f}\n'
          'sqErrorRel={average.squared_rel:.2f}\n'
          'absErrorRel={average.absrel:.2f}\n'
          'iRMSE={average.irmse:.2f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    logger.add_scalar('Test/SILog', avg.silog, epoch)
    logger.add_scalar('Test/sqErrorRel', avg.squared_rel, epoch)
    logger.add_scalar('Test/absErrorRel', avg.absrel, epoch)
    logger.add_scalar('Test/iRMSE', avg.irmse, epoch)
    return avg, img_merge


if __name__ == '__main__':
    main()
