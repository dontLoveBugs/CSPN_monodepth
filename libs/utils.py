# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 20:57
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com

import glob
import os
import shutil
import socket
import torch
from datetime import datetime
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.jet


def get_save_path(args):
    save_dir_root = os.getcwd()
    save_dir_root = os.path.join(save_dir_root, 'result', args.dataset, args.arch)
    if args.restore:
        return args.restore[:-len(args.restore.split('/')[-1])]
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_logger(output_directory):
    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)
    return logger


def write_config_file(args, output_directory):
    config_txt = os.path.join(output_directory, 'config.txt')

    # write training parameters to config file
    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str)


# save checkpoint
def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    # rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    rgb = np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    # print(rgb.shape, depth_target_col.shape, depth_pred_col.shape)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def merge_rgb_depth_into_row(input, depth):
    rgb = np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
    depth_pred_cpu = np.squeeze(depth.data.cpu().numpy())

    d_min = np.min(depth_pred_cpu)
    d_max = np.max(depth_pred_cpu)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    # print(rgb.shape, depth_target_col.shape, depth_pred_col.shape)
    img_merge = np.hstack([rgb, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


# write feature maps
fmap = plt.cm.jet


def feature_map(feature, min=None, max=None):
    if min is None:
        min = np.min(feature)
    if max is None:
        max = np.max(feature)

    relative = (feature - min) / (max - min)

    return 255 * fmap(relative)[:, :, :3]


def merge_features_into_row(features, featuers_num=9):
    features_cpu = np.squeeze(features.cpu().numpy())
    # print(features_cpu.shape)
    f_min = np.min(features_cpu)
    f_max = np.max(features_cpu)

    f = []

    for i in range(featuers_num):
        f.append(feature_map(features_cpu[i], min=f_min, max=f_max))
    img_merge = np.hstack(f)

    return img_merge


def add_features_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_featues_map(img_merge, name):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(name)


def save_features(features, filename, features_num=9):
    features_cpu = np.squeeze(features.cpu().numpy())
    # print(features_cpu.shape)
    f_min = np.min(features_cpu)
    f_max = np.max(features_cpu)

    f = []

    for i in range(features_num):
        f.append(feature_map(features_cpu[i], min=f_min, max=f_max))

    # print('f shape:', f[0].shape)

    img_merge = np.hstack(f)

    # print('img_merge shape:', img_merge.shape)

    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
