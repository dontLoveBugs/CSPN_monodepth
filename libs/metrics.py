# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 19:53
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import torch
import math
import numpy as np

import torch.nn.functional as F


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.squared_rel, self.silog = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

        self.loss = 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.squared_rel, self.silog = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

        self.loss = np.inf

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, silog, delta1, delta2, delta3,
               gpu_time, data_time, loss=None):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.squared_rel, self.silog = squared_rel, silog
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

        if loss:
            self.loss = loss

    def evaluate(self, output, target, loss=None):

        # print('target dim', target.dim())
        h, w = target.size(2), target.size(3)

        if h != output.size(2) or w != output.size(3):
            output = F.upsample(input=output, size=(h, w), mode='bilinear', align_corners=True)

        valid_mask = target > 0.1

        # convert from meters to mm
        output_mm = 1e3 * output[valid_mask]
        target_mm = 1e3 * target[valid_mask]

        abs_diff = (output_mm - target_mm).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        self.squared_rel = float(((abs_diff / target_mm) ** 2).mean())

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        # silog uses meters
        err_log = torch.log(target[valid_mask]) - torch.log(output[valid_mask])
        normalized_squared_log = (err_log ** 2).mean()
        log_mean = err_log.mean()
        self.silog = math.sqrt(normalized_squared_log - log_mean * log_mean) * 100

        # convert from meters to km
        inv_output_km = (1e-3 * output[valid_mask]) ** (-1)
        inv_target_km = (1e-3 * target[valid_mask]) ** (-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        if loss:
            self.loss = loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_squared_rel, self.sum_silog = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

        self.sum_loss = 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_squared_rel += n * result.squared_rel
        self.sum_lg10 += n * result.lg10
        self.sum_silog += n * result.silog
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time

        self.sum_loss += n * result.loss

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
            self.sum_silog / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count, self.sum_loss / self.count)
        return avg
