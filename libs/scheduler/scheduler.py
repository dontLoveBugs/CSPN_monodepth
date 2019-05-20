# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/15 17:54
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

        # print(self.decay_iter, self.max_iter, self.gamma, self.last_epoch)

    def get_lr(self):
        # print(self.last_epoch, self.decay_iter, self.max_iter)
        # print(self.last_epoch % self.decay_iter, self.last_epoch % self.max_iter)
        if self.last_epoch % self.decay_iter or self.last_epoch > self.max_iter:
            # print('keep lr')
            # return [base_lr for base_lr in self.optimizer.param_groups]
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            # print('Poly Testing: ')
            return [base_lr * factor for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(
        self, optimizer, scheduler, mode="linear", warmup_iters=100, gamma=0.2, last_epoch=-1
    ):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_iters:
            if self.mode == "linear":
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == "constant":
                factor = self.gamma
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs