# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 20:04
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.image_processor import sobel_filter


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        # print('target size:', target.size())
        # print('pred size:', pred.size())
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss


class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        loss = torch.sqrt(torch.mean(torch.abs(torch.log(real) - torch.log(fake)) ** 2))
        return loss


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        loss = torch.mean(torch.abs(10. * real - 10. * fake))
        return loss


class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()

    def forward(self, fake, real):
        assert fake.dim() == real.dim(), "inconsistent dimensions"

        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear', align_corners=True)

        valid_mask = (real > 0).detach()
        real = real[valid_mask]
        fake = fake[valid_mask]

        loss = torch.mean(torch.abs(torch.log(real) - torch.log(fake)))
        return loss


class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold

    def forward(self, real, fake):
        mask = real > 0
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        fake = fake * mask
        diff = torch.abs(real - fake)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 - delta ** 2, 0., -delta ** 2.) + delta ** 2
        part2 = part2 / (2. * delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        loss = torch.sqrt(torch.mean(torch.abs(10. * real - 10. * fake) ** 2))
        return loss


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        return torch.mean(torch.abs(target - pred))


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        assert grad_fake.dim() == grad_real.dim(), "inconsistent dimensions"

        # prod = (grad_fake[:, :, None, :] @ grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
        prod = grad_fake * grad_real
        fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))

        return 1 - torch.mean(prod / (fake_norm * real_norm))


class Criterion_No_DSN(nn.Module):
    '''
    No DSN : We don't need to consider other supervision for the model.
    '''

    def __init__(self, criterion=None):
        super(Criterion_No_DSN, self).__init__()
        self.criterion = criterion

    def forward(self, preds, target):
        h, w = target.size(2), target.size(3)

        if h != preds[0].size(2) or w != preds[0].size(3):
            scale_target = F.interpolate(input=target, size=preds[0].size()[-2:], mode='nearest')
        else:
            scale_target = target
        loss = self.criterion(preds[0], scale_target)

        return loss


class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider the other supervision for the model.
    '''

    def __init__(self, criterion=None):
        super(CriterionDSN, self).__init__()
        self.criterion = criterion

    def forward(self, preds, target):
        h, w = target.size(2), target.size(3)

        # print('dsn target size:', target.size())
        # print('dsn h = ', h)
        # print('dsn w = ', w)

        if h != preds[0].size(2) or w != preds[0].size(3):
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = preds[0]
        loss1 = self.criterion(scale_pred, target)

        if h != preds[1].size(2) or w != preds[1].size(3):
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = preds[1]
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2 * 0.4
