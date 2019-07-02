# -*- coding: utf-8 -*-
"""
 @Time    : 2019/6/20 16:05
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

"""
@author: Xinjing Cheng, https://github.com/XinJCheng/CSPN/blob/master/models/cspn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffinityPropagate(nn.Module):

    def __init__(self, prop_time, prop_kernel):
        super(AffinityPropagate, self).__init__()
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        self.in_feature = 1
        self.out_feature = 1

    def forward(self, guidance, blur_depth, sparse_depth=None):

        # normalize features
        gate1_wb_cmb = torch.abs(guidance.narrow(1, 0, self.out_feature))
        gate2_wb_cmb = torch.abs(guidance.narrow(1, 1 * self.out_feature, self.out_feature))
        gate3_wb_cmb = torch.abs(guidance.narrow(1, 2 * self.out_feature, self.out_feature))
        gate4_wb_cmb = torch.abs(guidance.narrow(1, 3 * self.out_feature, self.out_feature))
        gate5_wb_cmb = torch.abs(guidance.narrow(1, 4 * self.out_feature, self.out_feature))
        gate6_wb_cmb = torch.abs(guidance.narrow(1, 5 * self.out_feature, self.out_feature))
        gate7_wb_cmb = torch.abs(guidance.narrow(1, 6 * self.out_feature, self.out_feature))
        gate8_wb_cmb = torch.abs(guidance.narrow(1, 7 * self.out_feature, self.out_feature))

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)

        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

        gate_wb = torch.cat((gate1_wb_cmb, gate2_wb_cmb, gate3_wb_cmb, gate4_wb_cmb,
                             gate5_wb_cmb, gate6_wb_cmb, gate7_wb_cmb, gate8_wb_cmb), 1)

        # pad input and convert to 8 channel 3D features
        raw_depht_input = blur_depth
        # blur_depht_pad = nn.ZeroPad2d((1,1,1,1))
        result_depth = blur_depth

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()

        for i in range(self.prop_time):

            # one propagation
            spn_kernel = self.prop_kernel
            result_depth = self.pad_blur_depth(result_depth)
            neigbor_weighted_sum = self.eight_way_propagation(gate_wb, result_depth, spn_kernel)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum
            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depht_input

        return result_depth

    def pad_blur_depth(self, blur_depth):
        # top pad
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        blur_depth_1 = left_top_pad(blur_depth).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        blur_depth_2 = center_top_pad(blur_depth).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        blur_depth_3 = right_top_pad(blur_depth).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        blur_depth_4 = left_center_pad(blur_depth).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        blur_depth_5 = right_center_pad(blur_depth).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        blur_depth_6 = left_bottom_pad(blur_depth).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        blur_depth_7 = center_bottom_pad(blur_depth).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        blur_depth_8 = right_bottm_pad(blur_depth).unsqueeze(1)

        result_depth = torch.cat((blur_depth_1, blur_depth_2, blur_depth_3, blur_depth_4,
                                  blur_depth_5, blur_depth_6, blur_depth_7, blur_depth_8), 1)
        return result_depth

    def eight_way_propagation(self, weight_matrix, blur_matrix, kernel):
        sum_conv_weight = torch.ones((1, 8, 1, kernel//2, kernel//2), device=weight_matrix.device)

        _weight_sum = F.conv3d(weight_matrix, sum_conv_weight)
        _total_sum = F.conv3d(weight_matrix * blur_matrix, sum_conv_weight)

        out = torch.div(_total_sum, _weight_sum)
        return out