#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-07-02 10:37
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : adaptive_conv.py
"""


"""
    Reference: https://github.com/NVlabs/pacnet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from torch.nn.modules.utils import _pair
from torch._thnn import type2backend

import math
from numbers import Number

try:
    import pyinn as P

    has_pyinn = True
except ImportError:
    P = None
    has_pyinn = False
    pass


def nd2col(input_nd, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, transposed=False,
           use_pyinn_if_possible=False):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (output_padding,) * n_dims if isinstance(output_padding, Number) else output_padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    if transposed:
        assert n_dims == 2, 'Only 2D is supported for fractional strides.'
        w_one = input_nd.new_ones(1, 1, 1, 1)
        pad = [(k - 1) * d - p for (k, d, p) in zip(kernel_size, dilation, padding)]
        input_nd = F.conv_transpose2d(input_nd, w_one, stride=stride)
        input_nd = F.pad(input_nd, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        stride = _pair(1)
        padding = _pair(0)

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])
    # Use PyINN if possible (about 15% faster) TODO confirm the speed-up
    if n_dims == 2 and dilation == 1 and has_pyinn and torch.cuda.is_available() and use_pyinn_if_possible:
        output = P.im2col(input_nd, kernel_size, stride, padding)
    else:
        output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
        out_shape = (bs, nch) + tuple(kernel_size) + out_sz
        output = output.view(*out_shape).contiguous()
    return output


class Conv2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_size, stride=1, padding=0, dilation=1):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1 and kernel.size(1) != ch:
            raise ValueError('Incompatible input and kernel sizes.')
        ctx.input_size = in_sz
        ctx.kernel_size = _pair(kernel_size)
        ctx.kernel_ch = kernel.size(1)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.save_for_backward(input if ctx.needs_input_grad[1] else None,
                              kernel if ctx.needs_input_grad[0] else None)
        ctx._backend = type2backend[input.type()]

        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        output = cols.view(bs, ch, *kernel.shape[2:]) * kernel
        output = torch.einsum('ijklmn->ijmn', (output,))

        return output.clone()  # TODO check whether a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, kernel = ctx.saved_tensors
        grad_input = grad_kernel = None
        (bs, ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new()
            grad_im2col_output = torch.einsum('ijmn,izklmn->ijklmn', (grad_output, kernel))
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
                                                grad_im2col_output,
                                                grad_input,
                                                ctx.input_size[0], ctx.input_size[1],
                                                ctx.kernel_size[0], ctx.kernel_size[1],
                                                ctx.dilation[0], ctx.dilation[1],
                                                ctx.padding[0], ctx.padding[1],
                                                ctx.stride[0], ctx.stride[1])
        if ctx.needs_input_grad[1]:
            cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
            grad_kernel = torch.einsum('ijmn,ijklmn->ijklmn', (grad_output, cols))
            if ctx.kernel_ch == 1:
                grad_kernel = grad_kernel.sum(dim=1, keepdim=True)

        return grad_input, grad_kernel, None, None, None, None


def conv2d(input, kernel, kernel_size, stride=1, padding=0, dilation=1, native_impl=False):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    if native_impl:
        bs, in_ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        # im2col on input
        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

        # main computation
        im_cols *= kernel
        output = im_cols.view(bs, in_ch, -1, out_h, out_w).sum(dim=2, keepdim=False)
    else:
        output = Conv2dFn.apply(input, kernel, kernel_size, stride, padding, dilation)

    return output