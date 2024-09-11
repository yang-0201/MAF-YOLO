#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F

# from yolov6.layers.damo_yolo import ConvBNAct,RepGFPN,Focus,SuperResStem, DepthGFPN, SPDepthGFPN, SPRepGFPN
# from yolov6.layers.tiny_nas_csp import TinyNAS_CSP, TinyNAS_CSP_2
from yolov6.layers.CoAtNet import CoAtNetMBConv,ConvGE,CoAtNetTrans, MBConv_block, CoAtTrans_block
# from yolov6.layers.focal_transformer import FocalTransformer_block
# from yolov6.layers.BotNet import BotNet
# from models.Models.research import BoT3
# from yolov6.layers.RTMDet import CSPNeXtLayer, RTM_SepBNHead, ConvModule,DepthwiseSeparableConv, ELAN_Depth,ELAN_Depth_S, CSPRepLayer
# from yolov6.layers.yolov7 import E_ELAN,ELAN_H,ELAN, SPPCSPC
class SiLU(nn.Module):
    '''Activation of SiLU'''
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    '''Normal Conv with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimConv(nn.Module):
    '''Normal Conv with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ConvWrapper(nn.Module):
    '''Wrapper for normal Conv with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False):
        super().__init__()
        self.block = Conv(in_channels, out_channels, kernel_size, stride, groups, bias)

    def forward(self, x):
        return self.block(x)


class SimConvWrapper(nn.Module):
    '''Wrapper for normal Conv with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = SimConv(in_channels, out_channels, kernel_size, stride, groups, bias)

    def forward(self, x):
        return self.block(x)


class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class SPPF(nn.Module):
    '''Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher'''
    def __init__(self, in_channels, out_channels, kernel_size=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RealVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False,
    ):
        super(RealVGGBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

    def forward(self, inputs):
        out = self.relu(self.se(self.bn(self.conv(inputs))))
        return out


class ScaleLayer(torch.nn.Module):

    def __init__(self, num_features, use_bias=True, scale_init=1.0):
        super(ScaleLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        init.constant_(self.weight, scale_init)
        self.num_features = num_features
        if use_bias:
            self.bias = Parameter(torch.Tensor(num_features))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, inputs):
        if self.bias is None:
            return inputs * self.weight.view(1, self.num_features, 1, 1)
        else:
            return inputs * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)


#   A CSLA block is a LinearAddBlock with is_csla=True
class LinearAddBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False, is_csla=False, conv_scale_init=1.0):
        super(LinearAddBlock, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.scale_conv = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.scale_1x1 = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)
        if in_channels == out_channels and stride == 1:
            self.scale_identity = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=1.0)
        self.bn = nn.BatchNorm2d(out_channels)
        if is_csla:     # Make them constant
            self.scale_1x1.requires_grad_(False)
            self.scale_conv.requires_grad_(False)
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

    def forward(self, inputs):
        out = self.scale_conv(self.conv(inputs)) + self.scale_1x1(self.conv_1x1(inputs))
        if hasattr(self, 'scale_identity'):
            out += self.scale_identity(inputs)
        out = self.relu(self.se(self.bn(out)))
        return out


class DetectBackend(nn.Module):
    def __init__(self, weights='yolov6s.pt', device=None, dnn=True):

        super().__init__()
        assert isinstance(weights, str) and Path(weights).suffix == '.pt', f'{Path(weights).suffix} format is not supported.'
        from yolov6.utils.checkpoint import load_checkpoint
        model = load_checkpoint(weights, map_location=device)
        stride = int(model.stride.max())
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, val=False):
        y, _ = self.model(im)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return y


class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None
        elif block == BotRep:
            self.conv1 = BotRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BotRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None
        elif block == BottleCSPRepResP1:
            self.conv1 = BottleCSPRepResP1(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BottleCSPRepResP1(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None


    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class DepthBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    def __init__(self, in_channels, out_channels, n=1, change_channel=False, block=RepVGGBlock,
                 basic_block=RepVGGBlock):
        super().__init__()
        self.change_channel = change_channel
        if change_channel:
            self.conv0 = Conv(in_channels, out_channels)
        self.conv1 = block(out_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, 5, act_layer=nn.ReLU)

    def forward(self, x):
        if self.change_channel:
            x = self.conv0(x)
        x0 = self.conv1(x)
        y = self.conv2(x0)
        return x + y


class DepthBlock1(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    def __init__(self, in_channels, out_channels, n=1, change_channel=False, block=RepVGGBlock,
                 basic_block=RepVGGBlock):
        super().__init__()
        self.change_channel = change_channel
        if change_channel:
            self.conv0 = Conv(in_channels, out_channels)
        self.conv1 = block(out_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, 5, act_layer=nn.ReLU)
        self.conv3 = block(out_channels, out_channels)
        self.conv4 = DepthwiseSeparableConv(out_channels, out_channels, 5, act_layer=nn.ReLU)

    def forward(self, x):
        if self.change_channel:
            x = self.conv0(x)
        x0 = self.conv1(x)
        y = self.conv2(x0)
        y1 = x + y
        y2 = self.conv3(y1)
        y3 = self.conv4(y2)
        y4 = y1 + y3
        return y4
class RepConvBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = RepVGGBlock(in_channels, out_channels)
        self.conv2 = ConvWrapper(in_channels, out_channels)
        self.block = nn.Sequential(*(RepConv(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = input + x
        if self.block is not None:
            x = self.block(x)
        return x
class RepConvBlock1(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = RepVGGBlock(in_channels, out_channels)
        self.conv3 = RepVGGBlock(in_channels, out_channels)
        self.conv2 = ConvWrapper(in_channels, out_channels)


    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv3(x)
        x = self.conv2(x)
        x = input + x
        return x

class RepConv(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = RepVGGBlock(in_channels, out_channels)
        self.conv2 = ConvWrapper(in_channels, out_channels)


    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + input
        return x
class BottleRep(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs






# class Conv_C3(nn.Module):
#     '''Standard convolution in BepC3-Block'''
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#     def forward_fuse(self, x):
#         return self.act(self.conv(x))
class Conv_C3(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class BepC3(nn.Module):
    '''Beer-mug RepC3 Block'''
    def __init__(self, in_channels, out_channels, n=1,block=RepVGGBlock, e=0.5, concat=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv_C3(in_channels, c_, 1, 1)
        self.cv2 = Conv_C3(in_channels, c_, 1, 1)
        self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvWrapper:
            self.cv1 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv2 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1, act=nn.SiLU())

        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)
        self.concat = concat
        if not concat:
            self.cv3 = Conv_C3(c_, out_channels, 1, 1)

    def forward(self, x):
        if self.concat is True:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))

class CSPRepResP1(nn.Module):
    '''Beer-mug RepC3 Block'''
    def __init__(self, in_channels, out_channels, n=1,block=ConvWrapper, e=0.5, concat=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv_C3(in_channels, c_, 1, 1)
        self.cv2 = Conv_C3(in_channels, c_, 1, 1)
        self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvWrapper:
            self.cv1 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv2 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1, act=nn.SiLU())

        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleCSPRepResP1, basic_block=block)
        self.concat = concat
        if not concat:
            self.cv3 = Conv_C3(c_, out_channels, 1, 1)

    def forward(self, x):
        if self.concat is True:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))

class BottleCSPRepResP1(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        # self.conv2 = basic_block(out_channels, out_channels)
        self.conv2 = Conv_C3(out_channels, out_channels, 3, 1, act=nn.SiLU())
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_C3(c1, c_, 1, 1)
        self.cv2 = Conv_C3(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
# class E_ELAN(nn.Module):
#     # Spatial pyramid pooling layer used in YOLOv3-SPP
#     def __init__(self, c1, c2, n=1, k=2):
#         super(E_ELAN, self).__init__()
#         c_ = c1//2 # hidden channels
#
#         self.conv1 = Conv(c1,c_,1,1)
#         self.conv2 = Conv(c1,c_,1,1)
#         self.conv3 = Conv(c_, c_, 3, 1)
#         self.conv4 = Conv(c_, c_, 3, 1)
#         self.conv5 = Conv(c_, c_, 3, 1)
#         self.conv6 = Conv(c_, c_, 3, 1)
#         self.conv7 = Conv(c_, c_, 3, 1)
#         self.conv8 = Conv(c_, c_, 3, 1)
#         self.conv9 = Conv(c_ * 5, c2, 1, 1)
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         x7 = self.conv7(x6)
#         x8 = self.conv8(x7)
#         x = torch.cat([x8, x6, x4, x2, x1], dim = 1)
#         x = self.conv9(x)
#         return x
class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

class Out(nn.Module):
    def __init__(self, k=2):
        super(Out, self).__init__()

    def forward(self, x):
        outputs = []
        for y in x:
            outputs.append(y)
        return outputs

class MP1(nn.Module):
    def __init__(self, c1,c2, idx = 1,block = ""):  #c1 = c2
        c_ = c2//2
        self.idx = idx
        super(MP1, self).__init__()
        self.mp = MP()
        if idx == 1:
            self.conv1 = Conv(c1, c_, 1, 1)
            self.conv2 = Conv(c1, c_, 1, 1)
            self.conv3 = Conv(c_, c_, 3, 2)
        elif idx == 2:
            self.conv1 = Conv(c2, c2, 1, 1)
            self.conv2 = Conv(c2, c2, 1, 1)
            self.conv3 = Conv(c2, c2, 3, 2)
        elif idx == 3:
            self.conv1 = Conv(c1, c_, 1, 1)
            if block == "qa":
                self.conv2 = QARepVGGBlock(c1, c_, 3, 2)
            else:
                self.conv2 = RepVGGBlock(c1, c_, 3, 2)
            # self.conv3 = Conv(c2, c2, 3, 2)
        elif idx == 5:
            self.conv1 = SimConv(c1,c2,1,1)
            self.conv2 = RepVGGBlock(c1,c2,3,2)


    def forward(self, input):
        if self.idx == 3:
            x1 = self.mp(input)
            x1 = self.conv1(x1)
            x2 = self.conv2(input)
            out = torch.cat([x1, x2], dim=1)
        elif self.idx == 5:
            x = self.mp(input)
            x1 = self.conv1(x)
            x2 = self.conv2(input)
            out = x1 + x2
        else:
            x1 = self.mp(input)
            x1 = self.conv1(x1)
            x2 = self.conv2(input)
            x2 = self.conv3(x2)
            if self.idx == 1:
                out = torch.cat([x1, x2], dim=1)

        return out
class MPRep(nn.Module):
    def __init__(self, c1,c2):  #c1 = c2
        c_ = c2//2
        super(MPRep, self).__init__()
        self.mp = MP()
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = RepVGGBlock(c1, c_, 3, 2)

    def forward(self, input):

        x1 = self.mp(input)
        x1 = self.conv1(x1)
        x2 = self.conv2(input)
        out = torch.cat([x1, x2], dim=1)
        return out
class Stem(nn.Module):
    def __init__(self,in_channels,out_channels,block=RepVGGBlock):
        super(Stem, self).__init__()
        self.stem = block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )
    def forward(self,x):
        x = self.stem(x)
        return x

class DWConv(Conv_C3):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=nn.SiLU()):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class DepthWiseConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True,act = nn.SiLU()):
        super(DepthWiseConv, self).__init__()
        if in_channels == out_channels:
            self.shortcut = False
        else:
            self.shortcut = False
        self.dw = DWConv(in_channels, in_channels, k=kernel_size,s = stride,act = act)
        # self.point = Conv(in_channels,out_channels,1)
    def forward(self,x):
        x = self.dw(x)

        return x
class MPRep(nn.Module):
    def __init__(self, c1, c2, idx=1, block=""):  # c1 = c2
        c_ = c2 // 2
        self.idx = idx
        super(MPRep, self).__init__()
        self.mp = MP()
        self.conv1 = Conv(c1, c_, 1, 1)

        self.conv2 = RepVGGBlock(c1, c_, 3, 2)
        # self.conv3 = Conv(c2, c2, 3, 2)

    def forward(self, input):
        x1 = self.mp(input)
        x1 = self.conv1(x1)
        x2 = self.conv2(input)
        out = torch.cat([x1, x2], dim=1)
        return out
class Head_Depth(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_Depth, self).__init__()

        self.stem = Conv(in_channels, out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = DepthWiseConv(out_channels, out_channels, kernel_size=5, stride=1)
        self.reg_conv = DepthWiseConv(out_channels, out_channels, kernel_size=5, stride=1)
        # self.cls_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
        # reg_conv0
        # self.reg_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
        self.prior_prob = 1e-2
        self.initialize_biases()
    def initialize_biases(self):


        b = self.cls_pred.bias.view(-1, )
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.cls_pred.weight
        w.data.fill_(0.)
        self.cls_pred.weight = torch.nn.Parameter(w, requires_grad=True)


        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self,x):
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv(cls_x)
        cls_output = self.cls_pred(cls_feat)
        cls_output = torch.sigmoid(cls_output)
        reg_feat = self.reg_conv(reg_x)
        reg_output = self.reg_pred(reg_feat)

        return x, cls_output, reg_output
class DepthBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize = 5,
                 expansion_depth = 1,
                 small_kersize = 3,
                 use_depthwise=True):
        super(DepthBottleneck, self).__init__()


        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 3)
        self.shortcut = shortcut
        if use_depthwise:
            from yolov6.layers.RTMDet import DepthwiseSeparableConv
            # self.conv2  = DepthWiseConv(in_channels=mid_channel, out_channels=out_channels, kernel_size=kersize, stride=1, act=nn.ReLU())
            self.conv2 = ReparamLargeKernelConv(in_channels=mid_channel, out_channels=out_channels,
                                                kernel_size=kersize, stride=1,groups=mid_channel,small_kernel=small_kersize)
            self.one_conv = Conv(mid_channel,out_channels,kernel_size = 1)
            # self.conv2 = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1,act_layer=nn.ReLU)
            # self.conv2 = PConv(out_channels,4,kernel_size = 3)
            # self.conv2 = RepVGGBlock(out_channels, out_channels)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        shortcut = y
        y = self.conv2(y)
        y = y + shortcut
        y = self.one_conv(y)
        if self.shortcut:
            return x + y
        else:
            return y
class C2fBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize = 5,
                 expansion_depth = 1,
                 small_kersize = 3,
                 use_depthwise=True):
        super(C2fBlock, self).__init__()


        self.conv1 = Conv(in_channels, out_channels, 3)
        self.conv2 = Conv(out_channels, out_channels, 3)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1(x)
        # y = self.conv2(y)
        
        # y = self.one_conv(y)
        return y
class DepthBottleneckUni(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize = 5,
                 expansion_depth = 1,
                 small_kersize = 3,
                 use_depthwise=True):
        super(DepthBottleneckUni, self).__init__()


        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel,out_channels,kernel_size = 1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        
        y = self.act(self.conv2(y))
        # y = self.conv2(y)
        
        y = self.one_conv(y)
        return y
class RepHDW(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut = True, expansion = 0.5, kersize = 5,depth_expansion = 1,small_kersize = 3,use_depthwise = True):
        super(RepHDW, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(DepthBottleneckUni(self.c_, self.c_, shortcut,kersize,depth_expansion,small_kersize,use_depthwise) for _ in range(depth))
        self.conv2 = Conv(c_ * (depth+2), out_channels, 1, 1)

    def forward(self,x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return  y_out  
    
class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, shortcut=False, g=1, e=0.5, n =1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv_v8(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv_v8((1 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2fBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y[1:3], 1))
class C2fBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_v8(c1, c_, k[0], 1)
        self.cv2 = Conv_v8(c1, c_, k[1], 1)
        self.add = shortcut and c1 == c2 

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class Conv_v8(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class SAF1(nn.Module):
    def __init__(self, in_channel, out_channel, out_channel2, reduction =16):
        super(SAF1, self).__init__()
        self.in_numbers = len(in_channel)
        if self.in_numbers == 2:
            self.main_channel_in = in_channel[1]
            self.low_channel_in= in_channel[0]
            self.conv_main = Conv(self.main_channel_in, out_channel)
            # self.conv_med2 = Conv(self.med_channel_in, self.low_channel_out)
            self.conv_low = Conv(self.low_channel_in, out_channel)
            self.sigmoid= nn.Sigmoid()
        elif self.in_numbers == 3:
            self.main_channel_in2 = in_channel[2]
            self.main_channel_in = in_channel[1]
            self.low_channel_in= in_channel[0]
            self.conv_main = Conv(self.main_channel_in + self.main_channel_in2, out_channel)
            # self.conv_med2 = Conv(self.med_channel_in, self.low_channel_out)
            self.conv_low = Conv(self.low_channel_in, out_channel)
            self.sigmoid= nn.Sigmoid()
        



    def forward(self, x):
        if self.in_numbers == 2:
            low_x = x[0]
            main_x = x[1]
            low_x = self.conv_low(low_x)
            main_x = self.conv_main(main_x)
            main_weight = self.sigmoid(main_x)
            low_x  = low_x  * main_weight    
            out = torch.cat([low_x, main_x], 1)

        elif self.in_numbers == 3:
            low_x = x[0]
            main_x = x[1]
            high_x = x[2]
            low_x = self.conv_low(low_x)
            main_x = torch.cat([high_x, main_x], 1)
            main_x = self.conv_main(main_x)
            main_weight = self.sigmoid(main_x)
            low_x  = low_x  * main_weight 
            out = torch.cat([low_x, main_x], 1)

        return out
class Bottleneck2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize = 5,
                 expansion_depth = 1,
                 small_kersize = 3,
                 use_depthwise=True):
        super(Bottleneck2, self).__init__()


        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 3)
        self.conv2 = Conv(in_channels, mid_channel, 3)

        self.shortcut = shortcut
        

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
import math
class Head_layers(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_layers, self).__init__()

        self.stem = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # reg_conv0
        self.reg_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
        self.prior_prob = 1e-2
        self.initialize_biases()
    def initialize_biases(self):


        b = self.cls_pred.bias.view(-1, )
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.cls_pred.weight
        w.data.fill_(0.)
        self.cls_pred.weight = torch.nn.Parameter(w, requires_grad=True)


        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self,x):
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv(cls_x)
        cls_output = self.cls_pred(cls_feat)
        cls_output = torch.sigmoid(cls_output)
        reg_feat = self.reg_conv(reg_x)
        reg_output = self.reg_pred(reg_feat)

        return x, cls_output, reg_output

# class Head_Depth(nn.Module):
#     def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, num_anchors = 1):
#         super(Head_Depth, self).__init__()
#         from yolov6.layers.damo_yolo import DepthwiseConvModule
#         # self.stem = DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
#         self.stem = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
#         # cls_conv0
#         # self.cls_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1)
#         self.cls_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
#         # reg_conv0
#         self.reg_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
#         # cls_pred0
#         self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
#         # reg_pred0
#         self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
#         self.prior_prob = 1e-2
#         self.initialize_biases()
#     def initialize_biases(self):
#
#
#         b = self.cls_pred.bias.view(-1, )
#         b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
#         self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#         w = self.cls_pred.weight
#         w.data.fill_(0.)
#         self.cls_pred.weight = torch.nn.Parameter(w, requires_grad=True)
#
#
#         b = self.reg_pred.bias.view(-1, )
#         b.data.fill_(1.0)
#         self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#         w = self.reg_pred.weight
#         w.data.fill_(0.)
#         self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)
#
#     def forward(self,x):
#         x = self.stem(x)
#         cls_x = x
#         reg_x = x
#         cls_feat = self.cls_conv(cls_x)
#         cls_output = self.cls_pred(cls_feat)
#         cls_output = torch.sigmoid(cls_output)
#         reg_feat = self.reg_conv(reg_x)
#         reg_output = self.reg_pred(reg_feat)
#
#         return x, cls_output, reg_output


class Head_simota(nn.Module):
    def __init__(self,in_channels,out_channels, reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_simota, self).__init__()
        self.stem = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # reg_conv0
        self.reg_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)

        self.obj_pred = nn.Conv2d(in_channels=out_channels, out_channels=1 * (num_anchors), kernel_size=1)
        self.prior_prob = 1e-2
        self.initialize_biases()
    def initialize_biases(self):
        self.na = 1
        b = self.cls_pred.bias.view(self.na, -1)
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        b = self.obj_pred.bias.view(self.na, -1)
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self,x):
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv(cls_x)
        cls_output = self.cls_pred(cls_feat)
        # cls_output = torch.sigmoid(cls_output)
        reg_feat = self.reg_conv(reg_x)
        reg_output = self.reg_pred(reg_feat)
        obj_pred = self.obj_pred(reg_feat)

        return cls_output, reg_output, obj_pred
class Head_out(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_out, self).__init__()
        self.stem_cls = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.stem_reg = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = Conv(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=3, stride=1)
        # reg_conv0
        self.reg_conv = Conv(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=3, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=num_classes * num_anchors, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=4 * (reg_max + num_anchors), out_channels=4 * (reg_max + num_anchors), kernel_size=1)
        self.prior_prob = 1e-2
        self.initialize_biases()
    def initialize_biases(self):


        b = self.cls_pred.bias.view(-1, )
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.cls_pred.weight
        w.data.fill_(0.)
        self.cls_pred.weight = torch.nn.Parameter(w, requires_grad=True)


        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self,x):
        cls_x = self.stem_cls(x)
        reg_x = self.stem_reg(x)
        cls_output = self.cls_pred(self.cls_conv(cls_x))
        cls_output = torch.sigmoid(cls_output)
        reg_output = self.reg_pred(self.reg_conv(reg_x))

        return x, cls_output, reg_output

# BoT
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        # print('q shape:{},k shape:{},v shape:{}'.format(q.shape,k.shape,v.shape))  #1,4,64,256
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        # print("qkT=",content_content.shape)
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            # print("old content_content shape",content_content.shape) #1,4,256,256
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = content_position if (
                        content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            # print('new pos222-> shape:',content_position.shape)
            # print('new content222-> shape:',content_content.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out





class Head_DepthUni(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,kersize = 5, num_classes = 3, num_anchors = 1):
        super(Head_DepthUni, self).__init__()

        self.stem = Conv(in_channels, out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = UniRepLKNetBlock(out_channels, kernel_size=kersize)
        self.cls_conv_s = Conv(out_channels, out_channels)
        self.reg_conv = UniRepLKNetBlock(out_channels, kernel_size=kersize)
        self.reg_conv_s = Conv(out_channels, out_channels)
        # self.cls_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
        # reg_conv0
        # self.reg_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
        self.prior_prob = 1e-2
        self.initialize_biases()
    def initialize_biases(self):


        b = self.cls_pred.bias.view(-1, )
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.cls_pred.weight
        w.data.fill_(0.)
        self.cls_pred.weight = torch.nn.Parameter(w, requires_grad=True)


        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self,x):
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv_s(self.cls_conv(cls_x))
        
        cls_output = self.cls_pred(cls_feat)
        cls_output = torch.sigmoid(cls_output)
        reg_feat = self.reg_conv_s(self.reg_conv(reg_x))
        reg_output = self.reg_pred(reg_feat)

        return x, cls_output, reg_output
class AVG_down(nn.Module):
    def __init__(self, down_n = 2):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        # self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_n = down_n
        # self.output_size = np.array([H, W])
    
    def forward(self, x):
        B, C, H, W = x.shape
        H = int(H / self.down_n)
        W = int(W / self.down_n)
        output_size = np.array([H, W])
        x = self.avg_pool(x, output_size)
        # x = self.max(x)
        return x
class RepGELANMS(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut=True, expansion=0.5, kersize=5, depth_expansion=1,
                 small_kersize=3, use_depthwise=True):
        super(RepGELANMS, self).__init__()
        c1 = int(out_channels * expansion) * 3
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m1 = CSPDepthResELANUni(self.c_, self.c_, depth=depth, shortcut = shortcut, expansion = expansion, kersize = kersize,depth_expansion = depth_expansion,small_kersize = small_kersize)
        self.m2 = CSPDepthResELANUni(self.c_, self.c_, depth=depth, shortcut = shortcut, expansion = expansion, kersize = kersize,depth_expansion = depth_expansion,small_kersize = small_kersize)
        self.conv2 = Conv(c_ * 3, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_, self.c_), 1))
        x_out[1] = x_out[1] + x_out[0]
        x_out[1] = self.m1(x_out[1])

        x_out[2] = x_out[1] + x_out[2]
        x_out[2] = self.m2(x_out[2])

        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return y_out
class BottleneckTransformer(nn.Module):
    # Transformer bottleneck
    # expansion = 1

    def __init__(self, c1, c2, stride=1, heads=4, mhsa=True, resolution=None, expansion=1):
        super(BottleneckTransformer, self).__init__()
        c_ = int(c2 * expansion)
        self.cv1 = Conv(c1, c_, 1, 1)
        # self.bn1 = nn.BatchNorm2d(c2)
        if not mhsa:
            self.cv2 = Conv(c_, c2, 3, 1)
        else:
            self.cv2 = nn.ModuleList()
            self.cv2.append(MHSA(c2, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2d(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)
        self.shortcut = c1 == c2
        if stride != 1 or c1 != expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, expansion * c2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion * c2)
            )
        self.fc1 = nn.Linear(c2, c2)

    def forward(self, x):
        out = x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))
        return out


# class BoT3(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, e=0.5, e2=1, w=20, h=20):  # ch_in, ch_out, number, , expansion,w,h
#         super(BoT3, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
#         self.m = nn.Sequential(
#             *[BottleneckTransformer(c_, c_, stride=1, heads=4, mhsa=True, resolution=(w, h), expansion=e2) for _ in
#               range(n)])
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#
#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

from timm.models.layers import DropPath
class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = nn.BatchNorm2d(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
'''RepVGGBlock is a basic rep-style block, including training and deploy status
This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
'''
class RepGhostVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepGhostVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


#############RepGhost
# RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization
# https://github.com/ChengpengChen/RepGhost/blob/main/model/repghost.py
class RepGhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True, reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True
class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        dw_kernel_size=3,
        stride=1,
        c = 16,
        exp_size = 8,
        width = 0.5,
        se_ratio=0.0,
        shortcut=True,
        reparam=True,
        reparam_bn=True,
        reparam_identity=False,
        deploy=False,
    ):
        out_chs = _make_divisible(c * width, 4)
        mid_chs = _make_divisible(exp_size * width, 4)
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st repghost bottleneck
        x1 = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
import copy
def repghost_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    """
    taken from from https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
class RepGhostNet(nn.Module):
    def __init__(
        self,
        cfgs,
        num_classes=1000,
        width=1.0,
        dropout=0.2,
        shortcut=True,
        reparam=True,
        reparam_bn=True,
        reparam_identity=False,
        deploy=False,
    ):
        super(RepGhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout
        self.num_classes = num_classes

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = RepGhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(
                    block(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        se_ratio=se_ratio,
                        shortcut=shortcut,
                        reparam=reparam,
                        reparam_bn=reparam_bn,
                        reparam_identity=reparam_identity,
                        deploy=deploy
                    ),
                )
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width * 2, 4)
        stages.append(
            nn.Sequential(
                ConvBnAct(input_channel, output_channel, 1),
            ),
        )
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(
            input_channel, output_channel, 1, 1, 0, bias=True,
        )
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

    def convert_to_deploy(self):
        repghost_model_convert(self, do_copy=False)

#build backbone
def repghostnet(enable_se=True, **kwargs):
    """
    Constructs a RepGhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 8, 16, 0, 1]],
        # stage2
        [[3, 24, 24, 0, 2]],
        [[3, 36, 24, 0, 1]],
        # stage3
        [[5, 36, 40, 0.25 if enable_se else 0, 2]],
        [[5, 60, 40, 0.25 if enable_se else 0, 1]],
        # stage4
        [[3, 120, 80, 0, 2]],
        [
            [3, 100, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 240, 112, 0.25 if enable_se else 0, 1],
            [3, 336, 112, 0.25 if enable_se else 0, 1],
        ],
        # stage5
        [[5, 336, 160, 0.25 if enable_se else 0, 2]],
        [
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
        ],
    ]

    return RepGhostNet(cfgs, **kwargs)


############### end of repghost
#######focal#####
#focal_levels = [2, 2, 2, 2]  focal_windows=7, #[7, 5, 3, 1] depths [2,2,6,2]
class FocalTransformer(nn.Module):
    def __init__(self,in_chans, out_chans,depths=2,focal_windows=7,window_size=7):
        super(FocalTransformer, self).__init__()

        num_heads = out_chans // 32
        self.focal_transformer = FocalTransformer_block(in_chans = in_chans, embed_dim = out_chans,depths = depths,
                                                        num_heads = num_heads,focal_windows = focal_windows,window_size = window_size)
    def forward(self,x):
        x = self.focal_transformer(x)
        return x
class FocalC3(BepC3):
    def __init__(self, in_channels, out_channels, depths=2, focal_windows=7,window_size=7,e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(in_channels, out_channels)
        c_ = int(out_channels * e)
        num_heads = out_channels // 32
        self.m = FocalTransformer_block(in_chans = c_, embed_dim = c_,depths = depths,
                                                        num_heads = num_heads,focal_windows = focal_windows,window_size = window_size)
class MBConvC3(BepC3):
    def __init__(self, in_channels, out_channels, num_blocks=2, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(in_channels, out_channels)
        c_ = int(out_channels * e)
        self.m = MBConv_block(in_channels = c_, out_channels = c_,num_blocks = num_blocks)
class RepGhostC3(BepC3):
    def __init__(self, in_channels, out_channels,mid_c, num_blocks = 1, dw_kernel_size = 3,se_ratio = 0.0,width=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(in_channels, out_channels)
        c_ = int(out_channels * width)
        self.m = nn.Sequential(*(RepGhostBottleneck(in_chs = c_,exp_size = mid_c, c = int(c_/width),width = width,se_ratio = se_ratio,dw_kernel_size = dw_kernel_size) for _ in range(num_blocks)))


class SimCSPSPPF(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5):
        super(SimCSPSPPF, self).__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(in_channels, c_, 1, 1)
        self.cv3 = SimConv(c_, c_, 3, 1)
        self.cv4 = SimConv(c_, c_, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = SimConv(c_, c_, 3, 1)
        self.cv7 = SimConv(2 * c_, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat((y0, y3), dim=1))
class SPSimCSPSPPF(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5):
        super(SPSimCSPSPPF, self).__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.c_ = c_
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(in_channels, c_, 1, 1)
        self.cv3 = SimConv(c_, c_, 3, 1)
        self.cv3 = DepthwiseSeparableConv(c_, c_, 5, 1, act_layer=nn.ReLU)
        # self.cv4 = SimConv(c_, c_, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = SimConv(c_, c_, 3, 1)
        self.cv6 = DepthwiseSeparableConv(c_, c_, 5, 1, act_layer=nn.ReLU)
        self.cv7 = SimConv(4 * c_, out_channels, 1, 1)

    def forward(self, x):

        # x1 = list(x.split((self.c_,self.c_),1))
        # x2 = self.cv3(x1[-1])
        # x1.append(x2)
        # y1 = self.m(x1[-1])
        # y2 = self.m(y1)
        # y3 = self.m(y2)
        # y = torch.cat([x2, y1, y2, y3], 1)
        # y = self.cv5(y)
        # x1.append(y)
        # output = torch.cat(x1, 1)

        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x1)
        y1 = self.m(x3)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y4 = torch.cat([x3, y1, y2, y3], 1)
        y5 = self.cv5(y4)
        y6 = self.cv6(y5)
        output = torch.cat([x2,x3,y5,y6], 1)

        return self.cv7(output)
class BepBotC3(nn.Module):
    '''Beer-mug RepC3 Block'''

    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, e=0.5,
                 concat=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv_C3(in_channels, c_, 1, 1)
        self.cv2 = Conv_C3(in_channels, c_, 1, 1)
        self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvWrapper:
            self.cv1 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv2 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1, act=nn.SiLU())

        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BotRep, basic_block=block)
        self.concat = concat
        if not concat:
            self.cv3 = Conv_C3(c_, out_channels, 1, 1)

    def forward(self, x):
        if self.concat is True:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))
from yolov6.layers.BotNet import Attention
class BotRep(nn.Module):

    def __init__(self, in_channels, out_channels, head = 4,dim_head = 128,rel_pos_emb = False, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        attn_dim_out = head * dim_head
        self.conv1 = Conv(in_channels, out_channels)
        self.botAttention = nn.Sequential(
            Attention(
                dim = out_channels,
                heads = head,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            nn.Identity(),
            nn.BatchNorm2d(attn_dim_out),
            nn.ReLU(),
        )
        self.conv2 = Conv(attn_dim_out, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.botAttention(outputs)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs
class QARepVGGBlock(RepVGGBlock):
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(QARepVGGBlock, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.bn(self.se(self.rbr_reparam(inputs))))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))


    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3

        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias # remove bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        # keep post bn for QAT
        # if hasattr(self, 'bn'):
        #     self.__delattr__('bn')
        self.deploy = True
class Sequentially_Add(nn.Module):
    def __init__(self, c0 = 256,c1 = 128, c2 = 64, out = False):
        super(Sequentially_Add, self).__init__()
        self.up1 = nn.Upsample(None, 2, 'nearest')
        self.conv1 = ConvWrapper(c2,c1,kernel_size=1)
        self.conv2 = ConvWrapper(c1, c0, kernel_size=1)
        self.up2 = nn.Upsample(None, 2, 'nearest')
        self.out = out



    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        y1 = self.conv1(x1)
        y1 = self.up1(y1)
        y2 = x2 + y1
        y2 = self.conv2(y2)
        y2 = self.up2(y2)
        y3 = y2 + x3

        return y3
class Focal_Depth(nn.Module):
    def __init__(self, c1 ,c_ ,c2, out = False):
        super(Focal_Depth, self).__init__()
        self.conv0 = Conv(c1, c_)
        self.conv1 = DepthwiseSeparableConv(c_,c_,5,1)
        self.conv2 = DepthwiseSeparableConv(c_, c_, 5, 1,2)
        self.conv3 = Conv(c_, c2)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = x1+x0
        x3 = self.conv2(x2)
        x3 = x2+ x3
        x3 = x3 + x
        x4 = self.conv3(x3)

        return x4
def get_block(mode):
    if mode == 'repvgg':
        return RepVGGBlock
    elif mode == 'hyper_search':
        return LinearAddBlock
    elif mode == 'repopt':
        return RealVGGBlock
    elif mode == 'conv_relu':
        return SimConvWrapper
    elif mode == 'conv_silu':
        return ConvWrapper
    else:
        raise NotImplementedError("Undefied Repblock choice for mode {}".format(mode))

def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module

    elif isinstance(name, nn.Module):
        return name

    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
class CSPDepthResELAN(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut = True, expansion = 0.5, kersize = 5,depth_expansion = 1,small_kersize = 3,use_depthwise = True):
        super(CSPDepthResELAN, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(DepthBottleneck(self.c_, self.c_, shortcut,kersize,depth_expansion,small_kersize,use_depthwise) for _ in range(depth))
        self.conv2 = Conv(c_ * (depth+2), out_channels, 1, 1)

    def forward(self,x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return  y_out
class CSPDepthResELANUni(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut = True, expansion = 0.5, kersize = 5,depth_expansion = 1,small_kersize = 3,use_depthwise = True):
        super(CSPDepthResELANUni, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(DepthBottleneckUni(self.c_, self.c_, shortcut,kersize,depth_expansion,small_kersize,use_depthwise) for _ in range(depth))
        self.conv2 = Conv(c_ * (depth+2), out_channels, 1, 1)

    def forward(self,x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return  y_out
class CSPSDepthResELAN(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut = True, expansion = 0.5, kersize = 5,depth_expansion = 1,small_kersize = 3,use_depthwise = True):
        super(CSPSDepthResELAN, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(SDepthBottleneck(self.c_, self.c_, shortcut,kersize,depth_expansion,small_kersize,use_depthwise) for _ in range(depth))
        self.conv2 = Conv(c_ * (depth+2), out_channels, 1, 1)

    def forward(self,x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return  y_out

class SDepthBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize = 5,
                 expansion_depth = 1,
                 small_kersize = 3,
                 use_depthwise=True):
        super(SDepthBottleneck, self).__init__()


        mid_channel = int(in_channels * expansion_depth)
        self.shortcut = shortcut
        self.conv1 = Conv(in_channels, mid_channel, 1)
        # self.conv2  = DepthWiseConv(in_channels=mid_channel, out_channels=mid_channel, kernel_size=kersize, stride=1, act=nn.ReLU())
        self.conv2 = ReparamLargeKernelConv(in_channels=mid_channel, out_channels=mid_channel,
                                            kernel_size=kersize, stride=1,groups=mid_channel,small_kernel=small_kersize)
        # self.conv21 = ReparamLargeKernelConv(in_channels=mid_channel, out_channels=mid_channel,
        #                             kernel_size=kersize, stride=1,groups=mid_channel,small_kernel=small_kersize)
        # self.bn = nn.BatchNorm2d(mid_channel)
        # self.act = nn.ReLU()
        self.conv3 = Conv(mid_channel,out_channels,kernel_size = 1)


    def forward(self, x):
        shortcut = x
        y = self.conv1(x)
        y = self.conv2(y)
        # y = self.conv21(y)
        # y = self.act(self.bn(y))
        y = self.conv3(y)
        if self.shortcut:
            return y + shortcut
        else:
            return y
class SDepthMP(nn.Module):
    def __init__(self, in_channels, out_channels, depth = 1, shortcut = True, kersize = 5, small_kersize = 3, depth_expansion = 2,  expension_w = 3, use_depthwise = True):
        super(SDepthMP, self).__init__()
        self.in_channels = int(out_channels * expension_w / 2)
        self.mid_channel = int(out_channels / 2)
        self.in_conv = Conv(in_channels, self.in_channels, 1)
        self.m1 = nn.ModuleList(SDepthBottleneck(self.mid_channel, self.mid_channel, shortcut,kersize,depth_expansion,small_kersize,use_depthwise) for _ in range(depth))
        self.m2 = nn.ModuleList(Bottleneck2(self.mid_channel, self.mid_channel, shortcut,kersize,1,small_kersize,use_depthwise) for _ in range(depth))
        self.shortcut = shortcut
        self.conv2 = Conv(self.in_channels, out_channels, 1, 1)

    def forward(self,x):
        outs = []
        x = self.in_conv(x)
        x_out = list(x.split((self.mid_channel, self.mid_channel, self.mid_channel), 1))
        x_out1 = x_out[0]
        x_out2 = x_out[1]
        x_out3 = x_out[2]
        outs.append(x_out1)
            
        x_out2 = x_out2 + x_out1
        for conv in self.m1:
            y = conv(x_out2)
            outs.append(y)

        x_out3 = x_out3 + x_out2
        for conv in self.m2:
            y = conv(x_out3)
            outs.append(y)

        y_out = torch.cat(outs, axis=1)
        y_out = self.conv2(y_out)
        return  y_out
# class DepthBottleneck(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  shortcut=True,
#                  use_depthwise = True):
#         super(DepthBottleneck, self).__init__()
#         self.shortcut = shortcut
#         self.conv1 = Conv(in_channels, out_channels, 3, 1)
#         if use_depthwise:
#             from yolov6.layers.RTMDet import DepthwiseSeparableConv
#             self.conv2 = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
#         else:
#             self.conv2 = Conv(out_channels, out_channels, 3, 1)
#
#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.conv2(y)
#         if self.shortcut:
#             return x + y
#         else:
#             return y


class CSPRepResELAN(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut=True,  expansion=0.5, Rep_and_Conv = False):
        super(CSPRepResELAN, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(RepBottleneck(self.c_, self.c_, shortcut, Rep_and_Conv) for _ in range(depth))
        self.conv2 = Conv(c_ * (depth + 2), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return y_out


class RepBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 Rep_and_Conv=True):
        super(RepBottleneck, self).__init__()
        self.shortcut = shortcut
        if Rep_and_Conv:

            self.conv1 = RepVGGBlock(in_channels, out_channels, 3, 1)
            self.conv2 = Conv(out_channels, out_channels, 3, 1)
        else:
            from yolov6.layers.damo_yolo import RepConv
            self.conv1 = Conv(in_channels, out_channels, 3, 1)
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y
from torch import Tensor
class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x
    def forward_split_cat(self, x: Tensor) -> Tensor:

        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

class Head_PConv(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,dim = 4,num_classes = 3, num_anchors = 1):
        super(Head_PConv, self).__init__()
        from yolov6.layers.damo_yolo import DepthwiseConvModule
        # self.stem = DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
        # self.stem = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        # cls_conv0
        # self.cls_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.cls_conv = Partial_conv3(out_channels, dim, 'split_cat')
        # reg_conv0
        self.reg_conv = Partial_conv3(out_channels, dim, 'split_cat')

        self.cls_conv1 = Conv(out_channels,out_channels,1)
        self.reg_conv1 = Conv(out_channels, out_channels, 1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
        self.prior_prob = 1e-2
        self.initialize_biases()
    def initialize_biases(self):


        b = self.cls_pred.bias.view(-1, )
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.cls_pred.weight
        w.data.fill_(0.)
        self.cls_pred.weight = torch.nn.Parameter(w, requires_grad=True)


        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self,x):
        # x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv(cls_x)
        cls_feat = self.cls_conv1(cls_feat)
        cls_output = self.cls_pred(cls_feat)
        cls_output = torch.sigmoid(cls_output)
        reg_feat = self.reg_conv(reg_x)
        reg_feat = self.reg_conv1(reg_feat)
        reg_output = self.reg_pred(reg_feat)

        return x, cls_output, reg_output
    
    
import os
def get_block(mode):
    if mode == 'repvgg':
        return RepVGGBlock
    elif mode == 'hyper_search':
        return LinearAddBlock
    elif mode == 'repopt':
        return RealVGGBlock
    elif mode == 'conv_relu':
        return SimConvWrapper
    elif mode == 'conv_silu':
        return ConvWrapper
    else:
        raise NotImplementedError("Undefied Repblock choice for mode {}".format(mode))


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)



def get_bn(channels):
    return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, g)
        self.cv2 = Conv(c_, c_, 3, 1, c_)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
class Head_Ghost(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_Ghost, self).__init__()

        self.stem = Conv(in_channels, out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = GhostConv(out_channels, out_channels, k=3, s=1)
        self.reg_conv = GhostConv(out_channels, out_channels, k=3, s=1)
        # self.cls_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
        # reg_conv0
        # self.reg_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
        self.prior_prob = 1e-2
        self.initialize_biases()
    def initialize_biases(self):


        b = self.cls_pred.bias.view(-1, )
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.cls_pred.weight
        w.data.fill_(0.)
        self.cls_pred.weight = torch.nn.Parameter(w, requires_grad=True)


        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self,x):
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv(cls_x)
        cls_output = self.cls_pred(cls_feat)
        cls_output = torch.sigmoid(cls_output)
        reg_feat = self.reg_conv(reg_x)
        reg_output = self.reg_pred(reg_feat)

        return x, cls_output, reg_output

class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.in_channels = in_channels
        self.groups = groups
        self.act = nn.ReLU()
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels, in_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels,in_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)
                # self.one_conv = Conv(in_channels,out_channels,kernel_size = 1)
                # self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, inputs):
        # if hasattr(self, 'rbr_identity'):
        #     id_out = self.rbr_identity(inputs)
        # else:
        #     id_out = 0

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
                # out = self.nonlinearity(out)
        return self.act(out)
        # return self.one_conv(out)

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        # if self.rbr_identity is not None:
        #     bn_kernelid, bn_biasid = self._fuse_bn_tensor(self.rbr_identity)
        #     eq_k += bn_kernelid
        #     eq_b += bn_biasid
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)

        return eq_k, eq_b

    #融合bn层
    # def _fuse_bn_tensor(self, branch):
    #     if branch is None:
    #         return 0, 0
    #     assert isinstance(branch, nn.BatchNorm2d)
    #     if not hasattr(self, 'id_tensor'):
    #         input_dim = self.in_channels // self.groups
    #         kernel_value = np.zeros((self.in_channels, input_dim, self.kernel_size, self.kernel_size), dtype=np.float32)
    #         for i in range(self.in_channels):
    #             kernel_value[i, i % input_dim, 1, 1] = 1
    #         self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
    #     kernel = self.id_tensor
    #     running_mean = branch.running_mean
    #     running_var = branch.running_var
    #     gamma = branch.weight
    #     beta = branch.bias
    #     eps = branch.eps
    #     std = (running_var + eps).sqrt()
    #     t = (gamma / std).reshape(-1, 1, 1, 1)
    #     return kernel * t, beta - running_mean * gamma / std

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')
            

class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=2):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2).view(b,c)
        return y

class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gap_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)
    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2)
        return max_out

class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1,bias=True)
    def forward(self, x):
        x1 = self.conv1(x)
        return x1

class EAEF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_pool = Feature_Pool(dim)
        self.dwconv = nn.Conv2d(dim*2,dim*2,kernel_size=7,padding=3,groups=dim)
        self.ecse = Channel_Attention(dim*2)
        #self.ccse = Channel_Attention(dim)
        self.sse_r = Spatial_Attention(dim)
        self.sse_t = Spatial_Attention(dim)
    def forward(self, x):
        ############################################################################
        RGB,T = x[0],x[1]
        b, c, h, w = RGB.size()
        rgb_y = self.mlp_pool(RGB)
        t_y = self.mlp_pool(T)
        rgb_y = rgb_y / rgb_y.norm(dim=1, keepdim=True)
        t_y = t_y / t_y.norm(dim=1, keepdim=True)
        rgb_y = rgb_y.view(b, c, 1)
        t_y = t_y.view(b, 1, c)
        logits_per = c * rgb_y @ t_y
        cross_gate = torch.diagonal(torch.sigmoid(logits_per)).reshape(b, c, 1, 1)
        device = x[0].device
        add_gate = torch.ones(cross_gate.shape).cuda() - cross_gate
        ##########################################################################
        New_RGB_e = RGB * cross_gate
        New_T_e = T * cross_gate
        New_RGB_c = RGB * add_gate
        New_T_c = T * add_gate
        x_cat_e = torch.cat((New_RGB_e, New_T_e), dim=1)
        ##########################################################################
        fuse_gate_e = torch.sigmoid(self.ecse(self.dwconv(x_cat_e)))
        rgb_gate_e, t_gate_e = fuse_gate_e[:, 0:c, :], fuse_gate_e[:, c:c * 2, :]
        ##########################################################################
        New_RGB = New_RGB_e * rgb_gate_e + New_RGB_c
        New_T = New_T_e * t_gate_e + New_T_c
        ##########################################################################
        New_fuse_RGB = self.sse_r(New_RGB)
        New_fuse_T = self.sse_t(New_T)
        attention_vector = torch.cat([New_fuse_RGB, New_fuse_T], dim=1)
        attention_vector = torch.softmax(attention_vector, dim=1)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        New_RGB = New_RGB * attention_vector_l
        New_T = New_T * attention_vector_r
        New_fuse = New_T + New_RGB
        out = [New_RGB, New_T, New_fuse]
        ##########################################################################
        return out


class EAEF_out(nn.Module):
    def __init__(self, dim):
        super(EAEF_out, self).__init__()
        self.dim = dim
    def forward(self, x):
        return x[self.dim]




# From PyTorch internals
from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
def get_conv2d_uni(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    # if attempt_use_lk_impl and need_large_impl:
    #     print('---------------- trying to import iGEMM implementation for large-kernel conv')
    #     try:
    #         from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
    #         print('---------------- found iGEMM implementation ')
    #     except:
    #         DepthWiseConv2dImplicitGEMM = None
    #         print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
    #     if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
    #             and out_channels == groups and stride == 1 and dilation == 1:
    #         print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
    #         return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)
def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), dtype=kernel.dtype, device =kernel.device )
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel
class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d_uni(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        # if kernel_size == 17:
        #     self.kernel_sizes = [5, 9, 3, 3, 3]
        #     self.dilates = [1, 2, 4, 5, 7]
        # elif kernel_size == 15:
        #     self.kernel_sizes = [5, 7, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 5, 7]
        # elif kernel_size == 13:
        #     self.kernel_sizes = [5, 7, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 4, 5]
        # elif kernel_size == 11:
        #     self.kernel_sizes = [5, 5, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 4, 5]
        # elif kernel_size == 9:
        #     self.kernel_sizes = [5, 5, 3, 3]
        #     self.dilates = [1, 2, 3, 4]
        # elif kernel_size == 7:
        #     self.kernel_sizes = [5, 3, 3, 3]
        #     self.dilates = [1, 1, 2, 3]
        # elif kernel_size == 5:
        #     self.kernel_sizes = [3, 3, 1]
        #     self.dilates = [1, 2, 1]
        # elif kernel_size == 3:
        #     self.kernel_sizes = [3, 1]
        #     self.dilates = [1, 1]
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [7, 5, 3]
            self.dilates = [1, 1, 1]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3]
            self.dilates = [1, 1]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        elif kernel_size == 3:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        

        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d_uni(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

class UniRepLKNetBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 deploy=False,
                 attempt_use_lk_impl=True):
        super().__init__()
        if deploy:
            print('------------------------------- Note: deploy mode')
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 3:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            assert kernel_size in [3]
            self.dwconv = get_conv2d_uni(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=deploy,
                                     attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim)


    def forward(self, inputs):

        out = self.norm(self.dwconv(inputs))
        return out

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                            self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 self.dwconv.padding, self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()
            
            
class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv_v9(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv_v9(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
    
class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_v9(c1, c_, 1, 1)
        self.cv2 = Conv_v9(c1, c_, 1, 1)
        self.cv3 = Conv_v9(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    
class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv_v9(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv_v9(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv_v9(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv_v9):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
            

class Conv_v9(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
