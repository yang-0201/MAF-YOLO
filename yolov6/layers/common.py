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



class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

class Out(nn.Module):
    def __init__(self):
        super(Out, self).__init__()

    def forward(self, x):
        outputs = []
        for y in x:
            outputs.append(y)
        return outputs


# class MPRep(nn.Module):
#     def __init__(self, c1,c2):  #c1 = c2
#         c_ = c2//2
#         super(MPRep, self).__init__()
#         self.mp = MP()
#         self.conv1 = Conv(c1, c_, 1, 1)
#         self.conv2 = RepVGGBlock(c1, c_, 3, 2)
#
#     def forward(self, input):
#
#         x1 = self.mp(input)
#         x1 = self.conv1(x1)
#         x2 = self.conv2(input)
#         out = torch.cat([x1, x2], dim=1)
#         return out
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
class Head_DepthUni(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,kersize = 5, num_classes = 3, num_anchors = 1):
        super(Head_DepthUni, self).__init__()

        self.stem = Conv(in_channels, out_channels, kernel_size=1, stride=1)
        self.cls_conv = UniRepLKNetBlock(out_channels, kernel_size=kersize)
        self.cls_conv_s = Conv(out_channels, out_channels)
        self.reg_conv = UniRepLKNetBlock(out_channels, kernel_size=kersize)
        self.reg_conv_s = Conv(out_channels, out_channels)
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
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
            self.conv2 = ReparamLargeKernelConv(in_channels=mid_channel, out_channels=out_channels,
                                                kernel_size=kersize, stride=1,groups=mid_channel,small_kernel=small_kersize)
            self.one_conv = Conv(mid_channel,out_channels,kernel_size = 1)

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
        y = self.one_conv(y)
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


from einops import rearrange

class RepELANMS(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut=True, expansion=0.5, kersize=5, depth_expansion=1,
                 small_kersize=3, use_depthwise=True):
        super(RepELANMS, self).__init__()
        c1 = int(out_channels * expansion) * 3
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m1 = DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
        self.m2 = DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
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

class RepELANMS2(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut=True, expansion=0.5, kersize=5, depth_expansion=1,
                 small_kersize=3, use_depthwise=True):
        super(RepELANMS2, self).__init__()
        c1 = int(out_channels * expansion) * 3
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m1 = DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
        self.m11 = DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
        self.m2 = DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
        self.m22 = DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
        self.conv2 = Conv(c_ * 5, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_, self.c_), 1))
        x_out[1] = x_out[1] + x_out[0]
        x_out[1] = self.m1(x_out[1])
        add11 = self.m11(x_out[1])
        x_out.append(add11)

        x_out[2] = x_out[1] + x_out[2]
        x_out[2] = self.m2(x_out[2])
        x_out[2] = add11 + x_out[2]
        add22 = self.m22(x_out[2])
        x_out.append(add22)

        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return y_out


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

        self.deploy = True

class AVG_down(nn.Module):
    def __init__(self, down_n=2):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.down_n = down_n
        # self.output_size = np.array([H, W])

    def forward(self, x):
        B, C, H, W = x.shape
        H = int(H / self.down_n)
        W = int(W / self.down_n)
        output_size = np.array([H, W])
        x = self.avg_pool(x, output_size)
        return x


def get_block(mode):
    if mode == 'repvgg':
        return RepVGGBlock
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
        self.conv2 = ReparamLargeKernelConv(in_channels=mid_channel, out_channels=mid_channel,
                                            kernel_size=kersize, stride=1,groups=mid_channel,small_kernel=small_kersize)

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


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
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


class RepELANMSv2(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5,
                 small_kersize=3, use_depthwise=True):
        super(RepELANMSv2, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] += cascade[j]
                    if j == self.depth - 1:
                        cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out


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