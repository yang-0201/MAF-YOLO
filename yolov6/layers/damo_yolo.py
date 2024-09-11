############DAMO yolo

import numpy as np
import torch
import torch.nn as nn

class Focus(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act='silu'):
        super().__init__()
        self.conv = ConvBNAct(in_channels * 4,
                              out_channels,
                              ksize,
                              stride,
                              act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
class ConvKXBN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(ConvKXBN, self).__init__()
        self.conv1 = nn.Conv2d(in_c,
                               out_c,
                               kernel_size,
                               stride, (kernel_size - 1) // 2,
                               groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class ConvKXBNRELU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, act='silu'):
        super(ConvKXBNRELU, self).__init__()
        self.conv = ConvKXBN(in_c, out_c, kernel_size, stride)
        if act is None:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(act)

    def forward(self, x):
        output = self.conv(x)
        return self.activation_function(output)
class ResConvBlock(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(ResConvBlock, self).__init__()
        self.stride = stride
        if block_type == 'k1kx':
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size=1, stride=1)
        else:
            self.conv1 = ConvKXBN(in_c,
                                  btn_c,
                                  kernel_size=kernel_size,
                                  stride=1)

        if not reparam:
            self.conv2 = ConvKXBN(btn_c, out_c, kernel_size, stride)
        else:
            self.conv2 = RepConv(btn_c,
                                 out_c,
                                 kernel_size,
                                 stride,
                                 act='identity')

        self.activation_function = get_activation(act)

        if in_c != out_c and stride != 2:
            self.residual_proj = ConvKXBN(in_c, out_c, 1, 1)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            reslink = self.residual_proj(x)
        else:
            reslink = x
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        if self.stride != 2:
            x = x + reslink
        x = self.activation_function(x)
        return x
class SuperResStem(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 num_blocks,
                 with_spp=False,
                 act='relu',
                 reparam=True,
                 block_type='k1kx'):
        super(SuperResStem, self).__init__()
        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        for block_id in range(num_blocks):
            if block_id == 0:
                in_channels = in_c
                out_channels = out_c
                this_stride = stride
                this_kernel_size = kernel_size
            else:
                in_channels = out_c
                out_channels = out_c
                this_stride = 1
                this_kernel_size = kernel_size
            the_block = ResConvBlock(in_channels,
                                     out_channels,
                                     btn_c,
                                     this_kernel_size,
                                     this_stride,
                                     act=act,
                                     reparam=reparam,
                                     block_type=block_type)
            self.block_list.append(the_block)
            if block_id == 0 and with_spp:
                self.block_list.append(
                    SPPBottleneck(out_channels, out_channels))

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels,
                               hidden_channels,
                               1,
                               stride=1,
                               act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvBNAct(conv2_channels,
                               out_channels,
                               1,
                               stride=1,
                               act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x




class CSPStem(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 stride,
                 kernel_size,
                 num_blocks,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(CSPStem, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.stride = stride
        if self.stride == 2:
            self.num_blocks = num_blocks - 1
        else:
            self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.act = act
        self.block_type = block_type
        out_c = out_c // 2

        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        for block_id in range(self.num_blocks):
            if self.stride == 1 and block_id == 0:
                in_c = in_c // 2
            else:
                in_c = out_c
            the_block = ResConvBlock(in_c,
                                     out_c,
                                     btn_c,
                                     kernel_size,
                                     stride=1,
                                     act=act,
                                     reparam=reparam,
                                     block_type=block_type)
            self.block_list.append(the_block)

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output



















###########neck
class DepthCSPStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 n,
                 split_ratio = 0.5,
                 act='swish',
                 spp=False,
                 channel_attention = False):
        super(DepthCSPStage, self).__init__()

        ch_first = int(ch_out * split_ratio)
        ch_mid = ch_first
        self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse_Depth(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           shortcut=True,
                                           use_depthwise= True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)
        self.channel_attention = channel_attention
        if channel_attention:
            self.attention = ChannelAttention(ch_mid * n + ch_first)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        if self.channel_attention:
            y = self.attention(y)
        y = self.conv3(y)
        return y
class ChannelAttention(nn.Module):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out
class SPDepthCSPStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 n,
                 a = 0.5,
                 act='swish',
                 spp=False):
        super(SPDepthCSPStage, self).__init__()



        ch_first = int(ch_out * a)
        ch_mid = ch_first
        self.ch_mid = ch_mid
        self.conv1 = ConvBNAct(ch_in, ch_first * 2, 1, act=act)

        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse_Depth(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           shortcut=True,
                                           use_depthwise= True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * (n+1) + ch_first, ch_out, 1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        mid_out = list(x.split((self.ch_mid,self.ch_mid),1))


        for conv in self.convs:
            y2 = conv(mid_out[-1])
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y
class SPCSPStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 n,
                 act='swish',
                 spp=False):
        super(SPCSPStage, self).__init__()

        split_ratio = 2

        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.ch_mid = ch_mid
        self.conv1 = ConvBNAct(ch_in, 2 * ch_mid, 1, act=act)

        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           shortcut=False,
                                           use_depthwise= False))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * (n+1) + ch_first, ch_out, 1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        mid_out = list(x.split((self.ch_mid,self.ch_mid),1))


        for conv in self.convs:
            y2 = conv(mid_out[-1])
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y
class DepthGFPN(nn.Module):
    def __init__(self,in_channels,out_channels,depth=1.0,hidden_ratio = 1.0,split_ratio = 0.5,channel_attention = False,act = 'silu',block_name='BasicBlock_3x3_Reverse',spp = False):
        super(DepthGFPN, self).__init__()


        self.merge_3 = DepthCSPStage(block_name,
                                in_channels,
                                hidden_ratio,
                                out_channels,
                                round(3 * depth),
                                     split_ratio = split_ratio,
                                act=act,
                                     channel_attention = channel_attention)


    def forward(self,x):
        x = self.merge_3(x)
        return  x
class SPDepthGFPN(nn.Module):
    def __init__(self,in_channels,out_channels,depth=1.0,hidden_ratio = 1.0,a = 0.5,act = 'silu',block_name='BasicBlock_3x3_Reverse',spp = False):
        super(SPDepthGFPN, self).__init__()


        self.merge_3 = SPDepthCSPStage(block_name,
                                in_channels,
                                hidden_ratio,
                                out_channels,
                                round(3 * depth),
                                       a = a,
                                act=act)


    def forward(self,x):
        x = self.merge_3(x)
        return  x

class RepGFPN(nn.Module):
    def __init__(self,in_channels,out_channels,depth=1.0,hidden_ratio = 1.0,split_ratio = 0.5,shortcut = True,act = 'silu',block_name='BasicBlock_3x3_Reverse',spp = False):
        super(RepGFPN, self).__init__()


        self.merge_3 = CSPStage(block_name,
                                in_channels,
                                hidden_ratio,
                                out_channels,
                                round(3 * depth),
                                shortcut = shortcut,
                                split_ratio = split_ratio,
                                act=act)


    def forward(self,x):
        x = self.merge_3(x)
        return  x
class SPRepGFPN(nn.Module):
    def __init__(self,in_channels,out_channels,depth=1.0,hidden_ratio = 1.0,act = 'silu',block_name='BasicBlock_3x3_Reverse',spp = False):
        super(SPRepGFPN, self).__init__()


        self.merge_3 = SPCSPStage(block_name,
                                in_channels,
                                hidden_ratio,
                                out_channels,
                                round(3 * depth),
                                act=act)


    def forward(self,x):
        x = self.merge_3(x)
        return  x
class RepGFPN_RTM(nn.Module):
    def __init__(self,in_channels,out_channels,depth=1.0,hidden_ratio = 1.0,act = 'silu',block_name='Depth',spp = False):
        super(RepGFPN_RTM, self).__init__()


        self.merge_3 = CSPStage_RTM(block_name,
                                in_channels,
                                hidden_ratio,
                                out_channels,
                                round(3 * depth),
                                act=act)


    def forward(self,x):
        x = self.merge_3(x)
        return  x
class CSPStage_RTM(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 n,
                 act='swish',
                 spp=False):
        super(CSPStage_RTM, self).__init__()

        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'Depth':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse_Depth(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           use_depthwise = True,
                                           shortcut=True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y


class CSPStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 n,
                 split_ratio = 0.5,
                 shortcut = True,
                 act='swish',
                 spp=False):
        super(CSPStage, self).__init__()

        ch_first = int(ch_out * split_ratio)
        ch_mid = ch_first
        self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           shortcut=False))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y
class ConvBNAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride=1,
        act='relu',
        groups=1,
        bias=False,
        norm='bn',
        reparam=False,
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))
class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 act='relu',
                 shortcut=True,
                 use_depthwise = False,
                 rep_conv = False):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.rep_conv = rep_conv
        self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
        if use_depthwise:
            from yolov6.layers.RTMDet import DepthwiseSeparableConv
            self.conv1 = DepthwiseSeparableConv(ch_hidden, ch_out, 5, stride=1)
            self.conv2 = ConvBNAct(ch_in, ch_hidden, 3, 1)
        else:
            self.conv2 = RepConv(ch_in, ch_hidden, 3, stride=1)
        if rep_conv:
            self.conv2 = ConvBNAct(ch_in,ch_hidden,3,stride=1,act = act)
            self.conv1 = RepConv(ch_hidden, ch_out, 3, stride=1, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y
class BasicBlock_3x3_Reverse_Depth(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 act='relu',
                 shortcut=True,
                 use_depthwise = True,
                 rep_conv = False):
        super(BasicBlock_3x3_Reverse_Depth, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.rep_conv = rep_conv
        self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
        if use_depthwise:
            from yolov6.layers.RTMDet import DepthwiseSeparableConv
            self.conv1 = DepthwiseSeparableConv( ch_hidden, ch_out, 5, act_layer=nn.ReLU)
            self.conv2 = ConvBNAct(ch_in, ch_hidden, 3, 1,  act=act)
        else:
            self.conv2 = RepConv(ch_in, ch_hidden, 3, stride=1)
        if rep_conv:
            self.conv2 = ConvBNAct(ch_in,ch_hidden,3,stride=1,act = act)
            self.conv1 = RepConv(ch_hidden, ch_out, 3, stride=1, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y
def get_norm(name, out_channels, inplace=True):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    else:
        raise NotImplementedError
    return module
class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        act='swish',
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBNAct(ch_in, ch_out, k, act=act)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y
import torch.nn.functional as F
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)
class RepConv(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_11,
                                   groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

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
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
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
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
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
class DepthwiseConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias="auto",
        norm_cfg=dict(type="BN"),
        activation="ReLU",
        inplace=True,
        order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"),
    ):
        super(DepthwiseConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 6
        assert set(order) == {
            "depthwise",
            "dwnorm",
            "act",
            "pointwise",
            "pwnorm",
            "act",
        }

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias


        # build convolution layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.depthwise.in_channels
        self.out_channels = self.pointwise.out_channels
        self.kernel_size = self.depthwise.kernel_size
        self.stride = self.depthwise.stride
        self.padding = self.depthwise.padding
        self.dilation = self.depthwise.dilation
        self.transposed = self.depthwise.transposed
        self.output_padding = self.depthwise.output_padding

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            _, self.dwnorm = build_norm_layer(norm_cfg, in_channels)
            _, self.pwnorm = build_norm_layer(norm_cfg, out_channels)

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        if self.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"
        kaiming_init(self.depthwise, nonlinearity=nonlinearity)
        kaiming_init(self.pointwise, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.dwnorm, 1, bias=0)
            constant_init(self.pwnorm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer_name in self.order:
            if layer_name != "act":
                layer = self.__getattr__(layer_name)
                x = layer(x)
            elif layer_name == "act" and self.activation:
                x = self.act(x)
        return x
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
        elif name == 'swish':
            module = Swish(inplace=inplace)
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
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "SyncBN": ("bn", nn.SyncBatchNorm),
    "GN": ("gn", nn.GroupNorm),
    # and potentially 'SN'
}
def build_norm_layer(cfg, num_features, postfix=""):
    """Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        if layer_type == "SyncBN" and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)
activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ReLU6": nn.ReLU6,
    "SELU": nn.SELU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "PReLU": nn.PReLU,
    "SiLU": nn.SiLU,
    "HardSwish": nn.Hardswish,
    "Hardswish": nn.Hardswish,
    None: nn.Identity,
}


def act_layers(name):
    assert name in activations.keys()
    if name == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == "GELU":
        return nn.GELU()
    elif name == "PReLU":
        return nn.PReLU()
    else:
        return activations[name](inplace=True)