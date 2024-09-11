import torch
import torch.nn as nn
from timm1.models.layers import create_conv2d, DropPath, get_norm_act_layer
import numpy as np




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
def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class ConvModule(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=True, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d,
            se_layer=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, groups=groups)
        self.bn1 = norm_act_layer(in_chs, inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
class ELAN_Depth_S(nn.Module):
    def __init__(self,c1, c2,hidden_ratio = 0.5,use_depthwise = True,channel_attention = True,add_identity = True):
        c_ = int(c2 * hidden_ratio)
        super(ELAN_Depth_S, self).__init__()
        self.channel_attention = channel_attention
        self.add_identity = add_identity
        self.conv1 = ConvModule(c1, c_, 1, 1)
        self.conv2 = ConvModule(c1, c_, 1, 1)
        self.conv3 = ConvModule(c_, c_, 3, 1)
        if use_depthwise:
            self.conv4 = DepthwiseSeparableConv(c_, c_, 5, 1)
        else:
            self.conv4 = ConvModule(c_, c_, 3, 1)

        self.conv7 = ConvModule(3 * c_, c2, 1, 1)
        if self.channel_attention:
            self.attention = ChannelAttention(3 * c_)
    def forward(self,x):
        out1 = self.conv1(x)
        x = self.conv2(x)
        out2 = x
        x = self.conv3(x)
        x = self.conv4(x)
        if self.add_identity:
            x =out2+x
        out3 = x

        x = torch.cat([out1, out2, out3], dim = 1)
        if self.channel_attention:
            x = self.attention(x)
        out = self.conv7(x)
        return out
class ELAN_Depth(nn.Module):
    def __init__(self,c1, c2,hidden_ratio = 0.5,use_depthwise = True,channel_attention = True,add_identity = True):
        c_ = int(c2 * hidden_ratio)
        super(ELAN_Depth, self).__init__()
        self.channel_attention = channel_attention
        self.add_identity = add_identity
        self.conv1 = ConvModule(c1, c_, 1, 1)
        self.conv2 = ConvModule(c1, c_, 1, 1)
        self.conv3 = ConvModule(c_, c_, 3, 1)
        if use_depthwise:
            self.conv4 = DepthwiseSeparableConv(c_, c_, 5, 1)
        else:
            self.conv4 = ConvModule(c_, c_, 3, 1)
        self.conv5 = ConvModule(c_, c_, 3, 1)
        if use_depthwise:
            self.conv6 = DepthwiseSeparableConv(c_, c_, 5, 1)
        else:
            self.conv6 = ConvModule(c_, c_, 3, 1)
        self.conv7 = ConvModule(4 * c_, c2, 1, 1)
        if self.channel_attention:
            self.attention = ChannelAttention(4 * c_)
    def forward(self,x):
        out1 = self.conv1(x)
        x = self.conv2(x)
        out2 = x
        x = self.conv3(x)
        x = self.conv4(x)
        if self.add_identity:
            x =out2+x
        out3 = x
        x = self.conv5(x)
        x = self.conv6(x)
        if self.add_identity:
            x =out3+x
        out4 = x
        x = torch.cat([out1, out2, out3, out4], dim = 1)
        if self.channel_attention:
            x = self.attention(x)
        out = self.conv7(x)
        return out
class ELAN_Depth_M(nn.Module):
    def __init__(self,c1, c2,use_depthwise = True,channel_attention = True,add_identity = True ,hidden_ratio = 0.5):
        c_ = int (c2 * hidden_ratio )
        super(ELAN_Depth_M, self).__init__()
        self.channel_attention = channel_attention
        self.add_identity = add_identity
        self.conv1 = ConvModule(c1, c_, 1, 1)
        self.conv2 = ConvModule(c1, c_, 1, 1)
        self.conv3 = ConvModule(c_, c_, 3, 1)
        if use_depthwise:
            self.conv4 = DepthwiseSeparableConv(c_, c_, 5, 1)
        else:
            self.conv4 = ConvModule(c_, c_, 3, 1)
        self.conv5 = ConvModule(c_, c_, 3, 1)
        if use_depthwise:
            self.conv6 = DepthwiseSeparableConv(c_, c_, 5, 1)
        else:
            self.conv6 = ConvModule(c_, c_, 3, 1)
        self.conv7 = ConvModule(c_, c_, 3, 1)
        if use_depthwise:
            self.conv8 = DepthwiseSeparableConv(c_, c_, 5, 1)
        else:
            self.conv8 = ConvModule(c_, c_, 3, 1)
        self.conv9 = ConvModule(c_, c_, 3, 1)
        if use_depthwise:
            self.conv10 = DepthwiseSeparableConv(c_, c_, 5, 1)
        else:
            self.conv10 = ConvModule(c_, c_, 3, 1)
        self.conv11 = ConvModule(6 * c_, c2, 1, 1)
        if self.channel_attention:
            self.attention = ChannelAttention(6 * c_)
    def forward(self,x):
        out1 = self.conv1(x)
        x = self.conv2(x)
        out2 = x
        x = self.conv3(x)
        x = self.conv4(x)
        if self.add_identity:
            x =out2+x
        out3 = x
        x = self.conv5(x)
        x = self.conv6(x)
        if self.add_identity:
            x =out3+x
        out4 = x

        x = self.conv7(x)
        x = self.conv8(x)
        if self.add_identity:
            x =out4+x
        out5 = x

        x = self.conv9(x)
        x = self.conv10(x)
        if self.add_identity:
            x =out5+x
        out6 = x

        x = torch.cat([out1, out2,out3, out4,out5,out6], dim = 1)
        if self.channel_attention:
            x = self.attention(x)
        out = self.conv11(x)
        return out
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

class CSPNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels, out_channels, expansion=0.5, add_identity=True,use_depthwise = True,use__Rep = False,use_se = False):
        super(CSPNeXtBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)
        conv_depth = DepthwiseSeparableConv
        self.use_se = use_se
        if use_se:
            self.se_attention = SEAttention(hidden_channels)
        if use__Rep:
            self.conv1 = RepVGGBlock(in_channels, hidden_channels, kernel_size = 3, stride=1)
        else:
            self.conv1 = ConvModule(in_channels, hidden_channels, k = 3, s=1)
        if use_depthwise:
            self.conv2 = conv_depth(hidden_channels, out_channels, 5, stride=1)
        else:
            self.conv2 = ConvModule(hidden_channels, out_channels, 5, s=1)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        # if self.use_se:
        #     out = self.se_attention(out)

        if self.add_identity:
            return out + identity
        else:
            return out
class CSPRepBlock(nn.Module):

    def __init__(self,
                 in_channels, out_channels, expansion=0.5, add_identity=True,use_depthwise = False,use__Rep = False,use_se = False):
        super(CSPRepBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)
        conv_depth = DepthwiseSeparableConv
        self.use_se = use_se
        if use_se:
            self.se_attention = SEAttention(hidden_channels)
        self.conv1 = RepVGGBlock(in_channels, hidden_channels, kernel_size = 3, stride=1)
        if use_depthwise:
            self.conv2 = conv_depth(hidden_channels, out_channels, 5, stride=1)
        else:
            self.conv2 = ConvModule(hidden_channels, out_channels, 3, s=1)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        # if self.use_se:
        #     out = self.se_attention(out)

        if self.add_identity:
            return out + identity
        else:
            return out
from torch.nn import init
class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        # # AIEAGNY
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CSPNeXtLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 channel_attention = True,
                 add_identity=True,
                 use_depthwise=True,
                 expand_ratio=0.5):
        super(CSPNeXtLayer, self).__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid_channels, 1)
        self.short_conv = ConvModule(in_channels, mid_channels, 1)

        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1)
        self.channel_attention = channel_attention
        if self.channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid_channels, mid_channels, 1.0, add_identity, use_depthwise) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)

class CSPRepLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 channel_attention = True,
                 add_identity=True,
                 use_depthwise=False,
                 expand_ratio=0.5):
        super(CSPRepLayer, self).__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid_channels, 1)
        self.short_conv = ConvModule(in_channels, mid_channels, 1)

        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1)
        self.channel_attention = channel_attention
        if self.channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)
        self.blocks = nn.Sequential(*[
            CSPRepBlock(mid_channels, mid_channels, 1.0, add_identity, use_depthwise) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)
import math
# RTMDetHead with separated BN layers and shared conv layers.
class RTM_SepBNHead(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, stage = 3,stacked_convs_number = 2, num_anchors = 1,share_conv = True):
        super(RTM_SepBNHead, self).__init__()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.stage = stage
        self.stacked_convs_number = stacked_convs_number

        for n in range(self.stage):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs_number):
                chn = in_channels[n] if i == 0 else out_channels[i]
                cls_convs.append(
                    ConvModule(
                        chn,
                        out_channels[n],
                        3,
                        s=1,
                        p=1 ))
                reg_convs.append(
                    ConvModule(
                        chn,
                        out_channels[n],
                        3,
                        s=1,
                        p=1))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.rtm_cls.append(
                nn.Conv2d(
                    out_channels[n],
                    num_classes * num_anchors,
                    1,
                    padding=0))
            self.rtm_reg.append(
                nn.Conv2d(
                    out_channels[n],
                    4 * (reg_max + num_anchors),
                    1,
                    padding=0))
        if share_conv:
            for n in range(stage):
                for i in range(self.stacked_convs_number):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
        self.initialize_biases()


    def  initialize_biases(self):


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if isinstance(m,ConvModule):
                constant_init(m.bn, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg in zip(self.rtm_cls, self.rtm_reg):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)

    def forward(self,feats):
        cls_scores = []
        bbox_preds = []
        outputs = []
        for idx, x in enumerate(feats):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            #reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            reg_dist = self.rtm_reg[idx](reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            cls_score = torch.sigmoid(cls_score)
            output = [feats[idx],cls_score,reg_dist]
            outputs.append(output)

        return outputs

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init