#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.models.effidehead import Detect, build_effidehead_layer
from utils.general import LOGGER
from utils.torch_utils import model_info
def parse_model(d, ch = 3):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    gd, gw = d['depth_multiple'], d['width_multiple']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone']+d['neck']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_  = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv_C3,Bottleneck, SPPF,C3,RepBlock,SimConv,RepVGGBlock,Transpose,SimSPPF,BepC3]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [C3,RepBlock]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Out:
            pass
        elif m in [Stem,ConvWrapper,Transpose]:
            c1 = ch[f]
            c2 = args[0]
            args = [c1, c2, *args[1:]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        build_type = config.model.build_type
        self.build_type = build_type
        print(build_type)
        if build_type=="yaml":

        #self.mode = config.training_mode
            cfg = config.model.yaml_file
            import yaml  # for torch hub
            from copy import deepcopy
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict
            ch = self.yaml['ch'] = self.yaml.get('ch', 3)
            self.backbone, self.save = parse_model(deepcopy(self.yaml),ch=[ch])  # model, savelist
            self.detect = build_network_yaml(config, channels, num_classes, anchors, num_layers)
        else:
            self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, anchors, num_layers)

#############
        # Init Detect head
        begin_indices = config.model.head.begin_indices
        out_indices_head = config.model.head.out_indices
        self.stride = self.detect.stride
        self.detect.i = begin_indices
        self.detect.f = out_indices_head
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)
        self.info()

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export()
        if self.build_type == "yaml":
        ##############
            number_layer = 0
            y, dt = [], []  # outputs
            for m in self.backbone:

                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                try:
                    number_layer = number_layer+1
                    x = m(x)  # run
                except:
                    print("run error ,error layer: "+str(number_layer-1))

                y.append(x if m.i in self.save else None)  # save output
        ############
        else:
            x = self.backbone(x)
            x = self.neck(x)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)
        return x if export_mode is True else [x, featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self
    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, anchors, num_layers):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    num_anchors = config.model.head.anchors
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    if 'CSP' in config.model.backbone.type:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    head_layers = build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max)

    head = Detect(num_classes, anchors, num_layers, head_layers=head_layers, use_dfl=use_dfl)
    return backbone, neck, head

def build_network_yaml(config, channels, num_classes, anchors, num_layers):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_anchors = config.model.head.anchors
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    channels_list = config.model.head.effidehead_channels
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list)]

    head_layers = build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max)
    head = Detect(num_classes, anchors, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return head
def build_model(cfg, num_classes, device):
    model = Model(cfg, channels=3, num_classes=num_classes, anchors=cfg.model.head.anchors).to(device)
    return model
