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
from copy import deepcopy
def parse_model(d, ch = 3,nc = 0):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    gd, gw = d['depth_multiple'], d['width_multiple']
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone']+d['neck']+d['effidehead']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_  = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv_C3,Bottleneck, SPPF,RepBlock,RepConvBlock1,SimConv,RepConvBlock,RepVGGBlock,QARepVGGBlock,Transpose,SimSPPF,SPSimCSPSPPF,SimCSPSPPF,BepC3, BepBotC3, Conv]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 4)

            args = [c1, c2, *args[1:]]
            if m in [ RepBlock,RepConvBlock,RepConvBlock1]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in [CSPDepthResELAN,CSPRepResELAN, CSPSDepthResELAN, SDepthMP, CSPDepthResELANUni, RepGELANMS,RepHDW, C2f]:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
            args.insert(2, n)  # number of repeats
            n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in [Out]:
            pass
        elif m in [AVG_down]:
            c1 = ch[f]
            c2 = c1
            # c2 = make_divisible(c2 * gw, 8)
            # args = [args[1:]]
        elif m in [Head_layers, Head_out, Head_simota, Head_Depth]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)
            args = [c1,c2, args[1],nc]
        elif m in [Head_DepthUni]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)
            args = [c1,c2, args[1], args[2], nc]
        elif m is Head_PConv:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, args[1],args[2], nc]
        elif m in [Stem,ConvWrapper,Transpose]:
            c1 = ch[f]
            c2 = args[0]
            args = [c1, c2, *args[1:]]
        elif m in [ Focal_Depth]:
            c1 = ch[f]
            c2 = args[0]
            # c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2 ,*args[1:]]

        elif m in [ConvBnAct, FocalTransformer, CoAtNetMBConv, ConvGE, CoAtNetTrans, MBConv_block, CoAtTrans_block,
                   MBConvC3]:
            c1 = ch[f]
            c2 = args[0]
            args = [c1, c2, *args[1:]]
        elif m in [RepGhostC3]:
            c1 = ch[f]
            c2 = args[0]
            c_mid = args[1]
            c2 = make_divisible(c2 * gw, 8)
            c_mid = make_divisible(c_mid * gw, 8)
            args = [c1, c2, c_mid,*args[2:]]
        elif m in [RepGhostBottleneck]:
            c1 = args[0]
            c2 = int(args[3] * args[5])
        elif m is Sequentially_Add:
            c2 = args[0]
            pass
        elif m in [MP1,MPRep]:
            c1 = ch[f]
            c2 = args[0]
            c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m in [EAEF]:
            c1 = args[0]
            c2 = args[0]
        elif m is EAEF_out:
            c1 = args[1]
            c2 = args[1]
            args = [args[0]]
        elif m in [RepNCSPELAN4]:
            c1, c2 = ch[f], args[0]
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
        try:
            num_layers = config.model.head.num_layers
            build_type = config.model.build_type
        except:
            print("build model by office YOLOv6")
            build_type = "office"
        self.build_type = build_type
        # print(build_type)
        if build_type=="yaml":

        #self.mode = config.training_mode
            cfg = config.model.yaml_file
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict
            ch = self.yaml['ch'] = self.yaml.get('ch', 3)
            self.backbone, self.save = parse_model(deepcopy(self.yaml),ch=[ch],nc = num_classes)  # model, savelist
            # self.detect = build_network_yaml(config, channels, num_classes, anchors, num_layers)
            use_dfl = config.model.head.use_dfl
            stride = config.model.head.strides
            try:
                if config.model.target == "SimOTA":
                    use_simota = True
            except:
                use_simota = False



            if use_simota:
                self.detect = Detect_simota(num_classes, anchors, num_layers)
            else:
                self.detect = Detect_yaml(num_classes, anchors, num_layers, use_dfl=use_dfl, stride=stride)
            self.stride = self.detect.stride
            self.detect.initialize_biases()
        else:
            self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, anchors, num_layers,val_loss= None)
            begin_indices = config.model.head.begin_indices
            out_indices_head = config.model.head.out_indices
            self.stride = self.detect.stride
            self.detect.i = begin_indices
            self.detect.f = out_indices_head
            self.detect.initialize_biases()


        # Init weights
        initialize_weights(self)
        self.info()

    def forward(self, x, val_loss = False):
        export_mode = torch.onnx.is_in_onnx_export()
        try:
            self.build_type
        except:
            self.build_type = "office"

        if self.build_type == "yaml":
        ##############
            number_layer = 0
            y, dt = [], []  # outputs
            for m in self.backbone:

                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                # try:
                number_layer = number_layer+1
                x = m(x)  # run
                # except:
                #     print("run error ,error layer: "+str(number_layer-1))

                y.append(x if m.i in self.save else None)  # save output
        ############
        else:
            x = self.backbone(x)
            x = self.neck(x)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x, val_loss)
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


def build_network(config, channels, num_classes, anchors, num_layers,val_loss = None):
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
def build_model(cfg, num_classes, device,img_size):
    model = Model(cfg, channels=3, num_classes=num_classes, anchors=cfg.model.head.anchors).to(device)
    # model = torch.compile(model, backend = 'inductor')
    from yolov6.utils.events import LOGGER
    LOGGER.info("Model Summary: {}".format(get_model_info(model, img_size = img_size,cfg = cfg)))
    LOGGER.info("Because of the use of reparameterization, the number of parameters and floating point operations counted before training "+
                "may not be accurate and need to be verified using eval.py in the validation phase to get the correct information")
    return model

from yolov6.utils.general import dist2bbox
from yolov6.assigners.anchor_generator import generate_anchors
class Detect_yaml(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, num_classes=80, anchors=1, num_layers=3, inplace=True, use_dfl=True, reg_max=16,stride = [8, 16, 32]):  # detection layer
        super().__init__()
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0


    def initialize_biases(self):
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x, val_loss = False):
        if self.training or val_loss:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                cls_output = x[i][1]
                reg_output = x[i][2]
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            if self.nl == 4:
                x = [x[0][0], x[1][0], x[2][0], x[3][0]]           
            elif self.nl == 5:
                x = [x[0][0], x[1][0], x[2][0], x[3][0],x[4][0]]
            elif self.nl == 6:
                x = [x[0][0], x[1][0], x[2][0], x[3][0], x[4][0], x[5][0]]
            else:
                x = [x[0][0], x[1][0], x[2][0]]
            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []

            if self.nl == 4:
                x1 = [x[0][0], x[1][0], x[2][0], x[3][0]]
            elif self.nl == 5:
                x1 = [x[0][0], x[1][0], x[2][0], x[3][0],x[4][0]]
            elif self.nl == 6:
                x1 = [x[0][0], x[1][0], x[2][0], x[3][0], x[4][0], x[5][0]]
            else:
                x1 = [x[0][0], x[1][0], x[2][0]]
            anchor_points, stride_tensor = generate_anchors(
                x1, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x1[0].device, is_eval=True)

            for i in range(self.nl):
                b, _, h, w = x[i][0].shape
                l = h * w
                cls_output = x[i][1]
                reg_output = x[i][2]

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))


                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))

            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)


            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)
class Detect_simota(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, num_classes=80, anchors=1, num_layers=3, inplace=True, head_layers=None):  # detection layer
        super().__init__()
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)

        # Init decouple head


        # Efficient decoupled head layers
    def initialize_biases(self):
        pass



    def forward(self, x, val_loss = False):
        z = []
        for i in range(self.nl):

            cls_output = x[i][0]
            reg_output = x[i][1]
            obj_output = x[i][2]
            if self.training:
                x[i] = torch.cat([reg_output, obj_output, cls_output], 1)
                bs, _, ny, nx = x[i].shape
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            else:
                y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                bs, _, ny, nx = y.shape
                y = y.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                if self.grid[i].shape[2:4] != y.shape[2:4]:
                    d = self.stride.device
                    yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
                    self.grid[i] = torch.stack((xv, yv), 2).view(1, self.na, ny, nx, 2).float()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i] # wh
                else:
                    xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                    wh = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else torch.cat(z, 1)
def get_model_info(model, img_size=640, cfg = None):
    """Get model Params and GFlops.
    Code base on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/model_utils.py
    """
    from thop import profile
    try:  # FLOPs
        # stride = cfg.model.head.strides[-1]  #32/64
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)

        flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
        params /= 1e6
        flops /= 1e9
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        flops *= img_size[0] * img_size[1] / stride / stride * 2  # Gflops
        info = "Params: {:.2f}M, Gflops: {:.2f} for {}x{}".format(params, flops,img_size[0],img_size[1])
    except (ImportError, Exception):
        info = ""
        print("error in get_model_info")
    return info