import torch
import torch.nn as nn
from yolov6.layers.common import ConvModule
import torch.nn.functional as F

#####have bug
class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

from yolov6.layers.misc import SigmoidGeometricMean
class Task_aligned_Head(nn.Module):
    def __init__(self, in_channels, reg_max, class_num,strides, feat_channels = 0, stacked_convs = 6,num_anchors = 1):
        super(Task_aligned_Head, self).__init__()
        self.in_channels = in_channels
        if feat_channels == 0:
            feat_channels = in_channels
        self.feat_channels = feat_channels
        self.inter_convs = nn.ModuleList()
        self.stacked_convs = stacked_convs
        self.strides = [strides]
        self.stride = [(strides,strides)]
        self.anchor_type = 'anchor_free'


        self.base_anchors = self.gen_base_anchors()



        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                ConvModule(
                    chn,
                    in_channels,
                    3,
                    s=1,
                    p=1))
        self.cls_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                           )
        self.reg_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            )
        self.tood_cls = nn.Conv2d(
            self.feat_channels,
            num_anchors * class_num,
            3,
            padding=1)
        self.tood_reg = nn.Conv2d(
            self.feat_channels, 4 * (reg_max + num_anchors), 3, padding=1)
        self.cls_prob_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=1))
        self.reg_offset_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 4 * 2, 3, padding=1))


    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
                ``torch.float32``.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """

        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.stride[level_idx]
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors
    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.strides):
            center = None
            self.scales1 = torch.Tensor(self.strides)

            self.ratios = torch.Tensor([1])
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales1,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors
    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx
    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        self.center_offset = 0.0

        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        self.scale_major = True
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors
    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)
    def deform_sampling(self, feat, offset):
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for feature sampling
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        from torchvision.ops import deform_conv2d
        y = deform_conv2d(feat, offset, weight)
        return y

    def forward(self, x):
        cls_scores = []
        bbox_preds = []
        b, c, h, w = x.shape
        anchor = self.single_level_grid_priors((h, w), 0, device=x.device)
        anchor = torch.cat([anchor for _ in range(b)])

        inter_feats = []
        for inter_conv in self.inter_convs:
            x = inter_conv(x)
            inter_feats.append(x)    ## inter_feats 提取任务交互特征，X^inter，论文公式1,经过n个卷积激活层，列表，保存每一行
        # 将6个X inter concat 256-》1536
        feat = torch.cat(inter_feats, 1)
        # task decomposition
        #  avg_feat是公式中的x^inter，先经过vat，GAP 二元自适应均值汇聚层，由X ^ inter全局平均池化得到
        avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        # 实现了图中灰框的操作，在函数内部实现注意力机制，最后生成X^task
        # 计算Z^task，用两重卷积，论文公式4，用来分类的Z^task
        cls_feat = self.cls_decomp(feat, avg_feat)
        # 用来回归的Z^task
        reg_feat = self.reg_decomp(feat, avg_feat)

        # 分类得分的预测与对齐，计算Z^task，用两重卷积，论文公式4
        # cls_logits为80通道的类别，为P classification scores
        cls_logits = self.tood_cls(cls_feat)
        # cls_prob为M，论文公式7，通道数为1的是否有物体
        cls_prob = self.cls_prob_module(feat)
        # 计算对齐后的得分，论文公式5，通过使用计算P,M的任务交互特征共同考虑两个任务来对齐两个预测,cls_score 80通道
        sigmoid_geometric_mean = SigmoidGeometricMean.apply
        cls_score = sigmoid_geometric_mean(cls_logits, cls_prob)

        # reg prediction and alignment
        # 得到 O∈RH×W×8，用于调整每个位置的预测边界框。具体来说，学习到的空间偏移量使最对齐的锚点能够识别其周围的最佳边界预测
        stride = self.strides[0]
        if self.anchor_type == 'anchor_free':
            scale = Scale(1.0)
            reg_dist = scale(self.tood_reg(reg_feat).exp()).float()
            reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
            a = self.anchor_center(anchor) / stride,reg_dist
            reg_bbox = distance2bbox(
                self.anchor_center(anchor) / stride,
                reg_dist).reshape(b, h, w, 4).permute(0, 3, 1,
                                                      2)  # (b, c, h, w)
        elif self.anchor_type == 'anchor_based':
            reg_dist = self.scale(self.tood_reg(reg_feat)).float()
            reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
            reg_bbox = self.bbox_coder.decode(anchor, reg_dist).reshape(
                b, h, w, 4).permute(0, 3, 1, 2) / stride  # 变为4通道

        # 计算O，论文公式8 reg_offset为O
        reg_offset = self.reg_offset_module(feat)
        # 用可变性二维卷积根据offset对特征进行采样
        bbox_pred = self.deform_sampling(reg_bbox.contiguous(),
                                         reg_offset.contiguous())

        # After deform_sampling, some boxes will become invalid (The
        # left-top point is at the right or bottom of the right-bottom
        # point), which will make the GIoULoss negative.
        invalid_bbox_idx = (bbox_pred[:, [0]] > bbox_pred[:, [2]]) | \
                           (bbox_pred[:, [1]] > bbox_pred[:, [3]])
        invalid_bbox_idx = invalid_bbox_idx.expand_as(bbox_pred)
        bbox_pred = torch.where(invalid_bbox_idx, reg_bbox, bbox_pred)



        return x, cls_score, bbox_pred.half()
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import dynamic_clip_for_onnx
            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes

class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 conv_cfg=None,
                 norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            s=1,
            p=0)


    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.act(feat)

        return feat
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)