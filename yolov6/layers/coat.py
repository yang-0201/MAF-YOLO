import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeAttention(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, n_head, d_k, d_v, out_channels, attn_dropout=0.1, ff_dropout=0.1,
                 attn_bias=False):
        super().__init__()
        self.inp_h = inp_h
        self.inp_w = inp_w
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.Q = nn.Linear(in_channels, n_head * d_k, bias=attn_bias)
        self.K = nn.Linear(in_channels, n_head * d_k, bias=attn_bias)
        self.V = nn.Linear(in_channels, n_head * d_v, bias=attn_bias)
        self.ff = nn.Linear(n_head * d_v, out_channels)
        self.attn_dropout = nn.Dropout2d(attn_dropout)
        self.ff_dropout = nn.Dropout(ff_dropout)
        self.relative_bias = nn.Parameter(
            torch.randn(n_head, ((inp_h << 1) - 1) * ((inp_w << 1) - 1)),
            requires_grad=True
        )
        self.register_buffer('relative_indices', self._get_relative_indices(inp_h, inp_w))

    def _get_relative_indices(self, height, width):
        ticks_y, ticks_x = torch.arange(height), torch.arange(width)
        grid_y, grid_x = torch.meshgrid(ticks_y, ticks_x)
        area = height * width
        out = torch.empty(area, area).fill_(float('nan'))
        for idx_y in range(height):
            for idx_x in range(width):
                rel_indices_y = grid_y - idx_y + height
                rel_indices_x = grid_x - idx_x + width
                flatten_indices = (rel_indices_y * width + rel_indices_x).view(-1)
                out[idx_y * width + idx_x] = flatten_indices
        assert not out.isnan().any(), '`relative_indices` have blank indices'
        assert (out >= 0).all(), '`relative_indices` have negative indices'
        return out.long()

    def _interpolate_relative_bias(self, height, width):
        relative_bias = self.relative_bias.view(1, self.n_head, (self.inp_h << 1) - 1, -1)
        relative_bias = F.interpolate(relative_bias, size=((height << 1) - 1, (width << 1) - 1), mode='bilinear',
                                      align_corners=True)
        return relative_bias.view(self.n_head, -1)

    def update_relative_bias_and_indices(self, height, width):
        self.relative_indices = self._get_relative_indices(height, width)
        self.relative_bias = self._interpolate_relative_bias(height, width)

    def forward(self, x):
        b, c, H, W, h = *x.shape, self.n_head

        len_x = H * W
        x = x.view(b, c, len_x).transpose(-1, -2)
        q = self.Q(x).view(b, len_x, self.n_head, self.d_k).transpose(1, 2)
        k = self.K(x).view(b, len_x, self.n_head, self.d_k).transpose(1, 2)
        v = self.V(x).view(b, len_x, self.n_head, self.d_v).transpose(1, 2)

        if H == self.inp_h and W == self.inp_w:
            relative_indices = self.relative_indices
            relative_bias = self.relative_bias
        else:
            relative_indices = self._get_relative_indices(H, W).to(x.device)
            relative_bias = self._interpolate_relative_bias(H, W)

        relative_indices = relative_indices.view(1, 1, *relative_indices.size()).expand(b, h, -1, -1)
        relative_bias = relative_bias.view(1, relative_bias.size(0), 1, relative_bias.size(1)).expand(b, -1, len_x, -1)
        relative_biases = relative_bias.gather(dim=-1, index=relative_indices)

        similarity = torch.matmul(q, k.transpose(-1, -2)) + relative_biases
        similarity = similarity.softmax(dim=-1)
        similarity = self.attn_dropout(similarity)

        out = torch.matmul(similarity, v)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.n_head * self.d_v)
        out = self.ff(out)
        out = self.ff_dropout(out)
        out = out.transpose(-1, -2).view(b, -1, H, W)
        return out


class FeedForwardRelativeAttention(nn.Module):
    def __init__(self, in_dim, expand_dim, drop_ratio=0.1, act_fn='gelu'):
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, expand_dim, kernel_size=1)
        self.act_fn = get_act_fn(act_fn)
        self.fc2 = nn.Conv2d(expand_dim, in_dim, kernel_size=1)
        self.drop_ratio = drop_ratio

    def forward(self, x):
        x_in = x
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = drop_connect(x, drop_ratio=self.drop_ratio, training=self.training) + x_in
        return x
act_fn_map = {
    'swish': 'silu'
}
class MemoryEfficientSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return MemoryEfficientSwish.apply(x)


class MemoryEfficientMish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        v = 1. + i.exp()
        h = v.log()
        grad_gh = 1./h.cosh().pow_(2)
        grad_hx = i.sigmoid()
        grad_gx = grad_gh *  grad_hx
        grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx
        return grad_output * grad_f


class Mish(nn.Module):
    def forward(self, x):
        return MemoryEfficientMish.apply(x)

memory_efficient_map = {
    'swish': Swish,
    'mish': Mish
}
def get_act_fn(act_fn, prefer_memory_efficient=True):
    if isinstance(act_fn, str):
        if prefer_memory_efficient and act_fn in memory_efficient_map:
            return memory_efficient_map[act_fn]()
        if act_fn in act_fn_map:
            act_fn = act_fn_map[act_fn]
        return getattr(F, act_fn)
    return act_fn


def drop_connect(x, drop_ratio, training=True):
    if not training or drop_ratio == 0:
        return x
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn='mish', ff_dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.act_fn = get_act_fn(act_fn)
        self.dropout = nn.Dropout(ff_dropout)
        self.norm = nn.LayerNorm(in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class TransformerWithRelativeAttention(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, n_head, d_k=None, d_v=None,
                 out_channels=None, attn_dropout=0.1, ff_dropout=0.1,
                 act_fn='gelu', attn_bias=False, expand_ratio=4,
                 use_downsampling=False, **kwargs):
        super().__init__()
        self.use_downsampling = use_downsampling
        self.dropout = ff_dropout
        out_channels = out_channels or in_channels
        d_k = d_k or out_channels // n_head
        d_v = d_v or out_channels // n_head
        if use_downsampling:
            self.pool = nn.MaxPool2d((2, 2))
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(in_channels)
        self.attention = RelativeAttention(inp_h, inp_w, in_channels, n_head, d_k, d_v, out_channels,
                                           attn_dropout=attn_dropout, ff_dropout=ff_dropout, attn_bias=attn_bias)
        self.ff = FeedForwardRelativeAttention(out_channels, out_channels * expand_ratio, drop_ratio=ff_dropout,
                                               act_fn=act_fn)

    def forward(self, x):
        if self.use_downsampling:
            x_stem = self.pool(x)
            x_stem = self.conv(x_stem)
        else:
            x_stem = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        if self.use_downsampling:
            x = self.pool(x)
        x = self.attention(x)
        x = drop_connect(x, self.dropout, training=self.training)
        x = x_stem + x
        x_attn = x
        x = self.ff(x)
        x = drop_connect(x, self.dropout, training=self.training)
        x = x_attn + x
        return x