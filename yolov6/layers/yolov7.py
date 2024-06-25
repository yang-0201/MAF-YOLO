import torch
import torch.nn as nn


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
class ELAN(nn.Module):
    def __init__(self,c1, c2):
        c_ = c2//4
        super(ELAN, self).__init__()
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(c_, c_, 3, 1)
        self.conv4 = Conv(c_, c_, 3, 1)
        self.conv5 = Conv(c_, c_, 3, 1)
        self.conv6 = Conv(c_, c_, 3, 1)
        self.conv7 = Conv(c2, c2, 1, 1)
    def forward(self,x):
        out1 = self.conv1(x)
        x = self.conv2(x)
        out2 = x
        x = self.conv3(x)
        x = self.conv4(x)
        out3 = x
        x = self.conv5(x)
        x = self.conv6(x)
        out4 = x
        x = torch.cat([out1, out2, out3, out4], dim = 1)
        out = self.conv7(x)
        return out
class E_ELAN(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(E_ELAN, self).__init__()
        c_ = c1//2 # hidden channels
        self.downc = DownC(c1,c2)
        self.conv1 = Conv(c1,c_,1,1)
        self.conv2 = Conv(c1,c_,1,1)
        self.conv3 = Conv(c_, c_, 3, 1)
        self.conv4 = Conv(c_, c_, 3, 1)
        self.conv5 = Conv(c_, c_, 3, 1)
        self.conv6 = Conv(c_, c_, 3, 1)
        self.conv7 = Conv(c_, c_, 3, 1)
        self.conv8 = Conv(c_, c_, 3, 1)
        self.conv9 = Conv(c_ * 5, c2, 1, 1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x = torch.cat([x8, x6, x4, x2, x1], dim = 1)
        x = self.conv9(x)
        return x
class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)
class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2//2, 3, k)
        self.cv3 = Conv(c1, c2//2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)

class MP1(nn.Module):
    def __init__(self, c2, idx = 1):  #c1 = c2
        c_ = c2//2
        self.idx = idx
        super(MP1, self).__init__()
        self.mp = MP()
        if idx == 1:
            self.conv1 = Conv(c2, c_, 1, 1)
            self.conv2 = Conv(c2, c_, 1, 1)
            self.conv3 = Conv(c_, c_, 3, 2)
        elif idx == 2:
            self.conv1 = Conv(c2, c2, 1, 1)
            self.conv2 = Conv(c2, c2, 1, 1)
            self.conv3 = Conv(c2, c2, 3, 2)


    def forward(self, input):
        x1 = self.mp(input)
        x1 = self.conv1(x1)
        x2 = self.conv2(input)
        x2 = self.conv3(x2)
        if self.idx == 1:
            out = torch.cat([x1, x2], dim=1)

        return out

class ELAN_H(nn.Module):
    def __init__(self,c1, c2):
        c__ = c1//4
        c_ = c1//2
        super(ELAN_H, self).__init__()
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(c_, c__, 3, 1)
        self.conv4 = Conv(c__, c__, 3, 1)
        self.conv5 = Conv(c__, c__, 3, 1)
        self.conv6 = Conv(c__, c__, 3, 1)
        self.conv7 = Conv(c1 * 2, c2, 1, 1)
    def forward(self,x):
        out1 = self.conv1(x)
        x = self.conv2(x)
        out2 = x
        x = self.conv3(x)
        out3 = x
        x = self.conv4(x)
        out4 = x
        x = self.conv5(x)
        out5 = x
        x = self.conv6(x)
        out6 = x
        x = torch.cat([out1, out2, out3, out4, out5, out6], dim = 1)
        out = self.conv7(x)
        return out
class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))