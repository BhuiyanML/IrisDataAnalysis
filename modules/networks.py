import torch
from torch import nn
from torch.nn import init as init
from torch.nn import functional as F
import torchvision
from torchvision import models
from itertools import chain
from math import ceil
#from modules.layers import ConvOffset2D

import math 
import numpy as np

import logging
from collections import OrderedDict


class fclayer(nn.Module):
    def __init__(self, in_h = 8, in_w = 10, out_n = 6):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.out_n = out_n
        self.fc_list = []
        for i in range(out_n):
            self.fc_list.append(nn.Linear(in_h*in_w, 1))
        self.fc_list = nn.ModuleList(self.fc_list)
    def forward(self, x):
        x = x.reshape(-1, self.out_n, self.in_h, self.in_w)
        outs = []
        for i in range(self.out_n):
            outs.append(self.fc_list[i](x[:, i, :, :].reshape(-1, self.in_h*self.in_w)))
        out = torch.cat(outs, 1)
        return out

class conv(nn.Module):
    def __init__(self, in_channels=512, out_n = 6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_n, kernel_size=1, stride=1, padding='same')
    def forward(self, x):
        x = self.conv(x)
        return x

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


from torch.nn.functional import interpolate

class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([interpolate(p5, size=(p4.shape[2], p4.shape[3]), mode='bilinear'), p4], 1))
        h2 = self.h2(torch.cat([interpolate(h1, size=(p3.shape[2], p3.shape[3]), mode='bilinear'), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


class Head(torch.nn.Module):
    def __init__(self, nc=6, filters=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        c = max(filters[0], self.nc)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c, 3),
                                                           Conv(c, c, 3),
                                                           Conv(c, self.nc, 3)) for x in filters)
        self.linear = torch.nn.ModuleList(torch.nn.Sequential(nn.Linear(self.nl, 1)) for i in range(self.nc))
        #self.weights = nn.Parameter(torch.randn(self.nc, self.nl))

    def forward(self, x_list):
        for i in range(self.nl):
            x_list[i] = torch.mean(self.cls[i](x_list[i]), dim=(2,3)).unsqueeze(-1) #b x self.nc x 1
        
        x = torch.cat(x_list, 2) #b x self.nc x self.nl
        outs = []
        for i in range(self.nc):
            outs.append(self.linear[i](x[:,i,:]).reshape(-1,1)) #b x 1
        
        out = torch.cat(outs, 1)
        
        return out

class ModifiedYOLOBackbone(torch.nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.inp = Conv(1, width[0])
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)
        self.head = Head(num_classes, (width[3], width[4], width[5]))

    def forward(self, x):
        x = self.inp(x)
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v8_n(num_classes: int = 6):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return ModifiedYOLOBackbone(width, depth, num_classes)


def yolo_v8_s(num_classes: int = 6):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return ModifiedYOLOBackbone(width, depth, num_classes)


def yolo_v8_m(num_classes: int = 6):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return ModifiedYOLOBackbone(width, depth, num_classes)


def yolo_v8_l(num_classes: int = 6):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return ModifiedYOLOBackbone(width, depth, num_classes)


def yolo_v8_x(num_classes: int = 6):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return ModifiedYOLOBackbone(width, depth, num_classes)

def set_trainable_attr(m,b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b
    
def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def summary(model, input_shape, batch_size=-1, intputshow=True):

    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) 
                    and not (module == model)) and 'torch' in str(module.__class__):
            if intputshow is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(torch.zeros(input_shape))

    # remove these hooks
    for h in hooks:
        h.remove()

    model_info = ''

    model_info += "-----------------------------------------------------------------------\n"
    line_new = "{:>25}  {:>25} {:>15}".format("Layer (type)", "Input Shape", "Param #")
    model_info += line_new + '\n'
    model_info += "=======================================================================\n"

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        line_new = "{:>25}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )

        total_params += summary[layer]["nb_params"]
        if intputshow is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        model_info += line_new + '\n'

    model_info += "=======================================================================\n"
    model_info += "Total params: {0:,}\n".format(total_params)
    model_info += "Trainable params: {0:,}\n".format(trainable_params)
    model_info += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
    model_info += "-----------------------------------------------------------------------\n"

    return model_info

#outs = tanh(ylogit), outc = tanh(xlogit)) with a loss function 0.5((sin(pred) - outs)^2 + (cos(pred) - outc)^2'


class UNetUp(nn.Module):

    def __init__(self, in_channels, features, out_channels, is_bn = True):
        super().__init__()
        if is_bn:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, 3, padding = 1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(features, out_channels, 2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(features, out_channels, 2, stride=2),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.up(x)


class UNetDown(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn = True):
        super().__init__()

        if is_bn:
            self.down = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.down = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.down(x)

class UNet(nn.Module):

    def __init__(self, num_classes, num_channels, width=4):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(num_channels, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(width, 2*width)
        self.dec3 = UNetDown(2*width, 4*width)
        self.dec4 = UNetDown(4*width, 8*width)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(8*width, 16*width, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(16*width)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(16*width)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(16*width, 8*width, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(8*width)
        self.center3_relu = nn.ReLU(inplace=True)
        
        self.enc4 = UNetUp(16*width, 8*width, 4*width)
        self.enc3 = UNetUp(8*width, 4*width, 2*width)
        self.enc2 = UNetUp(4*width, 2*width, width)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2*width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(width, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1)
        
class UNet_radius_center_linear(nn.Module):

    def __init__(self, num_classes, num_channels, num_params):
        super().__init__()
        
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(4, 8)
        self.dec3 = UNetDown(8, 16)
        self.dec4 = UNetDown(16, 32)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1))
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr1_input = nn.Flatten()
        self.xyr1_linear = nn.Linear(19200, 128)
        self.xyr1_relu = nn.ReLU(inplace=True)
        self.xyr2_dropout = nn.Dropout(p=0)
        self.xyr2_linear = nn.Linear(128, 64)
        self.xyr2_relu = nn.ReLU(inplace=True)
        self.xyr3_dropout = nn.Dropout(p=0)
        self.xyr3_linear = nn.Linear(64, num_params)
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant_(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr2 = self.xyr1_input(center7)
        xyr3 = self.xyr1_linear(xyr2)
        xyr4 = self.xyr1_relu(xyr3)
        xyr5 = self.xyr2_dropout(xyr4)
        xyr6 = self.xyr2_linear(xyr5)
        xyr7 = self.xyr2_relu(xyr6)
        xyr8 = self.xyr3_dropout(xyr7)
        xyr9 = self.xyr3_linear(xyr8)
        

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr9
        
    def encode_xyr(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr2 = self.xyr1_input(center7)
        xyr3 = self.xyr1_linear(xyr2)
        xyr4 = self.xyr1_relu(xyr3)
        xyr5 = self.xyr2_dropout(xyr4)
        xyr6 = self.xyr2_linear(xyr5)
        xyr7 = self.xyr2_relu(xyr6)
        xyr8 = self.xyr3_dropout(xyr7)
        xyr9 = self.xyr3_linear(xyr8)
        
        return xyr9
        
class UNet_radius_center_conv1_red(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        ) # 4 x 320 x 240
        self.dec2 = UNetDown(4, 8)
        # 8 x 160 x 120
        self.dec3 = UNetDown(8, 16)
        # 16 x 80 x 60
        self.dec4 = UNetDown(16, 32)
        # 32 x 40 x 30

        self.center = nn.MaxPool2d(2, stride=2)
        # 32 x 20 x 15
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1))
        #self.center4_relu = nn.ReLU(inplace=True)
        #self.xyr_input = nn.Flatten()
        self.xyr1_conv = nn.Conv2d(64, 32, 3, padding=1)
        self.xyr1_bn = nn.BatchNorm2d(32)
        self.xyr1_relu = nn.ReLU(inplace=True)
        # 32 x 20 x 15
        self.xyr2_input = nn.Flatten()
        self.xyr2_linear = nn.Linear(32 * 20 * 15, num_params)
        self.ang_linear = nn.Linear(32 * 20 * 15, num_extra_params)
        self.ang_tanh = nn.Tanh()
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant_(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        xyr4 = self.xyr2_input(xyr3)
        xyr5 = self.xyr2_linear(xyr4)
        ang1 = self.ang_linear(xyr4)
        ang2 = self.ang_tanh(ang1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr5, ang2
        
    def encode_params(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        xyr4 = self.xyr2_input(xyr3)
        xyr5 = self.xyr2_linear(xyr4)
        ang1 = self.ang_linear(xyr4)
        ang2 = self.ang_tanh(ang1)       
        
        return xyr5, ang2
        
class UNet_radius_center_conv1(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params, residual=False):
        super().__init__()
        self.residual = residual
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        ) # 4 x 320 x 240
        self.dec2 = UNetDown(4, 8)
        # 8 x 160 x 120
        self.dec3 = UNetDown(8, 16)
        # 16 x 80 x 60
        self.dec4 = UNetDown(16, 32)
        # 32 x 40 x 30

        self.center = nn.MaxPool2d(2, stride=2)
        # 32 x 20 x 15
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1))
        #self.center4_relu = nn.ReLU(inplace=True)
        #self.xyr_input = nn.Flatten()
        self.xyr1_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr1_bn = nn.BatchNorm2d(64)
        self.xyr1_relu = nn.ReLU(inplace=True)
        
        self.xyr2_input = nn.Flatten()
        self.xyr2_linear = nn.Linear(64 * 20 * 15, num_params)
        self.ang_linear = nn.Linear(64 * 20 * 15, num_extra_params)
        self.ang_tanh = nn.Tanh()
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant_(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
            
        xyr4 = self.xyr2_input(xyr3)
        xyr5 = self.xyr2_linear(xyr4)
        ang1 = self.ang_linear(xyr4)
        ang2 = self.ang_tanh(ang1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr5, ang2
        
    def encode_params(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_input(xyr3)
        xyr5 = self.xyr2_linear(xyr4)
        ang1 = self.ang_linear(xyr4)
        ang2 = self.ang_tanh(ang1)
        
        return xyr5, ang2

class UNet_radius_center_conv2(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params, residual=False):
        super().__init__()
        self.residual = residual
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        ) # 4 x 320 x 240
        self.dec2 = UNetDown(4, 8)
        # 8 x 160 x 120
        self.dec3 = UNetDown(8, 16)
        # 16 x 80 x 60
        self.dec4 = UNetDown(16, 32)
        # 32 x 40 x 30

        self.center = nn.MaxPool2d(2, stride=2)
        # 32 x 20 x 15
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1))
        #self.center4_relu = nn.ReLU(inplace=True)
        #self.xyr_input = nn.Flatten()
        self.xyr1_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr1_bn = nn.BatchNorm2d(64)
        self.xyr1_relu = nn.ReLU(inplace=True)
        # 32 x 20 x 15
        self.xyr2_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(64)
        self.xyr2_relu = nn.ReLU(inplace=True)
        # 16 x 20 x 15
        self.xyr3_input = nn.Flatten()
        self.xyr3_linear = nn.Linear(64 * 20 * 15, num_params)
        self.ang_linear = nn.Linear(64 * 20 * 15, num_extra_params)
        self.ang_tanh = nn.Tanh()
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant_(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_input(xyr6)
        xyr8 = self.xyr3_linear(xyr7)
        ang1 = self.ang_linear(xyr7)
        ang2 = self.ang_tanh(ang1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr8, ang2
        
    def encode_params(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_input(xyr6)
        xyr8 = self.xyr3_linear(xyr7)
        ang1 = self.ang_linear(xyr7)
        ang2 = self.ang_tanh(ang1)
        
        return xyr8, ang2
        
class UNet_radius_center_conv3(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params, residual=False):
        super().__init__()
        self.residual = residual
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(4, 8)
        self.dec3 = UNetDown(8, 16)
        self.dec4 = UNetDown(16, 32)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1)
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr1_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr1_bn = nn.BatchNorm2d(64)
        self.xyr1_relu = nn.ReLU(inplace=True)
        # 64 x 10 x 7
        self.xyr2_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(64)
        self.xyr2_relu = nn.ReLU(inplace=True)
        # 64 x 5 x 3
        self.xyr3_conv = nn.Conv2d(64, 64, 3, padding =1)
        self.xyr3_bn = nn.BatchNorm2d(64)
        self.xyr3_relu = nn.ReLU(inplace=True)
        # 64 x 2 x 1
        self.xyr4_input = nn.Flatten()
        self.xyr4_linear = nn.Linear(64 * 20 * 15, num_params)
        self.ang_linear = nn.Linear(64 * 20 * 15, num_extra_params)
        self.ang_tanh = nn.Tanh()
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant_(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6 
        
        xyr10 = self.xyr4_input(xyr9)
        xyr11 = self.xyr4_linear(xyr10)
        ang1 = self.ang_linear(xyr10)
        ang2 = self.ang_tanh(ang1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr11, ang2
        
    def encode_params(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6 
        
        xyr10 = self.xyr4_input(xyr9)
        xyr11 = self.xyr4_linear(xyr10)
        ang1 = self.ang_linear(xyr10)
        ang2 = self.ang_tanh(ang1)
        
        return xyr11, ang2
        
class UNet_radius_center_conv4(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params, residual=False):
        super().__init__()
        self.residual = residual
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(4, 8)
        self.dec3 = UNetDown(8, 16)
        self.dec4 = UNetDown(16, 32)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1)
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr1_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr1_bn = nn.BatchNorm2d(64)
        self.xyr1_relu = nn.ReLU(inplace=True)
        # 32 x 20 x 15
        self.xyr2_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(64)
        self.xyr2_relu = nn.ReLU(inplace=True)
        # 16 x 20 x 15
        self.xyr3_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr3_bn = nn.BatchNorm2d(64)
        self.xyr3_relu = nn.ReLU(inplace=True)
        # 8 x 20 x 15
        self.xyr4_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr4_bn = nn.BatchNorm2d(64)
        self.xyr4_relu = nn.ReLU(inplace=True)
        # 4 x 20 x 15
        self.xyr5_input = nn.Flatten()
        self.xyr5_linear = nn.Linear(64 * 20 * 15, num_params)
        self.ang_linear = nn.Linear(64 * 20 * 15, num_extra_params)
        self.ang_tanh = nn.Tanh()
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant_(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6
        
        xyr10 = self.xyr4_conv(xyr9)
        xyr11 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr11)
        
        if self.residual:
            xyr12 = xyr12 + xyr9
        
        xyr13 = self.xyr5_input(xyr12)
        xyr14 = self.xyr5_linear(xyr13)
        ang1 = self.ang_linear(xyr13)
        ang2 = self.ang_tanh(ang1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr14, ang2
        
    def encode_xyr(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6
        
        xyr10 = self.xyr4_conv(xyr9)
        xyr11 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr11)
        
        if self.residual:
            xyr12 = xyr12 + xyr9
        
        xyr13 = self.xyr5_input(xyr12)
        xyr14 = self.xyr5_linear(xyr13)
        ang1 = self.ang_linear(xyr13)
        ang2 = self.ang_tanh(ang1)
        
        return xyr14, ang2
        
class UNet_radius_center_conv10(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params, residual=False):
        super().__init__()
        self.residual = residual
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(4, 8)
        self.dec3 = UNetDown(8, 16)
        self.dec4 = UNetDown(16, 32)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1)
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr1_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr1_bn = nn.BatchNorm2d(64)
        self.xyr1_relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.xyr2_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(64)
        self.xyr2_relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.xyr3_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr3_bn = nn.BatchNorm2d(64)
        self.xyr3_relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.xyr4_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr4_bn = nn.BatchNorm2d(64)
        self.xyr4_relu = nn.ReLU(inplace=True)
        
        self.xyr5_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr5_bn = nn.BatchNorm2d(64)
        self.xyr5_relu = nn.ReLU(inplace=True)
        
        self.xyr6_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr6_bn = nn.BatchNorm2d(64)
        self.xyr6_relu = nn.ReLU(inplace=True)
        
        self.xyr7_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr7_bn = nn.BatchNorm2d(64)
        self.xyr7_relu = nn.ReLU(inplace=True)
        
        self.xyr8_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr8_bn = nn.BatchNorm2d(64)
        self.xyr8_relu = nn.ReLU(inplace=True)
        
        self.xyr9_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr9_bn = nn.BatchNorm2d(64)
        self.xyr9_relu = nn.ReLU(inplace=True)
        
        self.xyr10_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr10_bn = nn.BatchNorm2d(64)
        self.xyr10_relu = nn.ReLU(inplace=True)
        
        # 64 x 20 x 15
        self.xyr11_input = nn.Flatten()
        self.xyr11_linear = nn.Linear(64 * 20 * 15, num_params)
        self.ang_linear = nn.Linear(64 * 20 * 15, num_extra_params)
        self.ang_tanh = nn.Tanh()
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant_(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6
        
        xyr10 = self.xyr4_conv(xyr9)
        xyr11 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr11)
        
        if self.residual:
            xyr12 = xyr12 + xyr9
        
        xyr13 = self.xyr5_conv(xyr12)
        xyr14 = self.xyr5_bn(xyr13)
        xyr15 = self.xyr5_relu(xyr14)
        
        if self.residual:
            xyr15 = xyr15 + xyr12
            
        xyr16 = self.xyr6_conv(xyr15)
        xyr17 = self.xyr6_bn(xyr16)
        xyr18 = self.xyr6_relu(xyr17)
        
        if self.residual:
            xyr18 = xyr18 + xyr15
        
        xyr19 = self.xyr7_conv(xyr18)
        xyr20 = self.xyr7_bn(xyr19)
        xyr21 = self.xyr7_relu(xyr20)
        
        if self.residual:
            xyr21 = xyr21 + xyr18
        
        xyr22 = self.xyr8_conv(xyr21)
        xyr23 = self.xyr8_bn(xyr22)
        xyr24 = self.xyr8_relu(xyr23)
        
        if self.residual:
            xyr24 = xyr24 + xyr21
        
        xyr25 = self.xyr9_conv(xyr24)
        xyr26 = self.xyr9_bn(xyr25)
        xyr27 = self.xyr9_relu(xyr26)
        
        if self.residual:
            xyr27 = xyr27 + xyr24
        
        xyr28 = self.xyr10_conv(xyr27)
        xyr29 = self.xyr10_bn(xyr28)
        xyr30 = self.xyr10_relu(xyr29)
        
        if self.residual:
            xyr30 = xyr30 + xyr27
        
        xyr31 = self.xyr11_input(xyr30)
        xyr32 = self.xyr11_linear(xyr31)
        ang1 = self.ang_linear(xyr31)
        ang2 = self.ang_tanh(ang1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr32, ang2
        
    def encode_params(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6
        
        xyr10 = self.xyr4_conv(xyr9)
        xyr11 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr11)
        
        if self.residual:
            xyr12 = xyr12 + xyr9
        
        xyr13 = self.xyr5_conv(xyr12)
        xyr14 = self.xyr5_bn(xyr13)
        xyr15 = self.xyr5_relu(xyr14)
        
        if self.residual:
            xyr15 = xyr15 + xyr12
            
        xyr16 = self.xyr6_conv(xyr15)
        xyr17 = self.xyr6_bn(xyr16)
        xyr18 = self.xyr6_relu(xyr17)
        
        if self.residual:
            xyr18 = xyr18 + xyr15
        
        xyr19 = self.xyr7_conv(xyr18)
        xyr20 = self.xyr7_bn(xyr19)
        xyr21 = self.xyr7_relu(xyr20)
        
        if self.residual:
            xyr21 = xyr21 + xyr18
        
        xyr22 = self.xyr8_conv(xyr21)
        xyr23 = self.xyr8_bn(xyr22)
        xyr24 = self.xyr8_relu(xyr23)
        
        if self.residual:
            xyr24 = xyr24 + xyr21
        
        xyr25 = self.xyr9_conv(xyr24)
        xyr26 = self.xyr9_bn(xyr25)
        xyr27 = self.xyr9_relu(xyr26)
        
        if self.residual:
            xyr27 = xyr27 + xyr24
        
        xyr28 = self.xyr10_conv(xyr27)
        xyr29 = self.xyr10_bn(xyr28)
        xyr30 = self.xyr10_relu(xyr29)
        
        if self.residual:
            xyr30 = xyr30 + xyr27
        
        xyr31 = self.xyr11_input(xyr30)
        xyr32 = self.xyr11_linear(xyr31)
        ang1 = self.ang_linear(xyr31)
        ang2 = self.ang_tanh(ang1)

        return xyr32, ang2

class UNet_radius_center_conv10_change(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params, complexity = 2, residual=False):
        super().__init__()
        self.residual = residual
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, complexity*4, 3, padding = 1),
            nn.BatchNorm2d(complexity*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(complexity*4, complexity*4, 3, padding = 1),
            nn.BatchNorm2d(complexity*4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(complexity*4, complexity*8)
        self.dec3 = UNetDown(complexity*8, complexity*16)
        self.dec4 = UNetDown(complexity*16, complexity*32)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(complexity*32, complexity*64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(complexity*64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(complexity*64)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(complexity*64, complexity*32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(complexity*32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1)
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr1_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr1_bn = nn.BatchNorm2d(complexity*64)
        self.xyr1_relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.xyr2_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(complexity*64)
        self.xyr2_relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.xyr3_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr3_bn = nn.BatchNorm2d(complexity*64)
        self.xyr3_relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.xyr4_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr4_bn = nn.BatchNorm2d(complexity*64)
        self.xyr4_relu = nn.ReLU(inplace=True)
        
        self.xyr5_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr5_bn = nn.BatchNorm2d(complexity*64)
        self.xyr5_relu = nn.ReLU(inplace=True)
        
        self.xyr6_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr6_bn = nn.BatchNorm2d(complexity*64)
        self.xyr6_relu = nn.ReLU(inplace=True)
        
        self.xyr7_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr7_bn = nn.BatchNorm2d(complexity*64)
        self.xyr7_relu = nn.ReLU(inplace=True)
        
        self.xyr8_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr8_bn = nn.BatchNorm2d(complexity*64)
        self.xyr8_relu = nn.ReLU(inplace=True)
        
        self.xyr9_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr9_bn = nn.BatchNorm2d(complexity*64)
        self.xyr9_relu = nn.ReLU(inplace=True)
        
        self.xyr10_conv = nn.Conv2d(complexity*64, complexity*64, 3, padding = 1)
        self.xyr10_bn = nn.BatchNorm2d(complexity*64)
        self.xyr10_relu = nn.ReLU(inplace=True)
        
        # 64 x 20 x 15
        self.xyr11_dim_red1 = nn.Conv2d(complexity*64, complexity*32, 3, padding = 1)
        self.xyr11_dim_red2 = nn.Conv2d(complexity*32, complexity*16, 3, padding = 1)
        self.xyr11_dim_red3 = nn.Conv2d(complexity*16, complexity*8, 3, padding = 1)
        self.xyr11_dim_red4 = nn.Conv2d(complexity*8, complexity*4, 3, padding = 1)
        self.xyr11_input = nn.Flatten()
        self.xyr12_linear = nn.Linear(complexity * 4 * 20 * 15, complexity * 4 * 20 * 15)
        self.xyr13_linear = nn.Linear(complexity * 4 * 20 * 15, num_params)
        self.ang1_linear = nn.Linear(complexity * 4 * 20 * 15, complexity * 4 * 20 * 15)
        self.ang2_linear = nn.Linear(complexity * 4 * 20 * 15, num_extra_params)
        self.ang3_tanh = nn.Tanh()
        
        self.enc4 = UNetUp(complexity*64, complexity*32, complexity*16)
        self.enc3 = UNetUp(complexity*32, complexity*16, complexity*8)
        self.enc2 = UNetUp(complexity*16, complexity*8, complexity*4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(complexity*8, complexity*4, 3, padding = 1),
            nn.BatchNorm2d(complexity*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(complexity*4, complexity*4, 3, padding = 1),
            nn.BatchNorm2d(complexity*4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(complexity*4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                #print(m.bias)
                #if m.bias:
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                     

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6
        
        xyr10 = self.xyr4_conv(xyr9)
        xyr11 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr11)
        
        if self.residual:
            xyr12 = xyr12 + xyr9
        
        xyr13 = self.xyr5_conv(xyr12)
        xyr14 = self.xyr5_bn(xyr13)
        xyr15 = self.xyr5_relu(xyr14)
        
        if self.residual:
            xyr15 = xyr15 + xyr12
            
        xyr16 = self.xyr6_conv(xyr15)
        xyr17 = self.xyr6_bn(xyr16)
        xyr18 = self.xyr6_relu(xyr17)
        
        if self.residual:
            xyr18 = xyr18 + xyr15
        
        xyr19 = self.xyr7_conv(xyr18)
        xyr20 = self.xyr7_bn(xyr19)
        xyr21 = self.xyr7_relu(xyr20)
        
        if self.residual:
            xyr21 = xyr21 + xyr18
        
        xyr22 = self.xyr8_conv(xyr21)
        xyr23 = self.xyr8_bn(xyr22)
        xyr24 = self.xyr8_relu(xyr23)
        
        if self.residual:
            xyr24 = xyr24 + xyr21
        
        xyr25 = self.xyr9_conv(xyr24)
        xyr26 = self.xyr9_bn(xyr25)
        xyr27 = self.xyr9_relu(xyr26)
        
        if self.residual:
            xyr27 = xyr27 + xyr24
        
        xyr28 = self.xyr10_conv(xyr27)
        xyr29 = self.xyr10_bn(xyr28)
        xyr30 = self.xyr10_relu(xyr29)
        
        if self.residual:
            xyr30 = xyr30 + xyr27
        
        xyr31 = self.xyr11_dim_red1(xyr30) 
        xyr32 = self.xyr11_dim_red2(xyr31)
        xyr33 = self.xyr11_dim_red3(xyr32)
        xyr34 = self.xyr11_dim_red4(xyr33)
        
        xyr35 = self.xyr11_input(xyr34)
        xyr36 = self.xyr12_linear(xyr35)
        
        if self.residual:
            xyr36 = xyr36 + xyr35
        
        xyr37 = self.xyr13_linear(xyr36)
        
        ang1 = self.ang1_linear(xyr35)
        
        if self.residual:
            ang1 = ang1 + xyr35
        
        ang2 = self.ang2_linear(ang1)
        
        #ang2 = self.ang_tanh(ang1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))
        final = self.final(enc1)
                
        return final, xyr37, ang2
        
    def encode_params(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6
        
        xyr10 = self.xyr4_conv(xyr9)
        xyr11 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr11)
        
        if self.residual:
            xyr12 = xyr12 + xyr9
        
        xyr13 = self.xyr5_conv(xyr12)
        xyr14 = self.xyr5_bn(xyr13)
        xyr15 = self.xyr5_relu(xyr14)
        
        if self.residual:
            xyr15 = xyr15 + xyr12
            
        xyr16 = self.xyr6_conv(xyr15)
        xyr17 = self.xyr6_bn(xyr16)
        xyr18 = self.xyr6_relu(xyr17)
        
        if self.residual:
            xyr18 = xyr18 + xyr15
        
        xyr19 = self.xyr7_conv(xyr18)
        xyr20 = self.xyr7_bn(xyr19)
        xyr21 = self.xyr7_relu(xyr20)
        
        if self.residual:
            xyr21 = xyr21 + xyr18
        
        xyr22 = self.xyr8_conv(xyr21)
        xyr23 = self.xyr8_bn(xyr22)
        xyr24 = self.xyr8_relu(xyr23)
        
        if self.residual:
            xyr24 = xyr24 + xyr21
        
        xyr25 = self.xyr9_conv(xyr24)
        xyr26 = self.xyr9_bn(xyr25)
        xyr27 = self.xyr9_relu(xyr26)
        
        if self.residual:
            xyr27 = xyr27 + xyr24
        
        xyr28 = self.xyr10_conv(xyr27)
        xyr29 = self.xyr10_bn(xyr28)
        xyr30 = self.xyr10_relu(xyr29)
        
        if self.residual:
            xyr30 = xyr30 + xyr27
        
        xyr31 = self.xyr11_input(xyr30)
        xyr32 = self.xyr12_linear(xyr31)
        
        if self.residual:
            xyr32 = xyr32 + xyr31
        
        xyr33 = self.xyr13_linear(xyr32)
        
        ang1 = self.ang1_linear(xyr31)
        
        if self.residual:
            ang1 = ang1 + xyr31
        
        ang2 = self.ang2_linear(ang1)

        return xyr37, ang2


class UNet_radius_center_denseconv4(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params, shape=320, is_ang_tanh = True, is_param_hardtanh = True, silu = True, bn = False, dense_bn = False):
        super().__init__()
        self.bn = bn
        self.dense_bn = dense_bn
        self.is_ang_tanh = is_ang_tanh
        self.is_param_hardtanh = is_param_hardtanh
        self.shape = shape
        self.silu = silu
        if self.silu:
            self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        if bn:
            self.first = nn.Sequential(
                nn.Conv2d(num_channels, 4, 3, padding = 1),
                nn.BatchNorm2d(4),
                self.activation,
                nn.Conv2d(4, 4, 3, padding = 1),
                nn.BatchNorm2d(4),
                self.activation
            )
        else:
            self.first = nn.Sequential(
                nn.Conv2d(num_channels, 4, 3, padding = 1),
                self.activation,
                nn.Conv2d(4, 4, 3, padding = 1),
                self.activation
            )
        self.dec2 = UNetDown(4, 8, is_batchnorm = bn, is_silu = self.silu)
        self.dec3 = UNetDown(8, 16, is_batchnorm = bn, is_silu = self.silu)
        self.dec4 = UNetDown(16, 32, is_batchnorm = bn, is_silu = self.silu)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = self.activation
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = self.activation
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = self.activation
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1)
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr1_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr1_bn = nn.BatchNorm2d(64)
        self.xyr1_relu = self.activation
        # 64 x 20 x 15
        self.xyr2_conv = nn.Conv2d(64*2, 64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(64)
        self.xyr2_relu = self.activation
        # 64 x 20 x 15
        self.xyr3_conv = nn.Conv2d(64*3, 64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(64)
        self.xyr3_relu = self.activation
        # 64 x 20 x 15
        self.xyr4_conv = nn.Conv2d(64*4, 64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(64)
        self.xyr4_relu = self.activation
        
        # 64 x 20 x 15
        self.xyr5_input = nn.Flatten()
        self.xyr5_linear = nn.Linear(64 * 20 * 15, num_params)
        self.xyr5_hardtanh = nn.Hardtanh(min_val=0, max_val=self.shape, inplace=True)
        self.ang_linear = nn.Linear(64 * 20 * 15, num_extra_params)
        self.ang_tanh = nn.Tanh()
        
        self.enc4 = UNetUp(64, 32, 16, is_batchnorm = bn, is_silu = self.silu)
        self.enc3 = UNetUp(32, 16, 8, is_batchnorm = bn, is_silu = self.silu)
        self.enc2 = UNetUp(16, 8, 4, is_batchnorm = bn, is_silu = self.silu)
        if bn:
            self.enc1 = nn.Sequential(
                nn.Conv2d(8, 4, 3, padding = 1),
                nn.BatchNorm2d(4),
                self.activation,
                nn.Conv2d(4, 4, 3, padding = 1),
                nn.BatchNorm2d(4),
                self.activation,
            )
        else:
            self.enc1 = nn.Sequential(
                nn.Conv2d(8, 4, 3, padding = 1),
                self.activation,
                nn.Conv2d(4, 4, 3, padding = 1),
                self.activation,
            )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if not self.silu:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()


    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        if self.bn:
            center2 = self.center_bn(center2)
        center4 = self.center_relu(center2)
        center5 = self.center2(center4)
        if self.bn:
            center5 = self.center2_bn(center5)
        center7 = self.center2relu(center5)
        center8 = self.center3(center7)
        if self.bn:
            center8 = self.center3_bn(center8)
        center10 = self.center3_relu(center8)
        
        xyr1 = self.xyr1_conv(center7)
        if self.dense_bn:
            xyr1 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr1)
        
        xyr4 = self.xyr2_conv(torch.cat([xyr3, center7], 1))
        if self.dense_bn:
            xyr4 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr4)
        
        xyr7 = self.xyr3_conv(torch.cat([xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr7 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr7)
        
        xyr10 = self.xyr4_conv(torch.cat([xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr10 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr10)
        
        xyr13 = self.xyr5_input(xyr12)
        xyr14 = self.xyr5_linear(xyr13)
        if self.is_param_hardtanh:
            xyr14 = self.xyr5_hardtanh(xyr14)
        ang1 = self.ang_linear(xyr13)
        if self.is_ang_tanh:
            ang1 = self.ang_tanh(ang1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr14, ang1
        
    def encode_params(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        if self.bn:
            center2 = self.center_bn(center2)
        center4 = self.center_relu(center2)
        center5 = self.center2(center4)
        if self.bn:
            center5 = self.center2_bn(center5)
        center7 = self.center2relu(center5)
        center8 = self.center3(center7)
        if self.bn:
            center8 = self.center3_bn(center8)
        center10 = self.center3_relu(center8)
        
        xyr1 = self.xyr1_conv(center7)
        if self.dense_bn:
            xyr1 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr1)
        
        xyr4 = self.xyr2_conv(torch.cat([xyr3, center7], 1))
        if self.dense_bn:
            xyr4 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr4)
        
        xyr7 = self.xyr3_conv(torch.cat([xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr7 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr7)
        
        xyr10 = self.xyr4_conv(torch.cat([xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr10 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr10)
        
        xyr13 = self.xyr11_input(xyr12)
        xyr14 = self.xyr11_linear(xyr13)
        if self.is_param_hardtanh:
            xyr14 = self.xyr5_hardtanh(xyr14)
        ang1 = self.ang_linear(xyr13)
        if self.ang_tanh:
            ang1 = self.ang_tanh(ang1)

        return xyr14, ang1

class UNet_radius_center_denseconv10(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params, shape=320, is_ang_tanh = True, is_param_hardtanh = True, silu = True, bn = False, dense_bn = False):
        super().__init__()
        self.bn = bn
        self.dense_bn = dense_bn
        self.is_ang_tanh = is_ang_tanh
        self.is_param_hardtanh = is_param_hardtanh
        self.shape = shape
        self.silu = silu
        if self.silu:
            self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        if bn:
            self.first = nn.Sequential(
                nn.Conv2d(num_channels, 4, 3, padding = 1),
                nn.BatchNorm2d(4),
                self.activation,
                nn.Conv2d(4, 4, 3, padding = 1),
                nn.BatchNorm2d(4),
                self.activation
            )
        else:
            self.first = nn.Sequential(
                nn.Conv2d(num_channels, 4, 3, padding = 1),
                self.activation,
                nn.Conv2d(4, 4, 3, padding = 1),
                self.activation
            )
        self.dec2 = UNetDown(4, 8, is_batchnorm = bn, is_silu = self.silu)
        self.dec3 = UNetDown(8, 16, is_batchnorm = bn, is_silu = self.silu)
        self.dec4 = UNetDown(16, 32, is_batchnorm = bn, is_silu = self.silu)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = self.activation
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = self.activation
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = self.activation
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1)
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr1_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr1_bn = nn.BatchNorm2d(64)
        self.xyr1_relu = self.activation
        # 64 x 20 x 15
        self.xyr2_conv = nn.Conv2d(64*2, 64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(64)
        self.xyr2_relu = self.activation
        # 64 x 20 x 15
        self.xyr3_conv = nn.Conv2d(64*3, 64, 3, padding = 1)
        self.xyr3_bn = nn.BatchNorm2d(64)
        self.xyr3_relu = self.activation
        # 64 x 20 x 15
        self.xyr4_conv = nn.Conv2d(64*4, 64, 3, padding = 1)
        self.xyr4_bn = nn.BatchNorm2d(64)
        self.xyr4_relu = self.activation
        # 64 x 20 x 15
        self.xyr5_conv = nn.Conv2d(64*5, 64, 3, padding = 1)
        self.xyr5_bn = nn.BatchNorm2d(64)
        self.xyr5_relu = self.activation
        # 64 x 20 x 15
        self.xyr6_conv = nn.Conv2d(64*6, 64, 3, padding = 1)
        self.xyr6_bn = nn.BatchNorm2d(64)
        self.xyr6_relu = self.activation
        # 64 x 20 x 15
        self.xyr7_conv = nn.Conv2d(64*7, 64, 3, padding = 1)
        self.xyr7_bn = nn.BatchNorm2d(64)
        self.xyr7_relu = self.activation
        # 64 x 20 x 15
        self.xyr8_conv = nn.Conv2d(64*8, 64, 3, padding = 1)
        self.xyr8_bn = nn.BatchNorm2d(64)
        self.xyr8_relu = self.activation
        # 64 x 20 x 15
        self.xyr9_conv = nn.Conv2d(64*9, 64, 3, padding = 1)
        self.xyr9_bn = nn.BatchNorm2d(64)
        self.xyr9_relu = self.activation
        # 64 x 20 x 15
        self.xyr10_conv = nn.Conv2d(64*10, 64, 3, padding = 1)
        self.xyr10_bn = nn.BatchNorm2d(64)
        self.xyr10_relu = self.activation
        
        # 64 x 20 x 15
        self.xyr11_input = nn.Flatten()
        self.xyr11_linear = nn.Linear(64 * 20 * 15, num_params)
        self.xyr11_hardtanh = nn.Hardtanh(min_val=0, max_val=self.shape, inplace=True)
        self.ang_linear = nn.Linear(64 * 20 * 15, num_extra_params)
        self.ang_tanh = nn.Tanh()
        
        self.enc4 = UNetUp(64, 32, 16, is_batchnorm = bn, is_silu = self.silu)
        self.enc3 = UNetUp(32, 16, 8, is_batchnorm = bn, is_silu = self.silu)
        self.enc2 = UNetUp(16, 8, 4, is_batchnorm = bn, is_silu = self.silu)
        if bn:
            self.enc1 = nn.Sequential(
                nn.Conv2d(8, 4, 3, padding = 1),
                nn.BatchNorm2d(4),
                self.activation,
                nn.Conv2d(4, 4, 3, padding = 1),
                nn.BatchNorm2d(4),
                self.activation,
            )
        else:
            self.enc1 = nn.Sequential(
                nn.Conv2d(8, 4, 3, padding = 1),
                self.activation,
                nn.Conv2d(4, 4, 3, padding = 1),
                self.activation,
            )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if not self.silu:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()


    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        if self.bn:
            center2 = self.center_bn(center2)
        center4 = self.center_relu(center2)
        center5 = self.center2(center4)
        if self.bn:
            center5 = self.center2_bn(center5)
        center7 = self.center2relu(center5)
        center8 = self.center3(center7)
        if self.bn:
            center8 = self.center3_bn(center8)
        center10 = self.center3_relu(center8)
        
        xyr1 = self.xyr1_conv(center7)
        if self.dense_bn:
            xyr1 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr1)
        
        xyr4 = self.xyr2_conv(torch.cat([xyr3, center7], 1))
        if self.dense_bn:
            xyr4 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr4)
        
        xyr7 = self.xyr3_conv(torch.cat([xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr7 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr7)
        
        xyr10 = self.xyr4_conv(torch.cat([xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr10 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr10)

        xyr13 = self.xyr5_conv(torch.cat([xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr13 = self.xyr5_bn(xyr13)
        xyr15 = self.xyr5_relu(xyr13)

        xyr16 = self.xyr6_conv(torch.cat([xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr16 = self.xyr6_bn(xyr16)
        xyr18 = self.xyr4_relu(xyr16)

        xyr19 = self.xyr7_conv(torch.cat([xyr18, xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr19 = self.xyr7_bn(xyr19)
        xyr21 = self.xyr7_relu(xyr19)

        xyr22 = self.xyr8_conv(torch.cat([xyr21, xyr18, xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr22 = self.xyr8_bn(xyr22)
        xyr24 = self.xyr8_relu(xyr22)

        xyr25 = self.xyr9_conv(torch.cat([xyr24, xyr21, xyr18, xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr25 = self.xyr9_bn(xyr25)
        xyr27 = self.xyr9_relu(xyr25)

        xyr28 = self.xyr10_conv(torch.cat([xyr27, xyr24, xyr21, xyr18, xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr28 = self.xyr10_bn(xyr28)
        xyr30 = self.xyr10_relu(xyr28)
        
        xyr31 = self.xyr11_input(xyr30)
        xyr32 = self.xyr11_linear(xyr31)
        if self.is_param_hardtanh:
            xyr32 = self.xyr11_hardtanh(xyr32)
        ang1 = self.ang_linear(xyr31)
        if self.ang_tanh:
            ang1 = self.ang_tanh(ang1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr32, ang1
        
    def encode_params(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        if self.bn:
            center2 = self.center_bn(center2)
        center4 = self.center_relu(center2)
        center5 = self.center2(center4)
        if self.bn:
            center5 = self.center2_bn(center5)
        center7 = self.center2relu(center5)
        center8 = self.center3(center7)
        if self.bn:
            center8 = self.center3_bn(center8)
        center10 = self.center3_relu(center8)
        
        xyr1 = self.xyr1_conv(center7)
        if self.dense_bn:
            xyr1 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr1)
        
        xyr4 = self.xyr2_conv(torch.cat([xyr3, center7], 1))
        if self.dense_bn:
            xyr4 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr4)
        
        xyr7 = self.xyr3_conv(torch.cat([xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr7 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr7)
        
        xyr10 = self.xyr4_conv(torch.cat([xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr10 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr10)

        xyr13 = self.xyr5_conv(torch.cat([xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr13 = self.xyr5_bn(xyr13)
        xyr15 = self.xyr5_relu(xyr13)

        xyr16 = self.xyr6_conv(torch.cat([xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr16 = self.xyr6_bn(xyr16)
        xyr18 = self.xyr4_relu(xyr16)

        xyr19 = self.xyr7_conv(torch.cat([xyr18, xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr19 = self.xyr7_bn(xyr19)
        xyr21 = self.xyr7_relu(xyr19)

        xyr22 = self.xyr8_conv(torch.cat([xyr21, xyr18, xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr22 = self.xyr8_bn(xyr22)
        xyr24 = self.xyr8_relu(xyr22)

        xyr25 = self.xyr9_conv(torch.cat([xyr24, xyr21, xyr18, xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr25 = self.xyr9_bn(xyr25)
        xyr27 = self.xyr9_relu(xyr25)

        xyr28 = self.xyr10_conv(torch.cat([xyr27, xyr24, xyr21, xyr18, xyr15, xyr12, xyr9, xyr6, xyr3, center7], 1))
        if self.dense_bn:
            xyr28 = self.xyr10_bn(xyr28)
        xyr30 = self.xyr10_relu(xyr28)
        
        xyr31 = self.xyr11_input(xyr30)
        xyr32 = self.xyr11_linear(xyr31)
        if self.is_param_hardtanh:
            xyr32 = self.xyr11_hardtanh(xyr32)
        ang1 = self.ang_linear(xyr31)
        if self.ang_tanh:
            ang1 = self.ang_tanh(ang1)

        return xyr32, ang1
        
class UNet_radius_decoding(nn.Module):

    def __init__(self, num_classes, num_channels, num_params, num_extra_params, input_dims = (320, 240), num_layers_rad_enc = 4, rad_enc_nf = 4):
        super().__init__()
        self.num_layers_rad_enc = num_layers_rad_enc
        self.input_dims = input_dims
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(4, 8)
        self.dec3 = UNetDown(8, 16)
        self.dec4 = UNetDown(16, 32)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self.rad_enc_nf = rad_enc_nf
        self.rad_enc_layers = []
        # 2 x 320 x 240
        self.rad_enc_layers.append(nn.Conv2d(num_classes, self.rad_enc_nf, 3, 1, 1))
        #self.rad_enc_layers.append(nn.BatchNorm2d(self.rad_enc_nf))
        self.rad_enc_layers.append(nn.ReLU(inplace=True))
        # self.nf x 160 x 120 -> self.nf*2 x 80 x 60 -> self.nf*4 x40 x 30 -> self.nf*8 x 20 x 15
        for i in range(self.num_layers_rad_enc):
            self.rad_enc_layers.append(nn.Conv2d(self.rad_enc_nf * (2 ** i), self.rad_enc_nf * (2 ** (i+1)), 3, 1, 1))
            self.rad_enc_layers.append(nn.ReLU(inplace=True))
            self.rad_enc_layers.append(nn.Conv2d(self.rad_enc_nf * (2 ** (i+1)), self.rad_enc_nf * ( 2 ** (i+1)), 2, 2, 0))
            #self.rad_enc_layers.append(nn.BatchNorm2d(self.rad_enc_nf * (2 ** (i+1))))
            #self.rad_enc_layers.append(nn.ReLU(inplace=True))
            
        #self.rad_enc_layers.append(nn.AdaptiveAvgPool2d((4,3)))
        self.rad_enc_layers.append(nn.Flatten())
        self.rad_enc = nn.Sequential(*self.rad_enc_layers)
        self.params_layer = nn.Linear(int((self.rad_enc_nf * self.input_dims[0] * self.input_dims[1])/(2 ** self.num_layers_rad_enc)), num_params)     
        self.theta_layer = nn.Linear(int((self.rad_enc_nf * self.input_dims[0] * self.input_dims[1])/(2 ** self.num_layers_rad_enc)), num_extra_params)
        
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0.01)

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))
        
        mask = self.final(enc1)
        params_enc = self.rad_enc(mask)
        params = self.params_layer(params_enc)
        theta = self.theta_layer(params_enc)
        
        return mask, params, theta

class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride =  1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x 
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.join = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x)
        x3 = self.join(torch.cat([x1, x2], 1))
        return x3
        
class NestedUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = nn.Upsample(scale_factor=0.5, mode='nearest')
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv0_0 = ConvBlock(num_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        self.conv0_1 = ConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output

class Resize(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)  

class NestedResUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = Resize(scale_factor=0.5, mode='bicubic')
        self.up = Resize(scale_factor=2, mode='bicubic')

        self.conv0_0 = ResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UpsampleBilinear(nn.Module):
    def __init__(self, scale=2):
        super(UpsampleBilinear, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1, bias=False),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1, bias=False),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

class ResUnetPlusPlus(nn.Module):
    def __init__(self, channel, width=32):
        super(ResUnetPlusPlus, self).__init__()
        filters = [width, width*2, width*4, width*8, width*16]
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = UpsampleBilinear(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = UpsampleBilinear(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = UpsampleBilinear(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out
        

class DenseBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride =  1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(middle_channels + in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(out_channels + middle_channels + in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv_res(x)
        x1 = self.net1(x)
        x2 = self.net2(torch.cat([x, x1], 1))
        x3 = self.net3(torch.cat([x, x1, x2], 1))
        x = self.relu((x3 + res) * (1 / math.sqrt(2)))
        return x
        
class NestedDenseUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = nn.AvgPool2d(2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv0_0 = DenseBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = DenseBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = DenseBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = DenseBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = DenseBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = DenseBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = DenseBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = DenseBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = DenseBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = DenseBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = DenseBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = DenseBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = DenseBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = DenseBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = DenseBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output
        
        
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
        #return summary(self, input_shape=(2, 3, 224, 224))
        
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
                
'''
-> BackBone Resnet_GCN
'''

class Block_Resnet_GCN(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1):
        super(Block_Resnet_GCN, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), )
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv21 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.relu22 = nn.ReLU(inplace=True)


    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.bn11(x1)
        x1 = self.relu11(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12(x1)
        x1 = self.relu12(x1)

        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu21(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        x2 = self.relu22(x2)

        x = x1 + x2
        return x

class BottleneckGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_channels_gcn, stride=1):
        super(BottleneckGCN, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else: self.downsample = None
        
        self.gcn = Block_Resnet_GCN(kernel_size, in_channels, out_channels_gcn)
        self.conv1x1 = nn.Conv2d(out_channels_gcn, out_channels, 1, stride=stride, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x = self.gcn(x)
        x = self.conv1x1(x)
        x = self.bn1x1(x)

        x += identity
        return x

class ResnetGCN(nn.Module):
    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128), kernel_sizes=(5, 7)):
        super(ResnetGCN, self).__init__()
        resnet = getattr(models, backbone)(pretrained=False)

        if in_channels == 3: conv1 = resnet.conv1
        else: conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = nn.Sequential(
            BottleneckGCN(512, 1024, kernel_sizes[0], out_channels_gcn[0], stride=2),
            *[BottleneckGCN(1024, 1024, kernel_sizes[0], out_channels_gcn[0])]*5)
        self.layer4 = nn.Sequential(
            BottleneckGCN(1024, 2048, kernel_sizes[1], out_channels_gcn[1], stride=2),
            *[BottleneckGCN(1024, 1024, kernel_sizes[1], out_channels_gcn[1])]*5)
        initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = (x.size(2), x.size(3))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz

'''
-> BackBone Resnet
'''

class Resnet(nn.Module):
    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128),
                    pretrained=True, kernel_sizes=(5, 7)):
        super(Resnet, self).__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained)

        if in_channels == 3: conv1 = resnet.conv1
        else: conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if not pretrained: initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = (x.size(2), x.size(3))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz

'''
-> Global Convolutionnal Network
'''

class GCN_Block(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(GCN_Block, self).__init__()

        assert kernel_size % 2 == 1, 'Kernel size must be odd'
        self.conv11 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.conv12 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size//2))

        self.conv21 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.conv22 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        initialize_weights(self)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x2 = self.conv21(x)
        x2 = self.conv22(x2)

        x = x1 + x2
        return x

class BR_Block(nn.Module):
    def __init__(self, num_channels):
        super(BR_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        initialize_weights(self)

    def forward(self, x):
        identity = x
        # x = self.conv1(self.relu1(self.bn1(x)))
        # x = self.conv2(self.relu2(self.bn2(x)))
        x = self.conv2(self.relu2(self.conv1(x)))
        x += identity
        return x

class GCN(BaseModel):
    def __init__(self, num_classes, in_channels=1, pretrained=True, use_resnet_gcn=False, backbone='resnet50', use_deconv=False,
                    num_filters=11, freeze_bn=False, freeze_backbone=False, **_):
        super(GCN, self).__init__()
        self.use_deconv = use_deconv
        if use_resnet_gcn:
            self.backbone = ResnetGCN(in_channels, backbone=backbone)
        else:
            self.backbone = Resnet(in_channels, pretrained=pretrained, backbone=backbone)

        if (backbone == 'resnet34' or backbone == 'resnet18'): resnet_channels = [64, 128, 256, 512]
        else: resnet_channels = [256, 512, 1024, 2048]
        
        self.gcn1 = GCN_Block(num_filters, resnet_channels[0], num_classes)
        self.br1 = BR_Block(num_classes)
        self.gcn2 = GCN_Block(num_filters, resnet_channels[1], num_classes)
        self.br2 = BR_Block(num_classes)
        self.gcn3 = GCN_Block(num_filters, resnet_channels[2], num_classes)
        self.br3 = BR_Block(num_classes)
        self.gcn4 = GCN_Block(num_filters, resnet_channels[3], num_classes)
        self.br4 = BR_Block(num_classes)

        self.br5 = BR_Block(num_classes)
        self.br6 = BR_Block(num_classes)
        self.br7 = BR_Block(num_classes)
        self.br8 = BR_Block(num_classes)
        self.br9 = BR_Block(num_classes)

        if self.use_deconv:
            self.decon1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
            self.decon2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
            self.decon3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
            self.decon4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
            self.decon5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.backbone], False)

    def forward(self, x):
        x1, x2, x3, x4, conv1_sz = self.backbone(x)

        x1 = self.br1(self.gcn1(x1))
        x2 = self.br2(self.gcn2(x2))
        x3 = self.br3(self.gcn3(x3))
        x4 = self.br4(self.gcn4(x4))

        if self.use_deconv:
            # Padding because when using deconv, if the size is odd, we'll have an alignment error
            x4 = self.decon4(x4)
            if x4.size() != x3.size(): x4 = self._pad(x4, x3)
            x3 = self.decon3(self.br5(x3 + x4))
            if x3.size() != x2.size(): x3 = self._pad(x3, x2)
            x2 = self.decon2(self.br6(x2 + x3))
            x1 = self.decon1(self.br7(x1 + x2))

            x = self.br9(self.decon5(self.br8(x1)))
        else:
            x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
            x3 = F.interpolate(self.br5(x3 + x4), size=x2.size()[2:], mode='bilinear', align_corners=True)
            x2 = F.interpolate(self.br6(x2 + x3), size=x1.size()[2:], mode='bilinear', align_corners=True)
            x1 = F.interpolate(self.br7(x1 + x2), size=conv1_sz, mode='bilinear', align_corners=True)

            x = self.br9(F.interpolate(self.br8(x1), size=x.size()[2:], mode='bilinear', align_corners=True))
        return self.final_conv(x)

    def _pad(self, x_topad, x):
        pad = (x.size(3) - x_topad.size(3), 0, x.size(2) - x_topad.size(2), 0)
        x_topad = F.pad(x_topad, pad, "constant", 0)
        return x_topad

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return [p for n, p in self.named_parameters() if 'backbone' not in n]

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
            
''' 
-> ResNet BackBone
'''

class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16: s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8: s3, s4, d3, d4 = (1, 1, 2, 4)
        
        if output_stride == 8: 
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features


"""
Created on Fri Sep 13 19:04:23 2019
@author: shirhe-lyh
Implementation of Xception model.
Xception: Deep Learning with Depthwise Separable Convolutions, F. Chollect,
    arxiv:1610.02357 (https://arxiv.org/abs/1610.02357).
Official tensorflow implementation:
    https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py
"""

import collections
import os
import torch


_DEFAULT_MULTI_GRID = [1, 1, 1]
# The cap for torch.clamp
_CLIP_CAP = 6
_BATCH_NORM_PARAMS = {
    'eps': 0.001,
    'momentum': 0.9997,
    'affine': True,
}


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing an Xception block.
    
    Its parts are:
        scope: The scope of the block.
        unit_fn: The Xception unit function which takes as input a tensor and
            returns another tensor with the output of the Xception unit.
        args: A list of length equal to the number of units in the block. The
            list contains one dictionary for each unit in the block to serve 
            as argument to unit_fn.
    """
    
    
def fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.
    
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        kernel_size: The kernel to be used in the conv2d or max_pool2d 
            operation. Should be a positive integer.
        rate: An integer, rate for atrous convolution.
        
    Returns:
        padded_inputs: A tensor of size [batch, height_out, width_out, 
            channels] with the input, either intact (if kernel_size == 1) or 
            padded (if kernel_size > 1).
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = torch.nn.functional.pad(
        inputs, pad=(pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class Conv2dSame(torch.nn.Module):
    """Strided 2-D convolution with 'SAME' padding."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, rate=1):
        """Constructor.
        
        If stride > 1 and use_explicit_padding is True, then we do explicit
        zero-padding, followed by conv2d with 'VALID' padding.
        
        Args:
            in_channels: An integer, the number of input filters.
            out_channels: An integer, the number of output filters.
            kernel_size: An integer with the kernel_size of the filters.
            stride: An integer, the output stride.
            rate: An integer, rate for atrous convolution.
        """
        super(Conv2dSame, self).__init__()
        self._kernel_size = kernel_size
        self._rate = rate
        self._without_padding = stride == 1
        if self._without_padding:
            # Here, we assume that floor(padding) = padding
            padding = (kernel_size - 1) * rate // 2
            self._conv = torch.nn.Conv2d(in_channels, 
                                         out_channels,
                                         kernel_size=kernel_size,
                                         stride=1,
                                         dilation=rate,
                                         padding=padding,
                                         bias=False)
        else:
            self._conv = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         dilation=rate,
                                         bias=False)
        self._batch_norm = torch.nn.BatchNorm2d(out_channels, 
                                                **_BATCH_NORM_PARAMS)
        self._relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: A 4-D tensor with shape [batch, height_in, width_in, channels].
        
        Returns:
            A 4-D tensor of size [batch, height_out, width_out, channels] with 
                the convolution output.
        """
        if not self._without_padding:
            x = fixed_padding(x, self._kernel_size, self._rate)
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class SeparableConv2dSame(torch.nn.Module):
    """Strided 2-D separable convolution with 'SAME' padding."""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 depth_multiplier, stride, rate, use_explicit_padding=True, 
                 activation_fn=None, regularize_depthwise=False, **kwargs):
        """Constructor.
        
        If stride > 1 and use_explicit_padding is True, then we do explicit
        zero-padding, followed by conv2d with 'VALID' padding.
        
        Args:
            in_channels: An integer, the number of input filters.
            out_channels: An integer, the number of output filters.
            kernel_size: An integer with the kernel_size of the filters.
            depth_multiplier: The number of depthwise convolution output
                channels for each input channel. The total number of depthwise
                convolution output channels will be equal to `num_filters_in *
                depth_multiplier`.
            stride: An integer, the output stride.
            rate: An integer, rate for atrous convolution.
            use_explicit_padding: If True, use explicit padding to make the
                model fully compatible with the open source version, otherwise
                use the nattive Pytorch 'SAME' padding.
            activation_fn: Activation function.
            regularize_depthwise: Whether or not apply L2-norm regularization
                on the depthwise convolution weights.
            **kwargs: Additional keyword arguments to pass to torch.nn.Conv2d.
        """
        super(SeparableConv2dSame, self).__init__()
        self._kernel_size = kernel_size
        self._rate = rate
        self._without_padding = stride == 1 or not use_explicit_padding
        
        out_channels_depthwise = in_channels * depth_multiplier
        if self._without_padding:
            # Separable convolution for padding 'SAME'
            # Here, we assume that floor(padding) = padding
            padding = (kernel_size - 1) * rate // 2
            self._conv_depthwise = torch.nn.Conv2d(in_channels, 
                                                   out_channels_depthwise,
                                                   kernel_size=kernel_size, 
                                                   stride=stride, 
                                                   dilation=rate,
                                                   groups=in_channels,
                                                   padding=padding,
                                                   bias=False,
                                                   **kwargs)
        else:
            # Separable convolution for padding 'VALID'
            self._conv_depthwise = torch.nn.Conv2d(in_channels,
                                                   out_channels_depthwise,
                                                   kernel_size=kernel_size, 
                                                   stride=stride,
                                                   dilation=rate,
                                                   groups=in_channels,
                                                   bias=False,
                                                   **kwargs)
        self._batch_norm_depthwise = torch.nn.BatchNorm2d(
            out_channels_depthwise, **_BATCH_NORM_PARAMS)
        self._conv_pointwise = torch.nn.Conv2d(out_channels_depthwise,
                                               out_channels,
                                               kernel_size=1, 
                                               stride=1,
                                               bias=False,
                                               **kwargs)
        self._batch_norm_pointwise = torch.nn.BatchNorm2d(
            out_channels, **_BATCH_NORM_PARAMS)
        self._activation_fn = activation_fn
    
    def forward(self, x):
        """
        Args:
            x: A 4-D tensor with shape [batch, height_in, width_in, channels].
        
        Returns:
            A 4-D tensor of size [batch, height_out, width_out, channels] with 
                the convolution output.
        """
        if not self._without_padding:
            x = fixed_padding(x, self._kernel_size, self._rate)
        x = self._conv_depthwise(x)
        x = self._batch_norm_depthwise(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        x = self._conv_pointwise(x)
        x = self._batch_norm_pointwise(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x
    

class XceptionModule(torch.nn.Module):
    """An Xception module.
    
    The output of one Xception module is equal to the sum of `residual` and
    `shortcut`, where `residual` is the feature computed by three seperable
    convolution. The `shortcut` is the feature computed by 1x1 convolution
    with or without striding. In some cases, the `shortcut` path could be a
    simple identity function or none (i.e, no shortcut).
    """
    
    def __init__(self, in_channels, depth_list, skip_connection_type, stride, 
                 unit_rate_list, rate=1, activation_fn_in_separable_conv=False, 
                 regularize_depthwise=False, use_bounded_activation=False,
                 use_explicit_padding=True):
        """Constructor.
        
        Args:
            in_channels: An integer, the number of input filters.
            depth_list: A list of three integers specifying the depth values
                of one Xception module.
            skip_connection_type: Skip connection type for the residual path.
                Only supports 'conv', 'sum', or 'none'.
            stride: The block unit's stride. Detemines the amount of 
                downsampling of the units output compared to its input.
            unit_rate_list: A list of three integers, determining the unit 
                rate for each separable convolution in the Xception module.
            rate: An integer, rate for atrous convolution.
            activation_fn_in_separable_conv: Includes activation function in
                the seperable convolution or not.
            regularize_depthwise: Whether or not apply L2-norm regularization
                on the depthwise convolution weights.
            use_bounded_activation: Whether or not to use bounded activations.
                Bounded activations better lend themselves to quantized 
                inference.
            use_explicit_padding: If True, use explicit padding to make the
                model fully compatible with the open source version, otherwise
                use the nattive Pytorch 'SAME' padding.
                
        Raises:
            ValueError: If depth_list and unit_rate_list do not contain three
                integers, or if stride != 1 for the third seperable convolution
                operation in the residual path, or unsupported skip connection
                type.
        """
        super(XceptionModule, self).__init__()
        
        if len(depth_list) != 3:
            raise ValueError('Expect three elements in `depth_list`.')
        if len(unit_rate_list) != 3:
            raise ValueError('Expect three elements in `unit_rate_list`.')
        if skip_connection_type not in ['conv', 'sum', 'none']:
            raise ValueError('Unsupported skip connection type.')
            
        # Activation function
        self._input_activation_fn = None
        if activation_fn_in_separable_conv:
            activation_fn = (torch.nn.ReLU6(inplace=False) if 
                             use_bounded_activation else 
                             torch.nn.ReLU(inplace=False))
        else:
            if use_bounded_activation:
                # When use_bounded_activation is True, we clip the feature
                # values and apply relu6 for activation.
                activation_fn = lambda x: torch.clamp(x, -_CLIP_CAP, _CLIP_CAP)
                self._input_activation_fn = torch.nn.ReLU6(inplace=False)
            else:
                # Original network design.
                activation_fn = None
                self._input_activation_fn = torch.nn.ReLU(inplace=False)
        self._use_bounded_activation = use_bounded_activation
        self._output_activation_fn = None
        if use_bounded_activation:
            self._output_activation_fn = torch.nn.ReLU6(inplace=True)
         
        # Separable conv block.
        layers = []
        in_channels_ = in_channels
        for i in range(3):
            if self._input_activation_fn is not None:
                layers += [self._input_activation_fn]
            layers += [
                SeparableConv2dSame(in_channels_,
                                    depth_list[i],
                                    kernel_size=3,
                                    depth_multiplier=1,
                                    regularize_depthwise=regularize_depthwise,
                                    rate=rate*unit_rate_list[i],
                                    stride=stride if i==2 else 1,
                                    activation_fn=activation_fn,
                                    use_explicit_padding=use_explicit_padding)]
            in_channels_ = depth_list[i]
        self._separable_conv_block = torch.nn.Sequential(*layers)
        
        # Skip connection
        self._skip_connection_type = skip_connection_type
        if skip_connection_type == 'conv':
            self._conv_skip_connection = torch.nn.Conv2d(in_channels,
                                                         depth_list[-1],
                                                         kernel_size=1,
                                                         stride=stride)
            self._batch_norm_shortcut = torch.nn.BatchNorm2d(
                depth_list[-1], **_BATCH_NORM_PARAMS)
            
    def forward(self, x):
        """
        Args:
            x: A 4-D tensor with shape [batch, height, width, channels].
        
        Returns:
            The Xception module's output.
        """
        residual = self._separable_conv_block(x)
        if self._skip_connection_type == 'conv':
            shortcut = self._conv_skip_connection(x)
            shortcut = self._batch_norm_shortcut(shortcut)
            if self._use_bounded_activation:
                residual = torch.clamp(residual, -_CLIP_CAP, _CLIP_CAP)
                shortcut = torch.clamp(shortcut, -_CLIP_CAP, _CLIP_CAP)
            outputs = residual + shortcut
            if self._use_bounded_activation:
                outputs = self._output_activation_fn(outputs)
        elif self._skip_connection_type == 'sum':
            if self._use_bounded_activation:
                residual = torch.clamp(residual, -_CLIP_CAP, _CLIP_CAP)
                x = torch.clamp(x, -_CLIP_CAP, _CLIP_CAP)
            outputs = residual + x
            if self._use_bounded_activation:
                outputs = self._output_activation_fn(outputs)
        else:
            outputs = residual
        return outputs
    
    
class StackBlocksDense(torch.nn.Module):
    """Stacks Xception blocks and controls output feature density.
    
    This class allows the user to explicitly control the output stride, which
    is the ratio of the input to output spatial resolution. This is useful for
    dense prediction tasks such as semantic segmentation or object detection.
    
    Control of the output feature density is implemented by atrous convolution.
    """
    
    def __init__(self, blocks, output_stride=None):
        """Constructor.
        
        Args:
            blocks: A list of length equal to the number of Xception blocks.
                Each element is an Xception Block object describing the units
                in the block.
            output_stride: If None, then the output will be computed at the
                nominal network stride. If output_stride is not None, it 
                specifies the requested ratio of input to output spatial
                resolution, which needs to be equal to the product of unit
                strides from the start up to some level of Xception. For
                example, if the Xception employs units with strides 1, 2, 1,
                3, 4, 1, then valid values for the output_stride are 1, 2, 6,
                24 or None (which is equivalent to output_stride=24).
                
        Raises:
            ValueError: If the target output_stride is not valid.
        """
        super(StackBlocksDense, self).__init__()
        
        # The current_stride variable keeps track of the effective stride of
        # the activations. This allows us to invoke atrous convolution whenever
        # applying the next residual unit would result in the activations 
        # having stride larger than the target output_stride.
        current_stride = 1
        
        # The atrous convolution rate parameter.
        rate = 1
        
        layers = []
        for block in blocks:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be '
                                     'reached.')
                # If we have reached the target output_stride, then we need to
                # employ atrous convolution with stride=1 and multiply the
                # atrous rate by the current unit's stride for use subsequent
                # layers.
                if output_stride is not None and current_stride == output_stride:
                    layers += [block.unit_fn(rate=rate, **dict(unit, stride=1))]
                    rate *= unit.get('stride', 1)
                else:
                    layers += [block.unit_fn(rate=1, **unit)]
                    current_stride *= unit.get('stride', 1)
        
        if output_stride is not None and current_stride != output_stride:
            raise ValueError('The target ouput_stride cannot be reached.')
            
        self._blocks = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: A tensor of shape [batch, height, widht, channels].
            
        Returns:
            Output tensor with stride equal to the specified output_stride.
        """
        x = self._blocks(x)
        return x
    
    
class Xception(torch.nn.Module):
    """Generator for Xception models.
    
    This class generates a family of Xception models. See the xception_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce Xception of various depths.
    """
    
    def __init__(self, blocks, num_classes=None, global_pool=True, 
                 keep_prob=0.5, output_stride=None, scope=None):
        """Constructor.
        
        Args:
            blocks: A list of length equal to the number of Xception blocks.
                Each element is an Xception Block object describing the units
                in the block.
            num_classes: Number of predicted classes for classification tasks.
                If 0 or None, we return the features before the logit layer.
            global_pool: If True, we perform global average pooling before
                computing logits. Set to True for image classification, False
                for dense prediction.
            keep_prob: Keep probability used in the pre-logits dropout layer.
            output_stride: If None, the the output will be computed at the 
                nominal network stride. If output_stride is not None, it
                specifies the requested ratio of input to output spatial
                resolution.
            scope: Optional variable_scope.
                
        Raises:
            ValueError: If the target output_stride is not valid.
        """
        super(Xception, self).__init__()
        
        self._scope = scope
        
        layers = []
        if output_stride is not None:
            if output_stride % 2 != 0:
                raise ValueError('The output_stride must be a multiple of 2.')
            output_stride /= 2
        # Root block function operated on inputs
        layers += [Conv2dSame(3, 32, 3, stride=2),
                   Conv2dSame(32, 64, 3, stride=1)]
        
        # Extract features for entry_flow, middle_flow, and exit_flow
        layers += [StackBlocksDense(blocks, output_stride)]
        
        if global_pool:
            # Global average pooling
            layers += [torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))]
        if num_classes:
            layers += [torch.nn.Dropout2d(p=keep_prob, inplace=True),
                       torch.nn.Conv2d(blocks[-1].args[-1]['depth_list'][-1], 
                                       num_classes, 1)]
        self._layers = torch.nn.Sequential(*layers)
                       
    def forward(self, x):
        """
        Args:
            x: A tensor of shape [batch, height, widht, channels].
            
        Returns:
            Output tensor with stride equal to the specified output_stride.
        """
        output = self._layers(x)
        
        
        x1 = self._layers[0](x)
        x2 = self._layers[1](x1)
        low_level_features = self._layers[2]._blocks[0](x2)
        
        #low_level_features = self._layers[2]._blocks[0](x1)
        
        #print('x1',x1.size())
        #print('x2',x2.size())        
        #print('low_level_features',low_level_features.size())
        '''
        if output_stride = None:
            output.size() torch.Size([2, 2048, 7, 7])
            low_level_features.size() torch.Size([2, 128, 56, 56])
        elif output_stride = 16:
            output.size() torch.Size([2, 2048, 14, 14])
            low_level_features.size() torch.Size([2, 128, 56, 56])
        
        
        '''
        
        
        return output,low_level_features
    
    @property
    def scope(self):
        return self._scope
    
    
def xception_block(scope,
                   in_channels,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   unit_rate_list=None):
    """Helper function for creating a Xception block.
    
    Args:
        scope: The scope of the block.
        in_channels: The number of input filters.
        depth_list: The depth of the bottleneck layer for each unit.
        skip_connection_type: Skip connection type for the residual path. Only
            supports 'conv', 'sum', or 'none'.
        activation_fn_in_separable_conv: Includes activation function in the
            separable convolution or not.
        regularize_depthwise: Whether or not apply L2-norm regularization on 
            the depthwise convolution weights.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last
            unit. All other units have stride=1.
        unit_rate_list: A list of three integers, determining the unit rate in
            the corresponding xception block.
            
    Returns:
        An xception block.
    """
    if unit_rate_list is None:
        unit_rate_list = _DEFAULT_MULTI_GRID
    return Block(scope, XceptionModule, [{
            'in_channels': in_channels,
            'depth_list': depth_list,
            'skip_connection_type': skip_connection_type,
            'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
            'regularize_depthwise': regularize_depthwise,
            'stride': stride,
            'unit_rate_list': unit_rate_list,
            }] * num_units)
    
    

def Xception41(num_classes=None,
               global_pool=True,
               keep_prob=0.5,
               output_stride=None,
               regularize_depthwise=False,
               multi_grid=None,
               scope='xception_41'):
    """Xception-41 model."""
    blocks = [
        xception_block('entry_flow/block1',
                       in_channels=64,
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       in_channels=128,
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block3',
                       in_channels=256,
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('middle_flow/block1',
                       in_channels=728,
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=8,
                       stride=1),
        xception_block('exit_flow/block1',
                       in_channels=728,
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       in_channels=1024,
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=multi_grid),
    ]
    return Xception(blocks=blocks, num_classes=num_classes,
                    global_pool=global_pool, keep_prob=keep_prob,
                    output_stride=output_stride, scope=scope)
    
    
def xception_41(num_classes=None,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                scope='xception_41',
                pretrained=True,
                checkpoint_path='./pretrained/xception_41.pth'):
    """Xception-41 model."""
    xception = Xception41(num_classes=num_classes, global_pool=global_pool, 
                          keep_prob=keep_prob, output_stride=output_stride,
                          scope=scope)
    if pretrained:
        _load_state_dict(xception, num_classes, checkpoint_path)
    return xception


def Xception65(num_classes=None,
               global_pool=True,
               keep_prob=0.5,
               output_stride=None,
               regularize_depthwise=False,
               multi_grid=None,
               scope='xception_65'):
    """Xception-65 model."""
    blocks = [
        xception_block('entry_flow/block1',
                       in_channels=64,
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       in_channels=128,
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block3',
                       in_channels=256,
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('middle_flow/block1',
                       in_channels=728,
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=16,
                       stride=1),
        xception_block('exit_flow/block1',
                       in_channels=728,
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       in_channels=1024,
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=multi_grid),
    ]
    return Xception(blocks=blocks, num_classes=num_classes,
                    global_pool=global_pool, keep_prob=keep_prob,
                    output_stride=output_stride, scope=scope)


def xception_65(num_classes=None,
                global_pool=False,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                scope='xception_65',
                pretrained=True,
                checkpoint_path='./pretrained/xception_65.pth'):
    """Xception-65 model."""
    xception = Xception65(num_classes=num_classes, global_pool=global_pool, 
                          keep_prob=keep_prob, output_stride=output_stride,
                          scope=scope)
    if pretrained:
        _load_state_dict(xception, num_classes, checkpoint_path='./pretrained/xception_65.pth')
    return xception


def Xception71(num_classes=None,
               global_pool=True,
               keep_prob=0.5,
               output_stride=None,
               regularize_depthwise=False,
               multi_grid=None,
               scope='xception_71'):
    """Xception-71 model."""
    blocks = [
        xception_block('entry_flow/block1',
                       in_channels=64,
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       in_channels=128,
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1),
        xception_block('entry_flow/block3',
                       in_channels=256,
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block4',
                       in_channels=256,
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1),
        xception_block('entry_flow/block5',
                       in_channels=728,
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('middle_flow/block1',
                       in_channels=728,
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=16,
                       stride=1),
        xception_block('exit_flow/block1',
                       in_channels=728,
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       in_channels=1024,
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=multi_grid),
    ]
    return Xception(blocks=blocks, num_classes=num_classes,
                    global_pool=global_pool, keep_prob=keep_prob,
                    output_stride=output_stride, scope=scope)


def xception_71(num_classes=None,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                scope='xception_71',
                pretrained=True,
                checkpoint_path='./pretrained/xception_71.pth'):
    """Xception-71 model."""
    xception = Xception71(num_classes=num_classes, global_pool=global_pool, 
                          keep_prob=keep_prob, output_stride=output_stride,
                          scope=scope)
    if pretrained:
        _load_state_dict(xception, num_classes, checkpoint_path)
    return xception


def _load_state_dict(model, num_classes, checkpoint_path):
    """Load pretrained weights."""
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        if num_classes is None or num_classes != 1001:
            state_dict.pop('_layers.5.weight')
            state_dict.pop('_layers.5.bias')
        model.load_state_dict(state_dict, strict=False)
        print('Load pretrained weights successfully.')
    else:
        raise ValueError('`checkpoint_path` does not exist.')







''' 
-> The Atrous Spatial Pyramid Pooling
'''

def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True))

class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]
        
        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x

''' 
-> Decoder
'''

class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x

'''
-> Deeplab V3 +
'''

class DeepLab(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet', pretrained=True, 
                output_stride=16, freeze_bn=False,freeze_backbone=False, **_):
                
        super(DeepLab, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256
        else:
            self.backbone = xception_65(output_stride=output_stride, pretrained=pretrained,global_pool=False,checkpoint_path='./pretrained/xception_65.pth')
            low_level_channels = 128

        self.ASSP = ASSP(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)

        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    # Two functions to yield the parameters of the backbone
    # & Decoder / ASSP to use differentiable learning rates
    # FIXME: in xception, we use the parameters from xception and not aligned xception
    # better to have higher lr for this backbone

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
            
class SegNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegNet, self).__init__()
        vgg_bn = models.vgg16_bn(pretrained= pretrained)
        encoder = list(vgg_bn.features.children())

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:],
                nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
                                    self.stage4_decoder, self.stage5_decoder)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.stage1_encoder, self.stage2_encoder, self.stage3_encoder, self.stage4_encoder, self.stage5_encoder], False)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()



class DecoderBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(DecoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels//4)
        self.conv2 = nn.ConvTranspose2d(inchannels//4, inchannels//4, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels//4)
        self.conv3 = nn.Conv2d(inchannels//4, inchannels//2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels//2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.ConvTranspose2d(inchannels, inchannels//2, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(inchannels//2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class LastBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(LastBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels//4)
        self.conv2 = nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels//4)
        self.conv3 = nn.Conv2d(inchannels//4, inchannels//4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels//4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False),
                nn.BatchNorm2d(inchannels//4))
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SegResNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        encoder = list(resnet50.children())
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        encoder[3].return_indices = True

        # Encoder
        self.first_conv = nn.Sequential(*encoder[:4])
        resnet50_blocks = list(resnet50.children())[4:-2]
        self.encoder = nn.Sequential(*resnet50_blocks)

        # Decoder
        resnet50_untrained = models.resnet50(pretrained=False)
        resnet50_blocks = list(resnet50_untrained.children())[4:-2][::-1]
        decoder = []
        channels = (2048, 1024, 512)
        for i, block in enumerate(resnet50_blocks[:-1]):
            new_block = list(block.children())[::-1][:-1]
            decoder.append(nn.Sequential(*new_block, DecoderBottleneck(channels[i])))
        new_block = list(resnet50_blocks[-1].children())[::-1][:-1]
        decoder.append(nn.Sequential(*new_block, LastBottleneck(256)))

        self.decoder = nn.Sequential(*decoder)
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.first_conv, self.encoder], False)

    def forward(self, x):
        inputsize = x.size()

        # Encoder
        x, indices = self.first_conv(x)
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)
        h_diff = ceil((x.size()[2] - indices.size()[2]) / 2)
        w_diff = ceil((x.size()[3] - indices.size()[3]) / 2)
        if indices.size()[2] % 2 == 1:
            x = x[:, :, h_diff:x.size()[2]-(h_diff-1), w_diff: x.size()[3]-(w_diff-1)]
        else:
            x = x[:, :, h_diff:x.size()[2]-h_diff, w_diff: x.size()[3]-w_diff]

        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        x = self.last_conv(x)
        
        if inputsize != x.size():
            h_diff = (x.size()[2] - inputsize[2]) // 2
            w_diff = (x.size()[3] - inputsize[3]) // 2
            x = x[:, :, h_diff:x.size()[2]-h_diff, w_diff: x.size()[3]-w_diff]
            if h_diff % 2 != 0: x = x[:, :, :-1, :]
            if w_diff % 2 != 0: x = x[:, :, :, :-1]

        return x

    def get_backbone_params(self):
        return chain(self.first_conv.parameters(), self.encoder.parameters())

    def get_decoder_params(self):
        return chain(self.decoder.parameters(), self.last_conv.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
            
class AtrousResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Conv2d(in_channels, int(out_channels/2)*2, 1, stride=1, bias=False)
        self.netinp1 = nn.Sequential(
            nn.Conv2d(in_channels, int(middle_channels/2), 3, padding=1, bias=False),
            nn.BatchNorm2d(int(middle_channels/2)),
            nn.ReLU(inplace=True)
        )
        self.netinp2 = nn.Sequential(
            nn.Conv2d(in_channels, int(middle_channels/2), 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(int(middle_channels/2)),
            nn.ReLU(inplace=True)
        )
        self.netproc1 = nn.Conv2d(int(middle_channels/2)*2, int(out_channels/2), 3, padding=1, bias=False)
        self.netproc2 = nn.Conv2d(int(middle_channels/2)*2, int(out_channels/2), 3, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(int(out_channels/2)*2)
        self.relu = nn.ReLU(inplace=True)
            

    def forward(self, x):
        res = self.conv_res(x)
        x1 = self.netinp1(x)
        x2 = self.netinp2(x)
        x3 = torch.cat([x1, x2], 1)
        x4 = self.netproc1(x3)
        x5 = self.netproc2(x3)
        x6 = torch.cat([x4, x5], 1)
        x7 = x6 + res
        x8 = self.bn(x7)
        x9 = self.relu(x8)
        return x9

class NestedAtrousResUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv0_0 = AtrousResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = AtrousResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = AtrousResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = AtrousResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = AtrousResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = AtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = AtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = AtrousResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = AtrousResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = AtrousResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = AtrousResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = AtrousResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = AtrousResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = AtrousResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = AtrousResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output
        
class SharedAtrousConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, padding='same'):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(int(out_channels/2), in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        nn.init.xavier_uniform_(self.weights)
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(int(out_channels/2)))
            self.bias2 = nn.Parameter(torch.zeros(int(out_channels/2)))
        else:
            self.bias1 = None
            self.bias2 = None
    def forward(self, x):
        b, c, h, w = x.shape
        
        if self.padding == 'same':
            pad_val1_h = int(((h - 1)*self.stride - h + self.kernel_size) / 2)
            pad_val1_w = int(((w - 1)*self.stride - w + self.kernel_size) / 2)
            pad_val2_h = int(((h - 1)*self.stride - h + 2 * self.kernel_size - 1) / 2)
            pad_val2_w = int(((w - 1)*self.stride - w + 2 * self.kernel_size - 1) / 2)
        elif self.padding == 'valid':
            pad_val1_h = 0
            pad_val1_w = 0
            pad_val2_h = 0
            pad_val2_w = 0
        
        x1 = nn.functional.conv2d(x, self.weights, stride=self.stride, padding=(pad_val1_h, pad_val1_w), bias=self.bias1)
        x2 = nn.functional.conv2d(x, self.weights, stride=self.stride, padding=(pad_val2_h, pad_val2_w), dilation=2, bias=self.bias2)
        x3 = torch.cat([x1, x2], 1)
        return x3

class SharedAtrousResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.net = nn.Sequential(
            SharedAtrousConv2d(in_channels, middle_channels, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            SharedAtrousConv2d(middle_channels, out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.relu(x)
        return x
    
class Resize(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor, antialias=self.antialias)
        
class NestedSharedAtrousResUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = Resize(scale_factor=0.5, mode='bilinear')
        self.up = Resize(scale_factor=2, mode='bilinear')

        self.conv0_0 = SharedAtrousResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SharedAtrousResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SharedAtrousResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SharedAtrousResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SharedAtrousResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SharedAtrousResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output


class NestedSharedAtrousResUNetSimple(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = Resize(scale_factor=0.5, mode='nearest')
        self.up = Resize(scale_factor=2, mode='nearest')

        self.conv0_0 = SharedAtrousResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SharedAtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SharedAtrousResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SharedAtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*5, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4], 1))
        return output


class SharedAtrousResNetParam(nn.Module):
    def __init__(self, num_params, num_channels, width=64, resolution=(240, 320), blocks_per_resolution=1):
        super().__init__()
        self.resolution = resolution
        self.blocks_per_resolution = blocks_per_resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]
        self.pool = Resize(scale_factor=0.5, mode='bilinear')
        self.conv0 = SharedAtrousResBlock(num_channels, nb_filter[0], nb_filter[0])
        conv0arr = []
        for i in range(blocks_per_resolution):
            conv0arr.append(SharedAtrousResBlock(nb_filter[0], nb_filter[0], nb_filter[0]))
        self.conv0seq = nn.Sequential(*conv0arr)
        self.conv1 = SharedAtrousResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        conv1arr = []
        for i in range(blocks_per_resolution):
            conv1arr.append(SharedAtrousResBlock(nb_filter[1], nb_filter[1], nb_filter[1]))
        self.conv1seq = nn.Sequential(*conv1arr)
        self.conv2 = SharedAtrousResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        conv2arr = []
        for i in range(blocks_per_resolution):
            conv2arr.append(SharedAtrousResBlock(nb_filter[2], nb_filter[2], nb_filter[2]))
        self.conv2seq = nn.Sequential(*conv2arr)
        self.conv3 = SharedAtrousResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        conv3arr = []
        for i in range(blocks_per_resolution):
            conv3arr.append(SharedAtrousResBlock(nb_filter[3], nb_filter[3], nb_filter[3]))
        self.conv3seq = nn.Sequential(*conv3arr)
        self.conv4 = SharedAtrousResBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        conv4arr = []
        for i in range(blocks_per_resolution):
            conv4arr.append(SharedAtrousResBlock(nb_filter[4], nb_filter[4], nb_filter[4]))
        self.conv4seq = nn.Sequential(*conv4arr)
        self.final = nn.Sequential(
            nn.Conv2d(nb_filter[4], width, kernel_size=1),
            nn.Flatten(),
            nn.Linear(int((self.resolution[0] * self.resolution[1])/256)*width, 6)
        )
        
    def forward(self, input):
        
        x0_0 = self.conv0(input) #320x240
        x0_1 = self.conv0seq(x0_0)
        x1_0 = self.conv1(self.pool(x0_1)) #160x120
        x1_1 = self.conv1seq(x1_0)
        x2_0 = self.conv2(self.pool(x1_1)) #80x60
        x2_1 = self.conv2seq(x2_0)
        x3_0 = self.conv3(self.pool(x2_1)) #40x30
        x3_1 = self.conv3seq(x3_0)
        x4_0 = self.conv4(self.pool(x3_1)) #20x15
        x4_1 = self.conv4seq(x4_0)
        
        output = self.final(x4_1)
        
        return output

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.GELU()
        self.conv1 = SharedAtrousConv2d(in_planes, out_planes, bias=False)
        #self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               #padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.GELU()
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = SharedAtrousConv2d(inter_planes, out_planes, bias=False)
        #self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               #padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.GELU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class SharedAtrousDenseNetParam(nn.Module):
    def __init__(self, in_channels, depth=100, num_params=6, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(SharedAtrousDenseNetParam, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = SharedAtrousConv2d(in_channels, in_planes, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.GELU()
        self.final = nn.Conv2d(in_planes, num_params, kernel_size=1, stride=1, padding='same')
        
        self.fc_list = []
        for i in range(num_params):
            self.fc_list.append(nn.Linear(4800, 1))
        
        self.fc_list = nn.ModuleList(self.fc_list)
        self.num_params = num_params
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        #print(out.shape)
        params = []
        for i in range(self.num_params):
            params.append(self.fc_list[i](out[:, i, :, :].reshape(out.shape[0], out.shape[2]*out.shape[3])))
        param_vec = torch.cat(params, 1)
        return param_vec

class SharedAtrousResBlockIN(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(SharedAtrousResBlockIN,self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding='same', bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.downsample = downsample
        if downsample:
            self.net_downsample = nn.Sequential(
                SharedAtrousConv2d(in_channels, middle_channels, kernel_size=4, stride=2, bias=False),
                nn.InstanceNorm2d(middle_channels),
                nn.GELU(),
                SharedAtrousConv2d(middle_channels, out_channels, kernel_size=3, stride=1, bias=False),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.net = nn.Sequential(
                SharedAtrousConv2d(in_channels, middle_channels, kernel_size=3, stride=1, bias=False),
                nn.InstanceNorm2d(middle_channels),
                nn.GELU(),
                SharedAtrousConv2d(middle_channels, out_channels, kernel_size=3, stride=1, bias=False),
                nn.InstanceNorm2d(out_channels)
            )
        self.act = nn.GELU()
    def forward(self, x):
        res = self.conv_res(x)
        if self.downsample:
            x = self.net_downsample(x)
            res = nn.functional.interpolate(res, scale_factor=0.5)
        else:
            x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.act(x)
        return x

class ResBlockIN(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(ResBlockIN,self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding='same', bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.downsample = downsample
        if downsample:
            self.net_downsample = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(middle_channels),
                nn.GELU(),
                nn.Conv2d(middle_channels, out_channels, 3, stride=1, padding='same', bias=False),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, 3, stride=1, padding='same', bias=False),
                nn.InstanceNorm2d(middle_channels),
                nn.GELU(),
                nn.Conv2d(middle_channels, out_channels, 3, stride=1, padding='same', bias=False),
                nn.InstanceNorm2d(out_channels)
            )
        self.act = nn.GELU()
    def forward(self, x):
        res = self.conv_res(x)
        if self.downsample:
            x = self.net_downsample(x)
            res = nn.functional.interpolate(res, scale_factor=0.5)
        else:
            x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.act(x)
        return x     

class AttBlockIN(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttBlockIN,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        
        self.act = nn.GELU()
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1+x1)
        psi = self.psi(psi)

        return x*psi

class NestedSharedAtrousAttentionResUNetIN(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = Resize(scale_factor=0.5, mode='bilinear')
        self.up = Resize(scale_factor=2, mode='bilinear')

        self.conv0_0 = SharedAtrousResBlockIN(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlockIN(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlockIN(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlockIN(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlockIN(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlockIN(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_1 = AttBlockIN(nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlockIN(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.att1_1 = AttBlockIN(nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlockIN(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.att2_1 = AttBlockIN(nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlockIN(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.att3_1 = AttBlockIN(nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlockIN(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_2 = AttBlockIN(nb_filter[1], nb_filter[0]*2, nb_filter[0])
        self.conv1_2 = SharedAtrousResBlockIN(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.att1_2 = AttBlockIN(nb_filter[2], nb_filter[1]*2, nb_filter[1])
        self.conv2_2 = ResBlockIN(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.att2_2 = AttBlockIN(nb_filter[3], nb_filter[2]*2, nb_filter[2])

        self.conv0_3 = SharedAtrousResBlockIN(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_3 = AttBlockIN(nb_filter[1], nb_filter[0]*3, nb_filter[0])
        self.conv1_3 = SharedAtrousResBlockIN(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.att1_3 = AttBlockIN(nb_filter[2], nb_filter[1]*3, nb_filter[1])

        self.conv0_4 = SharedAtrousResBlockIN(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_4 = AttBlockIN(nb_filter[1], nb_filter[0]*4, nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*5, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_0 = self.att0_1(g=self.up(x1_0), x=x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        
        x1_0 = self.att1_1(g=self.up(x2_0), x=x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        
        x0_01 = self.att0_2(g=self.up(x1_1), x=torch.cat([x0_0, x0_1], 1))
        x0_2 = self.conv0_2(torch.cat([x0_01, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_0 = self.att2_1(g=self.up(x3_0), x=x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_01 = self.att1_2(g=self.up(x2_1), x=torch.cat([x1_0, x1_1], 1))
        x1_2 = self.conv1_2(torch.cat([x1_01, self.up(x2_1)], 1))
        x0_012 = self.att0_3(g=self.up(x1_2), x=torch.cat([x0_0, x0_1, x0_2], 1))
        x0_3 = self.conv0_3(torch.cat([x0_012, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_0 = self.att3_1(g=self.up(x4_0), x=x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_01 = self.att2_2(g=self.up(x3_1), x=torch.cat([x2_0, x2_1], 1))
        x2_2 = self.conv2_2(torch.cat([x2_01, self.up(x3_1)], 1))
        x1_012 = self.att1_3(g=self.up(x2_2), x=torch.cat([x1_0, x1_1, x1_2], 1))
        x1_3 = self.conv1_3(torch.cat([x1_012, self.up(x2_2)], 1))
        x0_0123 = self.att0_4(g=self.up(x1_3), x=torch.cat([x0_0, x0_1, x0_2, x0_3], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0123, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4], 1))
        
        return output

class NestedSharedAtrousAttentionResUNetINDualHead(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = Resize(scale_factor=0.5, mode='bilinear')
        self.up = Resize(scale_factor=2, mode='bilinear')

        self.conv0_0 = SharedAtrousResBlockIN(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlockIN(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlockIN(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlockIN(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlockIN(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlockIN(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_1 = AttBlockIN(nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlockIN(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.att1_1 = AttBlockIN(nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlockIN(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.att2_1 = AttBlockIN(nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlockIN(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.att3_1 = AttBlockIN(nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlockIN(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_2 = AttBlockIN(nb_filter[1], nb_filter[0]*2, nb_filter[0])
        self.conv1_2 = SharedAtrousResBlockIN(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.att1_2 = AttBlockIN(nb_filter[2], nb_filter[1]*2, nb_filter[1])
        self.conv2_2 = ResBlockIN(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.att2_2 = AttBlockIN(nb_filter[3], nb_filter[2]*2, nb_filter[2])

        self.conv0_3 = SharedAtrousResBlockIN(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_3 = AttBlockIN(nb_filter[1], nb_filter[0]*3, nb_filter[0])
        self.conv1_3 = SharedAtrousResBlockIN(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.att1_3 = AttBlockIN(nb_filter[2], nb_filter[1]*3, nb_filter[1])

        self.conv0_4 = SharedAtrousResBlockIN(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_4 = AttBlockIN(nb_filter[1], nb_filter[0]*4, nb_filter[0])
        
        self.conv_iris = SharedAtrousResBlockIN(nb_filter[0]*5, nb_filter[0], nb_filter[0])
        self.final_iris = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
        self.conv_eyelid = SharedAtrousResBlockIN(nb_filter[0]*5, nb_filter[0], nb_filter[0])
        self.final_eyelid = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_0 = self.att0_1(g=self.up(x1_0), x=x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        
        x1_0 = self.att1_1(g=self.up(x2_0), x=x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        
        x0_01 = self.att0_2(g=self.up(x1_1), x=torch.cat([x0_0, x0_1], 1))
        x0_2 = self.conv0_2(torch.cat([x0_01, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_0 = self.att2_1(g=self.up(x3_0), x=x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_01 = self.att1_2(g=self.up(x2_1), x=torch.cat([x1_0, x1_1], 1))
        x1_2 = self.conv1_2(torch.cat([x1_01, self.up(x2_1)], 1))
        x0_012 = self.att0_3(g=self.up(x1_2), x=torch.cat([x0_0, x0_1, x0_2], 1))
        x0_3 = self.conv0_3(torch.cat([x0_012, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_0 = self.att3_1(g=self.up(x4_0), x=x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_01 = self.att2_2(g=self.up(x3_1), x=torch.cat([x2_0, x2_1], 1))
        x2_2 = self.conv2_2(torch.cat([x2_01, self.up(x3_1)], 1))
        x1_012 = self.att1_3(g=self.up(x2_2), x=torch.cat([x1_0, x1_1, x1_2], 1))
        x1_3 = self.conv1_3(torch.cat([x1_012, self.up(x2_2)], 1))
        x0_0123 = self.att0_4(g=self.up(x1_3), x=torch.cat([x0_0, x0_1, x0_2, x0_3], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0123, self.up(x1_3)], 1))
        
        features = torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4], 1)
        
        iris_feat = self.conv_iris(features)
        iris_out = self.final_iris(iris_feat)
        
        eyelid_feat = self.conv_eyelid(features)
        eyelid_out = self.final_eyelid(eyelid_feat)
        
        return iris_out, eyelid_out
               
            
        
    
        
