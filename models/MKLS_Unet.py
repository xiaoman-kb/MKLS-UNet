import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import os
from functools import partial
import math
import timm
from timm.models.layers import trunc_normal_
from timm.models.helpers import named_apply
from models.block.ls_conv import LKP
from models.block.ska import SKA
from models.block.simAM import SelectiveSimAM

from models.block.wave_downsample import (
    WaveDownsample, FrequencyAwareDownsample, OscillatoryDownsample,
    WaveletDownsample, AdaptiveWaveDownsample, HybridDownsample
)

def gcd(a, b):

    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    """
    初始化神经网络模块权重的函数
    Args:
        module: 神经网络模块
        name: 模块名称
        scheme: 权重初始化方案，可选值为 'normal', 'trunc_normal', 'xavier_normal', 'kaiming_normal' 或空（使用efficientnet风格初始化）
    """
    if isinstance(module, nn.Conv2d):
        # 卷积层权重初始化
        if scheme == 'normal':
            # 使用正态分布初始化权重
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                # 将偏置初始化为0
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            # 使用截断正态分布初始化权重
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            # 使用Xavier正态分布初始化权重
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            # 使用Kaiming正态分布初始化权重，适用于ReLU激活函数
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    """
    实现通道混洗(Channel Shuffle)操作，用于跨组信息交换
    参数:
        x: 输入张量，形状为(batchsize, num_channels, height, width)
        groups: 分组数，用于将通道分成多个组
    返回:
        混洗后的张量，形状与输入相同
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups  # 计算每个组的通道数

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class simam_module(nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class GroupedAttentionGate(nn.Module):
    """
    分组选择性SimAM注意力桥 (GSAB: Grouped Selective SimAM Attention Bridge)
    
    核心创新：
    1. 分组卷积通道调整：轻量化设计，减少参数量
    2. SelectiveSimAM：双向选择性注意力，根据解码器需求精炼编码器特征
    3. 语义对齐：缩小编码器-解码器特征之间的语义差距
    
    参数:
        F_g (int): 解码器特征g的通道数
        F_l (int): 编码器特征x的通道数
        F_int (int): 中间特征的通道数
        kernel_size (int): 卷积核大小，默认为1
        groups (int): 分组卷积的组数，默认为1
    """
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1):
        super(GroupedAttentionGate, self).__init__()
        if kernel_size == 1:
            groups = 1
        
        self.F_int = F_int
        self.F_l = F_l
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.selective_simam = SelectiveSimAM(channels=F_int, reduction=4)
        
        self.proj = nn.Sequential(
            nn.Conv2d(F_int, F_l, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_l)
        )

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        """
        前向传播
        参数:
            g: 解码器特征 (B, F_g, H, W) - 已上采样到与x相同分辨率
            x: 编码器特征 (B, F_l, H, W)
        返回:
            精炼后的编码器特征 (B, F_l, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        out = self.selective_simam(x1, g1)
        
        out = self.proj(out)
        
        return out
class LSIR(nn.Module):
    """
    使用 LSConv 核心机制的轻量倒残差块
    """
    def __init__(self, in_c, out_c, stride, kernel_sizes,expansion_factor=2, add=True, activation='relu6'):
        super(LSIR, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.add = add
        self.in_c = in_c
        self.out_c = out_c
        self.use_skip_connection = (stride == 1)
        self.kernel_sizes = kernel_sizes
        self.n_scales = len(kernel_sizes)
        self.ex_c = int(in_c * expansion_factor)

        # Pointwise Expansion
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )

        if self.add:
            self.combined_channels = self.ex_c
        else:
            self.combined_channels = self.ex_c * self.n_scales
        groups = max(1, self.in_c // 8)
        # Multi-Kernel LKP: 每个大核尺寸对应一个独立的 Large Kernel Predictor
        self.lkps = nn.ModuleList([
            LKP(dim=self.ex_c, lks=ks, sks=3, groups=groups)
            for ks in kernel_sizes
        ])
        self.ska = SKA()
        self.bn_ska = nn.BatchNorm2d(self.ex_c)

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.ex_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c)
        )

        # Skip connection adapter
        if self.use_skip_connection and (in_c != out_c):
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        identity = x

        # Expansion
        pout1 = self.pconv1(x)  # (B, ex_c, H, W)

        # LSConv 核心：Multi-Kernel LKP + SKA（无残差）
        # 多个大核分别预测小核权重，融合后作为 SKA 的动态卷积核
        w = 0
        for lkp in self.lkps:
            w = w + lkp(pout1)  # 每个 (B, G, 9, H, W)
        if self.n_scales > 1:
            w = w / self.n_scales  # 平均，保持权重量级与单一大核一致
        ska_out = self.ska(pout1, w)  # (B, ex_c, H, W)
        ska_out = self.bn_ska(ska_out)

        # Projection
        out = self.pconv2(ska_out)

        # Skip connection
        if self.use_skip_connection:
            if self.in_c != self.out_c:
                identity = self.conv1x1(identity)
            return identity + out
        else:
            return out


def ls_mkir_bottleneck(in_c, out_c, n, s,kernel_sizes, expansion_factor=2, activation='relu6'):
    """
    Create a series of LS-MKIR blocks (LS-based Multi-Kernel Inverted Residual).
    """
    convs = []
    # 第一个 block 可能有 stride=s
    xx = LSIR(in_c, out_c, s, expansion_factor=expansion_factor, activation=activation,kernel_sizes=kernel_sizes)
    convs.append(xx)

    # 后续 block stride=1
    for i in range(1, n):
        xx = LSIR(out_c, out_c, 1, expansion_factor=expansion_factor, activation=activation, kernel_sizes=kernel_sizes)
        convs.append(xx)

    return nn.Sequential(*convs)


# channels = [4,8,16,24,32] -T
# channels = [8,16,32,48,80] -S
# channels = [16,32,64,96,160] 
# channels = [32,64,128,192,320] M
# channels = [64,128,256,384,512] -L

class MKLS_UNet(nn.Module):

    def __init__(self, num_classes=1, in_channels=3, channels=[64,128,256,384,512], depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[5,7,9], expansion_factor=2, gag_kernel=3, downsample_type='adaptive_wave', **kwargs):
        super().__init__()
        
        self.downsample_type = downsample_type
        
        DOWNSAMPLE_MODULES = {
            'maxpool': None,
            'wave': WaveDownsample,
            'freq_aware': FrequencyAwareDownsample,
            'oscillatory': OscillatoryDownsample,
            'wavelet': WaveletDownsample,
            'adaptive_wave': AdaptiveWaveDownsample,
            'hybrid': HybridDownsample,
        }

        self.encoder1 = ls_mkir_bottleneck(in_channels, channels[0], depths[0], 1, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)
        self.encoder2 = ls_mkir_bottleneck(channels[0], channels[1], depths[1], 2, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)
        self.encoder3 = ls_mkir_bottleneck(channels[1], channels[2], depths[2], 2, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)
        self.encoder4 = ls_mkir_bottleneck(channels[2], channels[3], depths[3], 2, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)
        self.encoder5 = ls_mkir_bottleneck(channels[3], channels[4], depths[4], 2, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)

        if downsample_type != 'maxpool' and downsample_type in DOWNSAMPLE_MODULES:
            DownModule = DOWNSAMPLE_MODULES[downsample_type]
            self.down1 = DownModule(channels[0], channels[0], scale_factor=2)
            self.down2 = DownModule(channels[1], channels[1], scale_factor=2)
            self.down3 = DownModule(channels[2], channels[2], scale_factor=2)
            self.down4 = DownModule(channels[3], channels[3], scale_factor=2)
            self.down5 = DownModule(channels[4], channels[4], scale_factor=2)

        self.AG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
                                        kernel_size=gag_kernel, groups=channels[3] // 2)
        self.AG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
                                        kernel_size=gag_kernel, groups=channels[2] // 2)
        self.AG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
                                        kernel_size=gag_kernel, groups=channels[1] // 2)
        self.AG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2,
                                        kernel_size=gag_kernel, groups=channels[0] // 2)

        self.decoder1 = ls_mkir_bottleneck(channels[4], channels[3], 1, 1, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)
        self.decoder2 = ls_mkir_bottleneck(channels[3], channels[2], 1, 1, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)
        self.decoder3 = ls_mkir_bottleneck(channels[2], channels[1], 1, 1, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)
        self.decoder4 = ls_mkir_bottleneck(channels[1], channels[0], 1, 1, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)
        self.decoder5 = ls_mkir_bottleneck(channels[0], channels[0], 1, 1, expansion_factor=expansion_factor, kernel_sizes=kernel_sizes)


        self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B = x.shape[0]
        ### Encoder
        ### Stage 1
        enc1 = self.encoder1(x)
        if self.downsample_type != 'maxpool':
            out = self.down1(enc1)
        else:
            out = F.max_pool2d(enc1, 2, 2)
        t1 = out
        ### Stage 2
        enc2 = self.encoder2(out)
        if self.downsample_type != 'maxpool':
            out = self.down2(enc2)
        else:
            out = F.max_pool2d(enc2, 2, 2)
        t2 = out
        ### Stage 3
        enc3 = self.encoder3(out)
        if self.downsample_type != 'maxpool':
            out = self.down3(enc3)
        else:
            out = F.max_pool2d(enc3, 2, 2)
        t3 = out

        ### Stage 4
        enc4 = self.encoder4(out)
        if self.downsample_type != 'maxpool':
            out = self.down4(enc4)
        else:
            out = F.max_pool2d(enc4, 2, 2)
        t4 = out

        ### Bottleneck
        enc5 = self.encoder5(out)
        if self.downsample_type != 'maxpool':
            out = self.down5(enc5)
        else:
            out = F.max_pool2d(enc5, 2, 2)

        ### Stage 4
        # out = self.SiMAM1(out)

        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        t4 = self.AG1(g=out, x=t4)
        out = torch.add(out, t4)

        ### Stage 3
        # out = self.SiMAM2(out)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        p1 = F.interpolate(self.out1(out), scale_factor=(8, 8), mode='bilinear')
        t3 = self.AG2(g=out, x=t3)
        out = torch.add(out, t3)

        # out = self.SiMAM3(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        p2 = F.interpolate(self.out2(out), scale_factor=(4, 4), mode='bilinear')
        t2 = self.AG3(g=out, x=t2)
        out = torch.add(out, t2)

        # out = self.SiMAM4(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        p3 = F.interpolate(self.out3(out), scale_factor=(2, 2), mode='bilinear')
        t1 = self.AG4(g=out, x=t1)
        out = torch.add(out, t1)

        # out = self.SiMAM5(out)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        p4 = self.out4(out)

        return torch.sigmoid(p4)  # [p4, p3, p2, p1]


# EOF

