import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WaveDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, decay_init=0.1, velocity_init=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.decay = nn.Parameter(torch.ones(1) * decay_init)
        self.velocity = nn.Parameter(torch.ones(1) * velocity_init)
        
        self.proj = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.freq_weight = nn.Parameter(torch.ones(1, 1, 1, 1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_freq = torch.fft.fft2(x, norm='ortho')
        
        x_freq_shifted = torch.fft.fftshift(x_freq, dim=(-2, -1))
        
        decay_factor = torch.exp(-torch.abs(self.decay) / 2)
        
        center_h, center_w = H // 2, W // 2
        new_h, new_w = H // self.scale_factor, W // self.scale_factor
        
        start_h = center_h - new_h // 2
        start_w = center_w - new_w // 2
        end_h = start_h + new_h
        end_w = start_w + new_w
        
        x_freq_low = x_freq_shifted[:, :, start_h:end_h, start_w:end_w]
        
        x_freq_low = x_freq_low * decay_factor * self.freq_weight
        
        x_freq_unshifted = torch.fft.ifftshift(x_freq_low, dim=(-2, -1))
        
        x_spatial = torch.fft.ifft2(x_freq_unshifted, s=(new_h, new_w), norm='ortho').real
        
        out = self.bn(self.proj(x_spatial))
        
        return out


class FrequencyAwareDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.low_freq_pool = nn.AvgPool2d(scale_factor, scale_factor)
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=scale_factor, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        x_low = self.low_freq_pool(x)
        
        x_high = x - F.interpolate(self.low_freq_pool(x), size=x.shape[2:], mode='nearest')
        x_high = self.high_freq_conv(x_high)
        
        gate = self.freq_gate(x)
        gate_low = gate[:, 0:1, :, :].expand_as(x_low)
        gate_high = gate[:, 1:2, :, :].expand_as(x_high)
        
        x_low_weighted = x_low * gate_low
        x_high_weighted = x_high * gate_high
        
        out = self.fusion(torch.cat([x_low_weighted, x_high_weighted], dim=1))
        
        return out


class OscillatoryDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, num_oscillations=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_oscillations = num_oscillations
        
        self.oscillation_weights = nn.Parameter(torch.ones(num_oscillations) / num_oscillations)
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                         stride=scale_factor, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
            )
            for _ in range(num_oscillations)
        ])
        
        self.phase_offsets = nn.Parameter(torch.linspace(0, math.pi, num_oscillations))
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        outputs = []
        weights = F.softmax(self.oscillation_weights, dim=0)
        
        for i, branch in enumerate(self.branches):
            out = branch(x)
            phase_factor = torch.cos(self.phase_offsets[i])
            outputs.append(out * phase_factor)
        
        out = sum(w * o for w, o in zip(weights, outputs))
        
        return self.proj(out)


class WaveletDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        
        self.ll_conv = nn.Conv2d(in_channels, in_channels, 2, stride=2, groups=in_channels, bias=False)
        self.lh_conv = nn.Conv2d(in_channels, in_channels, 2, stride=2, groups=in_channels, bias=False)
        self.hl_conv = nn.Conv2d(in_channels, in_channels, 2, stride=2, groups=in_channels, bias=False)
        self.hh_conv = nn.Conv2d(in_channels, in_channels, 2, stride=2, groups=in_channels, bias=False)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.detail_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, 3, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ll = self.ll_conv(x)
        lh = self.lh_conv(x)
        hl = self.hl_conv(x)
        hh = self.hh_conv(x)
        
        detail = torch.cat([lh, hl, hh], dim=1)
        gate = self.detail_gate(detail)
        
        detail_gated = lh * gate[:, 0:1] + hl * gate[:, 1:2] + hh * gate[:, 2:3]
        
        out = self.fusion(torch.cat([ll, detail_gated], dim=1))
        
        return out


class AdaptiveWaveDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.decay = nn.Parameter(torch.ones(1) * 0.1)
        self.velocity = nn.Parameter(torch.ones(1) * 1.0)
        
        self.content_adaptor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 2, 1),
            nn.Sigmoid()
        )
        
        self.low_freq_branch = nn.Sequential(
            nn.AvgPool2d(scale_factor, scale_factor),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=scale_factor, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.freq_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        adapt_weights = self.content_adaptor(x)
        w_low = adapt_weights[:, 0:1, :, :]
        w_high = adapt_weights[:, 1:2, :, :]
        
        x_low = self.low_freq_branch(x)
        x_high = self.high_freq_branch(x)
        
        decay_factor = torch.exp(-torch.abs(self.decay) / 2)
        velocity_factor = torch.abs(self.velocity)
        
        x_low = x_low * (1 + decay_factor * w_low)
        x_high = x_high * velocity_factor * w_high
        
        out = x_low + x_high
        
        x_freq = torch.fft.fft2(out)
        x_freq = x_freq * (1 + decay_factor * 0.1)
        out_freq = torch.fft.ifft2(x_freq).real
        
        out = out + self.freq_enhance(out_freq)
        
        return out


class HybridDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.max_branch = nn.Sequential(
            nn.MaxPool2d(scale_factor, scale_factor),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.avg_branch = nn.Sequential(
            nn.AvgPool2d(scale_factor, scale_factor),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.wave_branch = AdaptiveWaveDownsample(in_channels, out_channels, scale_factor)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 3, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x_max = self.max_branch(x)
        x_avg = self.avg_branch(x)
        x_wave = self.wave_branch(x)
        
        gate = self.gate(x_max + x_avg + x_wave)
        
        out = gate[:, 0:1] * x_max + gate[:, 1:2] * x_avg + gate[:, 2:3] * x_wave
        
        return self.fusion(out)


def get_downsample_module(name, in_channels, out_channels, scale_factor=2):
    modules = {
        'wave': WaveDownsample,
        'freq_aware': FrequencyAwareDownsample,
        'oscillatory': OscillatoryDownsample,
        'wavelet': WaveletDownsample,
        'adaptive_wave': AdaptiveWaveDownsample,
        'hybrid': HybridDownsample,
    }
    
    if name in modules:
        return modules[name](in_channels, out_channels, scale_factor)
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=scale_factor, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
