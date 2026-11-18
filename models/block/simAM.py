import torch
from torch import nn
import torch.nn.functional as F


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


class SelectiveSimAM(nn.Module):
    """
    Selective SimAM: 结合SimAM的无参数特性与选择性注意力机制
    
    核心创新：
    1. 双向门控选择：根据解码器需求选择编码器特征
    2. 自适应融合：动态调整注意力强度
    3. 轻量化设计：参数量增加极少
    """
    def __init__(self, channels, e_lambda=1e-4, reduction=4):
        super().__init__()
        self.e_lambda = e_lambda
        self.channels = channels
        
        mid_channels = max(channels // reduction, 8)
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)

    def _simam_attention(self, x):
        """SimAM注意力计算（无参数）"""
        b, c, h, w = x.size()
        n = w * h - 1
        
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return torch.sigmoid(y)

    def forward(self, encoder_feat, decoder_feat):
        """
        Args:
            encoder_feat: 编码器特征 (B, C, H, W)
            decoder_feat: 解码器特征 (B, C, H, W)
        Returns:
            注意力精炼后的特征
        """
        encoder_attention = self._simam_attention(encoder_feat)
        decoder_attention = self._simam_attention(decoder_feat)
        
        gate = self.gate(encoder_feat + decoder_feat)
        
        attention = gate[:, 0:1] * encoder_attention * torch.sigmoid(self.alpha) + \
                   gate[:, 1:2] * decoder_attention * torch.sigmoid(self.beta)
        
        return encoder_feat * attention


if __name__ == '__main__':
    x_enc = torch.randn(4, 64, 128, 128).cuda()
    x_dec = torch.randn(4, 64, 128, 128).cuda()
    
    model = SelectiveSimAM(channels=64).cuda()
    
    out = model(x_enc, x_dec)
    print(f"Output shape: {out.shape}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params}")
