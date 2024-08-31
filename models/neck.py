import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Conv, Bottleneck

#code taken from #code taken from https://github.com/zeyuwang-zju/LFTDet

class SimAM(nn.Module):
    def __init__(self, lambda_=1e-4):
        super(SimAM, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - torch.mean(x, dim=[2, 3], keepdim=True)).pow(2)
        v = torch.sum(d, dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + self.lambda_)) + 0.5
        return x * torch.sigmoid(E_inv)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x):
        return self.upsample(x)

class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()
        compress_c = 8
        self.weight_level_1 = Conv(inter_dim, compress_c, 1, 1, 0)
        self.weight_level_2 = Conv(inter_dim, compress_c, 1, 1, 0)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(inter_dim, inter_dim, 3, 1, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = F.softmax(self.weight_levels(levels_weight_v), dim=1)
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + input2 * levels_weight[:, 1:2, :, :]
        return self.conv(fused_out_reduced)

class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()
        compress_c = 8
        self.weight_level_1 = Conv(inter_dim, compress_c, 1, 1, 0)
        self.weight_level_2 = Conv(inter_dim, compress_c, 1, 1, 0)
        self.weight_level_3 = Conv(inter_dim, compress_c, 1, 1, 0)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(inter_dim, inter_dim, 3, 1, 1)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = F.softmax(self.weight_levels(levels_weight_v), dim=1)
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + input2 * levels_weight[:, 1:2, :, :] + input3 * levels_weight[:, 2:, :, :]
        return self.conv(fused_out_reduced)

class My_Neck(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256, compress_ratio=8, num_blocks=4):
        super(My_Neck, self).__init__()
        self.in_channels = in_channels
        self.simam = SimAM()
        self.conv0 = Conv(in_channels[0], in_channels[0] // compress_ratio, 1, 1, 0)
        self.conv1 = Conv(in_channels[1], in_channels[1] // compress_ratio, 1, 1, 0)
        self.conv2 = Conv(in_channels[2], in_channels[2] // compress_ratio, 1, 1, 0)
        self.blocks_scalezero1 = Conv(in_channels[0] // compress_ratio, in_channels[0] // compress_ratio, 1, 1, 0)
        self.blocks_scaleone1 = Conv(in_channels[1] // compress_ratio, in_channels[1] // compress_ratio, 1, 1, 0)
        self.blocks_scaletwo1 = Conv(in_channels[2] // compress_ratio, in_channels[2] // compress_ratio, 1, 1, 0)
        self.downsample_scalezero1_2 = Downsample(in_channels[0] // compress_ratio, in_channels[1] // compress_ratio, scale_factor=2)
        self.upsample_scaleone1_2 = Upsample(in_channels[1] // compress_ratio, in_channels[0] // compress_ratio, scale_factor=2)
        self.asff_scalezero1 = ASFF_2(inter_dim=in_channels[0] // compress_ratio)
        self.asff_scaleone1 = ASFF_2(inter_dim=in_channels[1] // compress_ratio)
        self.blocks_scalezero2 = nn.Sequential(*[Bottleneck(in_channels[0] // compress_ratio, in_channels[0] // compress_ratio) for _ in range(num_blocks)])
        self.blocks_scaleone2 = nn.Sequential(*[Bottleneck(in_channels[1] // compress_ratio, in_channels[1] // compress_ratio) for _ in range(num_blocks)])
        self.downsample_scalezero2_2 = Downsample(in_channels[0] // compress_ratio, in_channels[1] // compress_ratio, scale_factor=2)
        self.downsample_scalezero2_4 = Downsample(in_channels[0] // compress_ratio, in_channels[2] // compress_ratio, scale_factor=4)
        self.downsample_scaleone2_2 = Downsample(in_channels[1] // compress_ratio, in_channels[2] // compress_ratio, scale_factor=2)
        self.upsample_scaleone2_2 = Upsample(in_channels[1] // compress_ratio, in_channels[0] // compress_ratio, scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(in_channels[2] // compress_ratio, in_channels[1] // compress_ratio, scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(in_channels[2] // compress_ratio, in_channels[0] // compress_ratio, scale_factor=4)
        self.asff_scalezero2 = ASFF_3(inter_dim=in_channels[0] // compress_ratio)
        self.asff_scaleone2 = ASFF_3(inter_dim=in_channels[1] // compress_ratio)
        self.asff_scaletwo2 = ASFF_3(inter_dim=in_channels[2] // compress_ratio)
        self.blocks_scalezero3 = nn.Sequential(*[Bottleneck(in_channels[0] // compress_ratio, in_channels[0] // compress_ratio) for _ in range(num_blocks)])
        self.blocks_scaleone3 = nn.Sequential(*[Bottleneck(in_channels[1] // compress_ratio, in_channels[1] // compress_ratio) for _ in range(num_blocks)])
        self.blocks_scaletwo3 = nn.Sequential(*[Bottleneck(in_channels[2] // compress_ratio, in_channels[2] // compress_ratio) for _ in range(num_blocks)])
        self.conv00 = Conv(in_channels[0] // compress_ratio, out_channels, 1, 1, 0)
        self.conv11 = Conv(in_channels[1] // compress_ratio, out_channels, 1, 1, 0)
        self.conv22 = Conv(in_channels[2] // compress_ratio, out_channels, 1, 1, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        x0, x1, x2 = inputs
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)
        scalezero = self.simam(scalezero)
        scaleone = self.simam(scaleone)
        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)
        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)
        scalezero = self.simam(scalezero)
        scaleone = self.simam(scaleone)
        scaletwo = self.simam(scaletwo)
        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)
        out0 = self.conv00(x0)
        out1 = self.conv11(x1)
        out2 = self.conv22(x2)
        return tuple([out0, out1, out2])
