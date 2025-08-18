# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Author  : Qianwen Ma
# @File    : DWTFreqNet.py
# @Software: PyCharm
# coding=utf-8
#Time: 2025.1.24

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from decoder_fuse.transformer_dec_fuse_none_posqkv_dropout import TransFuseModel
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
import numpy as np
from thop import profile

def get_DWTFreqNet_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1

    # ********** useless **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)

def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class HaarWaveletTransform(nn.Module):
    def __init__(self):
        super(HaarWaveletTransform, self).__init__()

        # Define Haar wavelet filters
        self.haar_matrix_LL = torch.tensor([
            [1 / 2, 1 / 2],
            [1 / 2, 1 / 2]
        ], dtype=torch.float32).reshape(1, 1, 2, 2)

        self.haar_matrix_LH = torch.tensor([
            [1 / 2, -1 / 2],
            [1 / 2, -1 / 2]
        ], dtype=torch.float32).reshape(1, 1, 2, 2)

        self.haar_matrix_HL = torch.tensor([
            [1 / 2, 1 / 2],
            [-1 / 2, -1 / 2]
        ], dtype=torch.float32).reshape(1, 1, 2, 2)

        self.haar_matrix_HH = torch.tensor([
            [1 / 2, -1 / 2],
            [-1 / 2, 1 / 2]
        ], dtype=torch.float32).reshape(1, 1, 2, 2)

    def forward(self, x):
        B, C, H, W = x.size()

        # Ensure the input tensor has even dimensions
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("Input dimensions must be even.")

        # Move haar_matrix to the same device as x
        device = x.device
        haar_matrix_LL = self.haar_matrix_LL.to(device)
        haar_matrix_LH = self.haar_matrix_LH.to(device)
        haar_matrix_HL = self.haar_matrix_HL.to(device)
        haar_matrix_HH = self.haar_matrix_HH.to(device)

        # Reshape input tensor for convolution
        x_reshaped = x.view(B*C, 1, H, W)

        # Perform convolutions for all channels at once
        out_LL = F.conv2d(x_reshaped, haar_matrix_LL, stride=2, padding=0).view(B, C, H//2, W//2)
        out_LH = F.conv2d(x_reshaped, haar_matrix_LH, stride=2, padding=0).view(B, C, H//2, W//2)
        out_HL = F.conv2d(x_reshaped, haar_matrix_HL, stride=2, padding=0).view(B, C, H//2, W//2)
        out_HH = F.conv2d(x_reshaped, haar_matrix_HH, stride=2, padding=0).view(B, C, H//2, W//2)

        return out_LL, out_LH, out_HL, out_HH


###反向小波变换######
class InverseHaarWaveletTransform(nn.Module):
    def __init__(self):
        super(InverseHaarWaveletTransform, self).__init__()

        # Define inverse Haar wavelet filters
        self.inv_haar_matrix = torch.tensor([
            [1 / 2, 1 / 2],
            [1 / 2, 1 / 2],
            [1 / 2, -1 / 2],
            [1 / 2, -1 / 2],
            [1 / 2, 1 / 2],
            [-1 / 2, -1 / 2],
            [1 / 2, -1 / 2],
            [-1 / 2, 1 / 2]
        ], dtype=torch.float32).reshape(4, 1, 2, 2)  # Adjusted shape

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.size()

        # Stack the coefficients
        coeffs = torch.stack([LL, LH, HL, HH], dim=2)  # Shape (B, C, 4, H, W)

        # Move inv_haar_matrix to the same device as coefficients
        device = LL.device
        inv_haar_matrix = self.inv_haar_matrix.to(device)

        # Perform the inverse Haar wavelet transform
        output = F.conv_transpose2d(coeffs.view(B * C, 4, H, W), inv_haar_matrix, stride=2, padding=0)

        return output.view(B, C, H * 2, W * 2)

class WaveDownattention(nn.Module):###这个是那个小波注意力机制
    def __init__(self, in_channels):
        super().__init__()

        # self.dwt = DWT_2D(wave='haar')
        self.conv_A_H = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv_A_V = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv_A_D = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.to_att = nn.Sequential(
                    nn.Conv2d(2, 1, 1, 1, 0),
                    nn.Sigmoid()
        )
        self.att_weights = nn.Parameter(torch.ones(3) / 3)
        # self.pw = nn.Conv2d(in_channels * 4, in_channels * 2, 1, 1, 0)

    def forward(self, A,H,V,D):
        # x = self.dwt(x)
        # x_ll, x_lh, x_hl, x_hh = x.chunk(4, dim=1)
        # get attention
        AH =  self.conv_A_H(A + H)
        AV =  self.conv_A_V(A + V)
        AD = self.conv_A_D(A + D)

##空间注意力机制
        AH_att_maxpool, _ = torch.max(AH, dim=1, keepdim=True)
        # 在通道维度上平均池化 [b,1,h,w]
        AH_att_avgpool = torch.mean(AH, dim=1, keepdim=True)
        AH_att = torch.cat([AH_att_maxpool, AH_att_avgpool], dim=1)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]

        AV_att_maxpool, _ = torch.max(AV, dim=1, keepdim=True)
        # 在通道维度上平均池化 [b,1,h,w]
        AV_att_avgpool = torch.mean(AV, dim=1, keepdim=True)
        AV_att = torch.cat([AV_att_maxpool, AV_att_avgpool], dim=1)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]

        AD_att_maxpool, _ = torch.max(AD, dim=1, keepdim=True)
        # 在通道维度上平均池化 [b,1,h,w]
        AD_att_avgpool = torch.mean(AD, dim=1, keepdim=True)
        AD_att = torch.cat([AD_att_maxpool, AD_att_avgpool], dim=1)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]

        # wave_att = AH_att+AV_att+AD_att
        wave_att = self.att_weights[0] * AH_att + self.att_weights[1] * AV_att + self.att_weights[2] * AD_att

        ##空间注意力机制

        att_map = self.to_att(wave_att)
        # squeeze
        # x_s = self.pw(x)
        o = torch.mul(A, att_map) + A  #这里虽然mul后的两个tensor维度不统一，但是通过广播机制能够将那个1的维度自行复制，以达到维度统一
        # hi_bands = torch.cat([x_lh, x_hl, x_hh], dim=1)
        return o #hi_bands #第二个分量好像是高频分量，原网络用于上采样

class DWTFreqNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel  # basic channel 64
        block = Res_block
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_wavelet_inchannel_global = self._make_layer(block, 32*3, in_channels)
        self.conv_wavelet_inchannel_local = self._make_layer(block, 32, in_channels)
        self.inc = self._make_layer(block, n_channels, 32)
        self.global_encoder1_1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.global_encoder2_1 = self._make_layer(block, in_channels * 6, in_channels * 4, 1)  # 64  128
        self.global_encoder3_1 = self._make_layer(block, in_channels * 12, in_channels * 8, 1)  # 64  128
        self.global_encoder4_1 = self._make_layer(block, in_channels * 24, in_channels * 8, 1)  # 64  128
        self.global_encoder1_2 = self._make_layer(block, in_channels * 2 + in_channels * 4, in_channels * 2,1 )
        self.global_encoder2_2 = self._make_layer(block, in_channels * 4 + in_channels * 6 + in_channels * 8, in_channels * 4, 1)
        self.global_encoder3_2 = self._make_layer(block, in_channels * 8 + in_channels * 12 + in_channels * 8, in_channels * 8, 1)
        self.global_encoder1_3 = self._make_layer(block, in_channels * 2 * 2 + in_channels * 4, in_channels * 2, 1)
        self.global_encoder2_3 = self._make_layer(block, in_channels * 4 * 2 + in_channels * 6 + in_channels * 8, in_channels * 4, 1)
        self.global_encoder1_4 = self._make_layer(block, in_channels * 2 * 3 + in_channels * 4, in_channels * 2, 1)


        ##局部低频
        self.local_encoder1_1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.local_encoder2_1 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)  # 64  128
        self.local_encoder3_1 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  # 64  128
        self.local_encoder4_1 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)  # 64  128
        self.local_encoder1_2 = self._make_layer(block, in_channels * 2 + in_channels * 4, in_channels * 2)
        self.local_encoder2_2 = self._make_layer(block, in_channels * 4 + in_channels * 2 + in_channels * 8, in_channels * 4, 1)
        self.local_encoder3_2 = self._make_layer(block, in_channels * 8 + in_channels * 4 + in_channels * 8, in_channels * 8, 1)
        self.local_encoder1_3 = self._make_layer(block, in_channels * 2 * 2 + in_channels * 4, in_channels * 2)
        self.local_encoder2_3 = self._make_layer(block, in_channels * 4 * 2 + in_channels * 2 + in_channels * 8, in_channels * 4, 1)
        self.local_encoder1_4 = self._make_layer(block, in_channels * 2 * 3 + in_channels * 4, in_channels * 2)
        ##局部低频

        ##Dense全局结构改变通道
        self.global_channel1_2 = nn.Conv2d(in_channels * 4, in_channels * 4 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel2_2 = nn.Conv2d(in_channels * 8, in_channels * 8 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel3_2 = nn.Conv2d(in_channels * 8, in_channels * 8 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel1_3 = nn.Conv2d(in_channels * 4, in_channels * 4 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel2_3 = nn.Conv2d(in_channels * 8, in_channels * 8 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel1_4 = nn.Conv2d(in_channels * 4, in_channels * 4 * 3, kernel_size=(1, 1), stride=(1, 1))


        self.from_input2out = self._make_layer(block, in_channels, in_channels // 2, 1)
        self.outc_global = nn.Conv2d(in_channels // 2, 1, kernel_size=(1, 1), stride=(1, 1))

        ##小波解码器通道对齐##

        self.decoder4_channel = self._make_layer(block, in_channels * 8, in_channels * 24, 1)
        self.decoder3_channel = self._make_layer(block, in_channels * 8, in_channels * 12, 1)
        self.decoder2_channel = self._make_layer(block, in_channels * 4, in_channels * 6, 1)
        self.decoder1_channel = self._make_layer(block, in_channels * 2, in_channels * 3, 1)

        self.decoder3_channel_local = self._make_layer(block, in_channels * 8, in_channels * 4, 1)
        self.decoder2_channel_local = self._make_layer(block, in_channels * 4, in_channels * 2, 1)
        self.decoder1_channel_local = self._make_layer(block, in_channels * 2, in_channels * 1, 1)

        ##局部低频
        ##局部低频

        ##小波相关##
        self.har = HaarWaveletTransform()
        self.inversehar = InverseHaarWaveletTransform()

        ##构造我需要的图像尺寸进入Transformer相关##
        self.wavel_channel_down_x1_global_output_1_4 = nn.Conv2d(in_channels * 2, 1, kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_x_inut = nn.Conv2d(32, 1,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_x2_global_output_2_3 = nn.Conv2d(in_channels * 4, 1,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_x3_global_output_3_2 = nn.Conv2d(in_channels * 8, 1,kernel_size=(1, 1), stride=(1, 1))

        self.stand_cahnnel1 = conv1x1_bn_relu(in_channels * 2, 128)
        self.stand_cahnnel2 = conv1x1_bn_relu(in_channels * 4, 128)
        self.stand_cahnnel3 = conv1x1_bn_relu(in_channels * 8, 128)
        self.stand_cahnnel_input = conv1x1_bn_relu(32, 128)

        self.TransTo_input = TransFuseModel(num_blocks=1, x_channels=128, nx=4096, y_channels=128, ny=5376)
        self.TransTo3e = TransFuseModel(num_blocks=1, x_channels=128, nx=256, y_channels=128, ny=9216)
        self.TransTo2e = TransFuseModel(num_blocks=1, x_channels=128, nx=1024, y_channels=128, ny=8448)
        self.TransTo1e = TransFuseModel(num_blocks=1, x_channels=128, nx=4096, y_channels=128, ny=5376)

        self.wavel_channel_down_to_origin_x_inut = nn.Conv2d(128, 32,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_to_origin_x1_global_output_1_4 = nn.Conv2d(128, in_channels * 2,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_to_origin_x2_global_output_2_3 = nn.Conv2d(128, in_channels * 4,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_to_origin_x3_global_output_3_2 = nn.Conv2d(128, in_channels * 8,kernel_size=(1, 1), stride=(1, 1))

        self.wavel_channel_up_x_inut = nn.Conv2d(1, 32,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_up_x1_global_output_1_4 = nn.Conv2d(1, in_channels * 2,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_up_x2_global_output_2_3 = nn.Conv2d(1, in_channels * 4,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_up_x3_global_output_3_2 = nn.Conv2d(1, in_channels * 8,kernel_size=(1, 1), stride=(1, 1))

        ##注意力相关##
        self.wave_att_input_t = WaveDownattention(32)
        self.wave_att_f1 = WaveDownattention(in_channels * 2)
        self.wave_att_f2 = WaveDownattention(in_channels * 4)
        self.wave_att_f3 = WaveDownattention(in_channels * 8)

        ###输出相关的卷积
        self.out4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        self.out3 = self._make_layer(block, in_channels * 4, in_channels * 4, 1)
        self.out2 = self._make_layer(block, in_channels * 2, in_channels * 2, 1)
        self.out1 = self._make_layer(block, in_channels, in_channels, 1)


        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        ##
        x_inut = self.inc(x)  # 32 256 256
        A1_1,H1_1,V1_1,D1_1 = self.har(x_inut)   ##H/2

    ##第一层重构网络
        concat_hvd_global_1_1 = torch.cat((H1_1, V1_1, D1_1), dim=1)
        x1_global = self.conv_wavelet_inchannel_global(concat_hvd_global_1_1)  ## 32  H/2
        x1_global_output_1_1 = self.global_encoder1_1(x1_global)  # 64 128 128

        x1_local = self.conv_wavelet_inchannel_local(A1_1)
        x1_local_output_1_1 = self.local_encoder1_1(x1_local)  # 64 128 128

    ##第二层重构网络
        A2_1, H2_1, V2_1, D2_1 = self.har(x1_local_output_1_1)
        x2_global_2_1 = torch.cat((H2_1, V2_1, D2_1), dim=1)
        x2_global_output_2_1 = self.global_encoder2_1(x2_global_2_1)  # 128 64  64

        x2_local_output_2_1 = self.local_encoder2_1(A2_1)  # 128 64  64

    ##构建1_2
        x1_global_output_1_2 = self.global_encoder1_2(torch.cat([x1_global_output_1_1,self.up(x2_global_output_2_1)],1))
        x1_global_input_1_2_f_2_1 = self.global_channel1_2(x2_global_output_2_1)
        H_1_2_f_2_1, V_1_2_f_2_1, D_1_2_f_2_1 = torch.chunk(x1_global_input_1_2_f_2_1, 3, 1)
        x2_inversewavel_2_1 = self.inversehar(x2_local_output_2_1, H_1_2_f_2_1, V_1_2_f_2_1, D_1_2_f_2_1)
        x1_local_output_1_2 = self.local_encoder1_2(torch.cat([x1_local_output_1_1, x2_inversewavel_2_1],1))

    ##第三层重构网络
        A3_1, H3_1, V3_1, D3_1 = self.har(x2_local_output_2_1)
        x3_global_3_1 = torch.cat((H3_1, V3_1, D3_1), dim=1)
        x3_global_output_3_1 = self.global_encoder3_1(x3_global_3_1)  # 256 32  32

        x3_local_output_3_1 = self.local_encoder3_1(A3_1)  # 256 32  32

    ##构建2_2 来自2_1 1_2 3_1##调整通道了 不是缩减为1了
        A2_2_onec, H2_2onec, V2_2onec, D2_2onec = self.har(x1_local_output_1_2)
        x2_global_2_2_threec = torch.cat((H2_2onec, V2_2onec, D2_2onec), dim=1)
        x2_global_output_2_2 = self.global_encoder2_2(torch.cat([x2_global_output_2_1, x2_global_2_2_threec, self.up(x3_global_output_3_1)],1))

        x1_global_input_2_2_f_3_1 = self.global_channel2_2(x3_global_output_3_1)
        H_2_2_f_3_1, V_2_2_f_3_1, D_2_2_f_3_1 = torch.chunk(x1_global_input_2_2_f_3_1, 3, 1)
        x3_inversewavel_3_1 = self.inversehar(x3_local_output_3_1, H_2_2_f_3_1, V_2_2_f_3_1, D_2_2_f_3_1)
        x2_local_output_2_2 = self.local_encoder2_2(torch.cat([x2_local_output_2_1, A2_2_onec, x3_inversewavel_3_1], 1))

    ##第四层重构网络
        A4_1, H4_1, V4_1, D4_1 = self.har(x3_local_output_3_1)
        x4_global_4_1 = torch.cat((H4_1, V4_1, D4_1), dim=1)
        x4_global_output_4_1 = self.global_encoder4_1(x4_global_4_1)  # 256 16  16

        x4_local_output_4_1 = self.local_encoder4_1(A4_1)  # 256 16  16

    ##构建3_2 来自 3_1 2_2 4_1##调整了通道
        A3_2_onec, H3_2onec, V3_2onec, D3_2onec = self.har(x2_local_output_2_2)
        x2_global_3_2_threec = torch.cat((H3_2onec, V3_2onec, D3_2onec), dim=1)
        x3_global_output_3_2 = self.global_encoder3_2(torch.cat([x3_global_output_3_1, x2_global_3_2_threec, self.up(x4_global_output_4_1)],1))

        x1_global_input_3_2_f_4_1 = self.global_channel3_2(x4_global_output_4_1)
        H_3_2_f_4_1, V_3_2_f_4_1, D_3_2_f_4_1 = torch.chunk(x1_global_input_3_2_f_4_1, 3, 1)
        x4_inversewavel_4_1 = self.inversehar(x4_local_output_4_1, H_3_2_f_4_1, V_3_2_f_4_1, D_3_2_f_4_1)
        x3_local_output_3_2 = self.local_encoder3_2(torch.cat([x3_local_output_3_1, A3_2_onec, x4_inversewavel_4_1],1))

    ##构建1_3 来自 1_1 1_2 2_2
        x1_global_output_1_3 = self.global_encoder1_3(torch.cat([x1_global_output_1_1, x1_global_output_1_2,self.up(x2_global_output_2_2)],1))
        x1_global_input_1_3_f_2_2 = self.global_channel1_3(x2_global_output_2_2)
        H_1_3_f_2_2, V_1_3_f_2_2, D_1_3_f_2_2 = torch.chunk(x1_global_input_1_3_f_2_2, 3, 1)
        x2_inversewavel_2_2 = self.inversehar(x2_local_output_2_2, H_1_3_f_2_2, V_1_3_f_2_2, D_1_3_f_2_2)
        x1_local_output_1_3 = self.local_encoder1_3(torch.cat([x1_local_output_1_1, x1_local_output_1_2, x2_inversewavel_2_2],1))

    ##构建2_3 来自2_1 2_2 1_3 3_2##调整了通道
        A2_3_onec, H2_3onec, V2_3onec, D2_3onec = self.har(x1_local_output_1_3)
        x2_global_2_3_threec = torch.cat((H2_3onec, V2_3onec, D2_3onec), dim=1)
        x2_global_output_2_3 = self.global_encoder2_3(torch.cat([x2_global_output_2_1, x2_global_output_2_2, x2_global_2_3_threec, self.up(x3_global_output_3_2)],1))

        x1_global_input_2_3_f_3_2 = self.global_channel2_3(x3_global_output_3_2)
        H_2_3_f_3_2, V_2_3_f_3_2, D_2_3_f_3_2 = torch.chunk(x1_global_input_2_3_f_3_2, 3, 1)
        x3_inversewavel_3_2 = self.inversehar(x3_local_output_3_2, H_2_3_f_3_2, V_2_3_f_3_2, D_2_3_f_3_2)
        x2_local_output_2_3 = self.local_encoder2_3(torch.cat([x2_local_output_2_1, x2_local_output_2_2, A2_3_onec, x3_inversewavel_3_2],1))

    ##构建1_4 来自 1_1 1_2 1_3 2_3
        x1_global_output_1_4 = self.global_encoder1_4(torch.cat([x1_global_output_1_1, x1_global_output_1_2, x1_global_output_1_3, self.up(x2_global_output_2_3)],1))
        x1_global_input_1_4_f_2_3 = self.global_channel1_4(x2_global_output_2_3)
        H_1_4_f_2_3, V_1_4_f_2_3, D_1_4_f_2_3 = torch.chunk(x1_global_input_1_4_f_2_3, 3, 1)
        x2_inversewavel_2_3 = self.inversehar(x2_local_output_2_3, H_1_4_f_2_3, V_1_4_f_2_3, D_1_4_f_2_3)
        x1_local_output_1_4 = self.local_encoder1_4(torch.cat([x1_local_output_1_1, x1_local_output_1_2, x1_local_output_1_3,x2_inversewavel_2_3],1))


        f_input = x_inut
        f1 = x1_global_output_1_4
        f2 = x2_global_output_2_3
        f3 = x3_global_output_3_2
        #  CCT

        finput_A,finput_H,finput_V,finput_D = self.har(x_inut)
        finput_AA, finput_HH, finput_VV, finput_DD = self.har(finput_A)
        finput_att = self.wave_att_input_t(finput_AA, finput_HH, finput_VV, finput_DD)
        finput_HHVVDD = self.stand_cahnnel_input(finput_att).flatten(2).permute(0, 2, 1) #64 64=4096

                #f_1
        f1_A,f1_H,f1_V,f1_D = self.har(x1_global_output_1_4)
        f1_att = self.wave_att_f1(f1_A,f1_H,f1_V,f1_D)  ##这个得出的注意力机制，实际上是对A这个低频分量的注意力机制
        f1_HVD = self.stand_cahnnel1(f1_att).flatten(2).permute(0, 2, 1)#64 64=4096
                #f_2
        f2_A,f2_H,f2_V,f2_D = self.har(x2_global_output_2_3)
        f2_att = self.wave_att_f2(f2_A,f2_H,f2_V,f2_D)
        f2_HVD = self.stand_cahnnel2(f2_att).flatten(2).permute(0, 2, 1) #32 32=1024
                #f_3
        f3_A,f3_H,f3_V,f3_D = self.har(x3_global_output_3_2)
        f3_att = self.wave_att_f3(f3_A,f3_H,f3_V,f3_D)
        f3_HVD = self.stand_cahnnel3(f3_att).flatten(2).permute(0, 2, 1)#16 16=256

        ###得到进入trans之前的图像尺寸
        binput, cinput, hinput, winput = finput_att.shape
        b1, c1, h1, w1 = f1_att.shape
        b2, c2, h2, w2 = f2_att.shape
        b3, c3, h3, w3 = f3_att.shape


        ##构造我需要的图像尺寸进入Transformer##

        ##trans的处理##
        f3_HVDe = self.TransTo3e(f3_HVD, torch.cat((finput_HHVVDD, f1_HVD, f2_HVD), dim=1)) #256 9216
        f2_HVDe = self.TransTo2e(f2_HVD, torch.cat((finput_HHVVDD, f1_HVD, f3_HVDe), dim=1))#1024 8448
        f1_HVDe = self.TransTo1e(f1_HVD, torch.cat((finput_HHVVDD, f2_HVDe, f3_HVDe), dim=1))  ##里面加个代码，最后返回的得是图像而不是序列  4096  5376
        finput_HHVVDDe = self.TransTo_input(finput_HHVVDD, torch.cat((f1_HVDe, f2_HVDe, f3_HVDe), dim=1))  ##4096  5376
        ##trans的处理##

        ##从序列变为之前的图像形状##

        f3_HVDe = rearrange(f3_HVDe, 'b (h w) c -> b c h w', h=h3, w=w3)

        f2_HVDe = rearrange(f2_HVDe, 'b (h w) c -> b c h w', h=h2, w=w2)

        f1_HVDe = rearrange(f1_HVDe, 'b (h w) c -> b c h w', h=h1, w=w1)

        finput_HHVVDDe = rearrange(finput_HHVVDDe, 'b (h w) c -> b c h w', h=hinput, w=winput)
        ##从序列变为之前的图像形状##

        ##在给返回回去##
                ##f_input
        finput_HHVVDDe = self.wavel_channel_down_to_origin_x_inut(finput_HHVVDDe)
        finput_A = self.inversehar(finput_HHVVDDe, finput_HH, finput_VV, finput_DD)
        x_inut = self.inversehar(finput_A, finput_H, finput_V, finput_D)


                ##f_1
        f1_HVDe = self.wavel_channel_down_to_origin_x1_global_output_1_4(f1_HVDe)
        x1_global_output_1_4 = self.inversehar(f1_HVDe, f1_H, f1_V, f1_D)

                ##f_2
        f2_HVDe = self.wavel_channel_down_to_origin_x2_global_output_2_3(f2_HVDe)
        x2_global_output_2_3 = self.inversehar(f2_HVDe, f2_H, f2_V, f2_D)
                ##f_3
        f3_HVDe = self.wavel_channel_down_to_origin_x3_global_output_3_2(f3_HVDe)
        x3_global_output_3_2 = self.inversehar(f3_HVDe, f3_H, f3_V, f3_D)

        x_inut = x_inut
        x1_global_output_1_4 = x1_global_output_1_4 + f1
        x2_global_output_2_3 = x2_global_output_2_3 + f2
        x3_global_output_3_2 = x3_global_output_3_2 + f3
        ##第四层的小波上采样
        x4_global_output_de = self.decoder4_channel(x4_global_output_4_1)
        split_tensors_4 = torch.chunk(x4_global_output_de, chunks=3, dim=1)
        H4_de, V4_de, D4_de = split_tensors_4
        x4_out = self.out4(x4_local_output_4_1 + H4_de + V4_de + D4_de)

        x3_local_input_de = self.inversehar(x4_local_output_4_1, H4_de, V4_de, D4_de)

        ##第三层的小波上采样
        x3_global_output_de = self.decoder3_channel(x3_global_output_3_2)
        split_tensors_3 = torch.chunk(x3_global_output_de, chunks=3, dim=1)
        H3_de, V3_de, D3_de = split_tensors_3
        x3_local_output_3_2_de = self.decoder3_channel_local(x3_local_output_3_2+x3_local_input_de)
        x3_out = self.out3(x3_local_output_3_2_de+H3_de+V3_de+D3_de)

        x2_local_input_de = self.inversehar(x3_local_output_3_2_de, H3_de, V3_de, D3_de)

        ##第二层的小波上采样
        x2_global_output_de = self.decoder2_channel(x2_global_output_2_3)
        split_tensors_2 = torch.chunk(x2_global_output_de, chunks=3, dim=1)
        H2_de, V2_de, D2_de = split_tensors_2
        x2_local_output_2_3_de = self.decoder2_channel_local(x2_local_output_2_3+x2_local_input_de)
        x2_out = self.out2(x2_local_output_2_3_de + H2_de + V2_de + D2_de)

        x1_local_input_de = self.inversehar(x2_local_output_2_3_de, H2_de, V2_de, D2_de)

        ##第一层的小波上采样
        x1_global_output_de = self.decoder1_channel(x1_global_output_1_4)
        split_tensors_1 = torch.chunk(x1_global_output_de, chunks=3, dim=1)
        H1_de, V1_de, D1_de = split_tensors_1
        x1_local_output_1_4_de = self.decoder1_channel_local(x1_local_output_1_4+x1_local_input_de)
        x1_out = self.out1(x1_local_output_1_4_de + H1_de + V1_de + D1_de)

        x1_local_final_raw_de = self.inversehar(x1_local_output_1_4_de, H1_de, V1_de, D1_de)
        out = self.outc_global(self.from_input2out(x1_local_final_raw_de + x_inut))

        # deep supervision
        if self.deepsuper:
            gt_5 = self.gt_conv5(x4_out)
            gt_4 = self.gt_conv4(x3_out)
            gt_3 = self.gt_conv3(x2_out)
            gt_2 = self.gt_conv2(x1_out)
            # 原始深监督
            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out))
            else:
                return torch.sigmoid(out)
        else:
            print("不进入这里")
            return torch.sigmoid(out)


if __name__ == '__main__':
    config_vit = get_DWTFreqNet_config()
    model = DWTFreqNet(config_vit, mode='test', deepsuper=True)
    model = model
    inputs = torch.rand(1, 1, 256, 256)
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
