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


class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14 * 14 = 196

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x

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

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    # def forward(self, x, h, w):
    def forward(self, x):
        if x is None:
            return None

        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


# spatial-embedded Single-head Channel-cross Attention (SSCA)
class Attention_org(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.mhead1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, channel_num[0] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[0] * self.num_attention_heads // 2, bias=False)
        self.q2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, channel_num[1] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[1] * self.num_attention_heads // 2, bias=False)
        self.q3 = nn.Conv2d(channel_num[2] * self.num_attention_heads, channel_num[2] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[2] * self.num_attention_heads // 2, bias=False)
        self.q4 = nn.Conv2d(channel_num[3] * self.num_attention_heads, channel_num[3] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[3] * self.num_attention_heads // 2, bias=False)
        self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)

        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
        self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)


        # ****************** useless ***************************************
        self.q1_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q1_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q1_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q1_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q2_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q2_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q2_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q2_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q3_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q3_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q3_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q3_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q4_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q4_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q4_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q4_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        b, c, h, w = emb1.shape
        q1 = self.q1(self.mhead1(emb1))
        q2 = self.q2(self.mhead2(emb2))
        q3 = self.q3(self.mhead3(emb3))
        q4 = self.q4(self.mhead4(emb4))
        k = self.k(self.mheadk(emb_all))
        v = self.v(self.mheadv(emb_all))
        # k, v = kv.chunk(2, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q3 = rearrange(q3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q4 = rearrange(q4, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        q3 = torch.nn.functional.normalize(q3, dim=-1)
        q4 = torch.nn.functional.normalize(q4, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, c1, _ = q1.shape
        _, _, c2, _ = q2.shape
        _, _, c3, _ = q3.shape
        _, _, c4, _ = q4.shape
        _, _, c, _ = k.shape

        attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn2 = (q2 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn3 = (q3 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn4 = (q4 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attn1))
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))
        attention_probs4 = self.softmax(self.psi(attn4))

        out1 = (attention_probs1 @ v)
        out2 = (attention_probs2 @ v)
        out3 = (attention_probs3 @ v)
        out4 = (attention_probs4 @ v)

        out_1 = out1.mean(dim=1)
        out_2 = out2.mean(dim=1)
        out_3 = out3.mean(dim=1)
        out_4 = out4.mean(dim=1)

        out_1 = rearrange(out_1, 'b  c (h w) -> b c h w', h=h, w=w)
        out_2 = rearrange(out_2, 'b  c (h w) -> b c h w', h=h, w=w)
        out_3 = rearrange(out_3, 'b  c (h w) -> b c h w', h=h, w=w)
        out_4 = rearrange(out_4, 'b  c (h w) -> b c h w', h=h, w=w)

        O1 = self.project_out1(out_1)
        O2 = self.project_out2(out_2)
        O3 = self.project_out3(out_3)
        O4 = self.project_out4(out_4)
        weights = None

        return O1, O2, O3, O4, weights


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = eca_layer_2d(dim)

    def forward(self, x):
        x_3,x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x = torch.cat([x1_3, x1_5], dim=1)
        x = self.project_out(x)
        x = self.eca(x)
        return x


#  Spatial-channel Cross Transformer Block (SCTB)
class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')

        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)


    def forward(self, emb1, emb2, emb3, emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        emb_all = torch.cat(embcat, dim=1)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)  # 1 196 960
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = x
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


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
        self.pool = nn.MaxPool2d(2, 2)
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


        ##Dense全局结构改变通道

        ##Dense改变通道，减轻小波的计算量，先下采样进入，再上采样回去
        self.wavel_channel2_2_input = nn.Conv2d(in_channels * 2, 1, kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel3_2_input = nn.Conv2d(in_channels * 4, 1, kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel2_3_input = nn.Conv2d(in_channels * 2, 1, kernel_size=(1, 1), stride=(1, 1))
        ##Dense改变通道，减轻小波的计算量，先下采样进入，再上采样回去


        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
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

        #  CCT
        f_input = x_inut
        f1 = x1_global_output_1_4
        f2 = x2_global_output_2_3
        f3 = x3_global_output_3_2
        #  CCT
###我构造的transfoemer shishan代码###
        ##构造我需要的图像尺寸进入Transformer##
                #x_input
        # x_inut = self.wavel_channel_down_x_inut(x_inut)
        finput_A,finput_H,finput_V,finput_D = self.har(x_inut)
        finput_AA, finput_HH, finput_VV, finput_DD = self.har(finput_A)
        finput_att = self.wave_att_input_t(finput_AA, finput_HH, finput_VV, finput_DD)
        # finput_HHVVDD = torch.cat((finput_HH,finput_VV,finput_DD), dim=1)
        finput_HHVVDD = self.stand_cahnnel_input(finput_att).flatten(2).permute(0, 2, 1) #64 64=4096

                #f_1
        # x1_global_output_1_4 = self.wavel_channel_down_x1_global_output_1_4(x1_global_output_1_4)
        f1_A,f1_H,f1_V,f1_D = self.har(x1_global_output_1_4)
        f1_att = self.wave_att_f1(f1_A,f1_H,f1_V,f1_D)  ##这个得出的注意力机制，实际上是对A这个低频分量的注意力机制
        # f1_HVD = torch.cat((f1_H, f1_V, f1_D), dim=1)
        f1_HVD = self.stand_cahnnel1(f1_att).flatten(2).permute(0, 2, 1)#64 64=4096
                #f_2
        #x2_global_output_2_3 = self.wavel_channel_down_x2_global_output_2_3(x2_global_output_2_3)
        f2_A,f2_H,f2_V,f2_D = self.har(x2_global_output_2_3)
        f2_att = self.wave_att_f2(f2_A,f2_H,f2_V,f2_D)
        # f2_HVD = torch.cat((f2_H, f2_V, f2_D), dim=1)
        f2_HVD = self.stand_cahnnel2(f2_att).flatten(2).permute(0, 2, 1) #32 32=1024
                #f_3
        # x3_global_output_3_2 = self.wavel_channel_down_x3_global_output_3_2(x3_global_output_3_2)
        f3_A,f3_H,f3_V,f3_D = self.har(x3_global_output_3_2)
        f3_att = self.wave_att_f3(f3_A,f3_H,f3_V,f3_D)
        # f3_HVD = torch.cat((f3_H, f3_V, f3_D), dim=1)
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

        # B2,N2,C2 = f2_HVDe.shape

        f2_HVDe = rearrange(f2_HVDe, 'b (h w) c -> b c h w', h=h2, w=w2)

        # B1,N1,C1 = f1_HVDe.shape

        f1_HVDe = rearrange(f1_HVDe, 'b (h w) c -> b c h w', h=h1, w=w1)

        # BINPUT,NINPUT,CINPUT = finput_HHVVDDe.shape

        finput_HHVVDDe = rearrange(finput_HHVVDDe, 'b (h w) c -> b c h w', h=hinput, w=winput)
        ##从序列变为之前的图像形状##

        ##在给返回回去##
                ##f_input
        finput_HHVVDDe = self.wavel_channel_down_to_origin_x_inut(finput_HHVVDDe)
        # finput_HHe, finput_VVe, finput_DDe = torch.chunk(finput_HHVVDDe,3,1)
        finput_A = self.inversehar(finput_HHVVDDe, finput_HH, finput_VV, finput_DD)
        x_inut = self.inversehar(finput_A, finput_H, finput_V, finput_D)
        # x_inut = self.wavel_channel_up_x_inut(x_inut)##上采样到之前的样子

                ##f_1
        f1_HVDe = self.wavel_channel_down_to_origin_x1_global_output_1_4(f1_HVDe)
        # f1_He, f1_Ve, f1_De = torch.chunk(f1_HVDe,3,1)
        x1_global_output_1_4 = self.inversehar(f1_HVDe, f1_H, f1_V, f1_D)
        # x1_global_output_1_4 = self.wavel_channel_up_x1_global_output_1_4(x1_global_output_1_4)##上采样到之前的样子

                ##f_2
        f2_HVDe = self.wavel_channel_down_to_origin_x2_global_output_2_3(f2_HVDe)
        # f2_He, f2_Ve, f2_De = torch.chunk(f2_HVDe,3,1)
        x2_global_output_2_3 = self.inversehar(f2_HVDe, f2_H, f2_V, f2_D)
        # x2_global_output_2_3 = self.wavel_channel_up_x2_global_output_2_3(x2_global_output_2_3)##上采样到之前的样子

                ##f_3
        f3_HVDe = self.wavel_channel_down_to_origin_x3_global_output_3_2(f3_HVDe)
        # f3_He, f3_Ve, f3_De = torch.chunk(f3_HVDe,3,1)
        x3_global_output_3_2 = self.inversehar(f3_HVDe, f3_H, f3_V, f3_D)
        # x3_global_output_3_2 = self.wavel_channel_up_x3_global_output_3_2(x3_global_output_3_2)##上采样到之前的样子



###我构造的transfoemer shishan代码###

       # x_inut, x1_global_output_1_4, x2_global_output_2_3, x3_global_output_3_2, att_weights = self.mtc(x_inut, x1_global_output_1_4, x2_global_output_2_3, x3_global_output_3_2)
        x_inut = x_inut
        x1_global_output_1_4 = x1_global_output_1_4 + f1
        x2_global_output_2_3 = x2_global_output_2_3 + f2
        x3_global_output_3_2 = x3_global_output_3_2 + f3
        #  Feature fusion上采样融合
        ##第四层的小波上采样
        x4_global_output_de = self.decoder4_channel(x4_global_output_4_1)
        split_tensors_4 = torch.chunk(x4_global_output_de, chunks=3, dim=1)
        H4_de, V4_de, D4_de = split_tensors_4
        x4_out = self.out4(x4_local_output_4_1 + H4_de + V4_de + D4_de)

        x3_local_input_de = self.inversehar(x4_local_output_4_1, H4_de, V4_de, D4_de)
        # d3_local = self.up_decoder4(x3_local_input_de, x3_global_output_3_2)

        ##第三层的小波上采样
        x3_global_output_de = self.decoder3_channel(x3_global_output_3_2)
        split_tensors_3 = torch.chunk(x3_global_output_de, chunks=3, dim=1)
        H3_de, V3_de, D3_de = split_tensors_3
        x3_local_output_3_2_de = self.decoder3_channel_local(x3_local_output_3_2+x3_local_input_de)
        x3_out = self.out3(x3_local_output_3_2_de+H3_de+V3_de+D3_de)


        x2_local_input_de = self.inversehar(x3_local_output_3_2_de, H3_de, V3_de, D3_de)
        # d2_local = self.up_decoder3(x2_local_input_de, x2_global_output_2_3)


        ##第二层的小波上采样
        x2_global_output_de = self.decoder2_channel(x2_global_output_2_3)
        split_tensors_2 = torch.chunk(x2_global_output_de, chunks=3, dim=1)
        H2_de, V2_de, D2_de = split_tensors_2
        x2_local_output_2_3_de = self.decoder2_channel_local(x2_local_output_2_3+x2_local_input_de)
        x2_out = self.out2(x2_local_output_2_3_de + H2_de + V2_de + D2_de)

        x1_local_input_de = self.inversehar(x2_local_output_2_3_de, H2_de, V2_de, D2_de)
        # d1_local = self.up_decoder2(x1_local_input_de, x1_global_output_1_4)


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
