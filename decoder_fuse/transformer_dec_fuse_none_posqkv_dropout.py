# -*- coding: utf-8 -*-
"""

@software: PyCharm
@file: transformer.py
@time: 2021/12/8 01:00
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import math
import time
from decoder_fuse.transformer_block import get_sinusoid_encoding
from thop import profile, clever_format
from einops import rearrange, repeat

class EfficientAttention(nn.Module): # this is multiAttention
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.softmax = nn.Softmax(dim=-1)
        self.keys = nn.Linear(self.in_channels, self.key_channels)
        self.queries = nn.Linear(self.in_channels, self.key_channels)
        self.values = nn.Linear(self.in_channels, self.value_channels)
        self.reprojection = nn.Linear(self.value_channels, self.in_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_):
        B,N,C = input_.size()
        assert C == self.in_channels,"C {} != inchannels {}".format(C, self.in_channels)
        # assert input_.shape[1:] == x_pos_embed.shape[1:], "x.shape {} != x_pos_embed.shape {}".format(input_.shape, x_pos_embed.shape)
        keys = self.keys(input_) #.reshape((n, self.key_channels, h * w))
        queries = self.queries(input_) #.reshape(n, self.key_channels, h * w)
        values = self.values(input_)#.reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = keys[
                            :,
                           : ,
                  i * head_key_channels
                  :(i + 1) * head_key_channels
                            ]
            query = queries[
                              :,
                             : ,
                    i * head_key_channels
                    :(i + 1) * head_key_channels
                              ]
            value = values[
                    :,
                    : ,
                    i * head_value_channels:(i + 1) * head_value_channels
                    ]
            # context = key @ value.transpose(1, 2)
            # query = torch.nn.functional.normalize(query, dim=-1)
            # key = torch.nn.functional.normalize(key, dim=-1)
            context = query @ key.transpose(1,2) / math.sqrt(N)
            context = self.softmax(context)
            attended_value = context @ value

            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=2)
        # aggregated_values = aggregated_values.transpose(1,2)
        reprojected_value = self.reprojection(aggregated_values)
        reprojected_value = self.dropout(F.relu(reprojected_value))

        return reprojected_value

class Multi_EfficientAttention(nn.Module): # this is multiAttention
    def __init__(self, x_channels, y_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.queries = nn.Linear(self.x_channels, self.key_channels)
        self.keys = nn.Linear(self.y_channels, self.key_channels)
        self.values = nn.Linear(self.y_channels, self.value_channels)
        self.reprojection = nn.Linear(self.value_channels, self.x_channels)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, y):
        Bx,Nx,Cx = x.size()
        assert Cx == self.x_channels,"Cx {} != inchannels {}".format(Cx, self.x_channels)
        # assert x.shape[1:] == x_pos_embed.shape[1:], "x.shape {} != x_pos_embed.shape {}".format(x.shape, x_pos_embed.shape)
        By, Ny, Cy = y.size()
        assert Cy == self.y_channels, "Cy {} != inchannels {}".format(Cy, self.y_channels)
        # assert y.shape[1:] == y_pos_embed.shape[1:], "y.shape {} != y_pos_embed.shape {}".format(y.shape, y_pos_embed.shape)


        queries = self.queries(x) #.reshape(n, self.key_channels, h * w)
        keys = self.keys(y)  # .reshape((n, self.key_channels, h * w))
        values = self.values(y)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = keys[
                            :,
                            : ,
                            i * head_key_channels:(i + 1) * head_key_channels
                            ]
            query = queries[
                              :,
                              : ,
                              i * head_key_channels:(i + 1) * head_key_channels
                              ]
            value = values[
                    :,
                   :,
                    i * head_value_channels
                    : (i + 1) * head_value_channels
                    ]

            # context = key @ value.transpose(1, 2)
            context = query @ key.transpose(1, 2) / math.sqrt(Ny)
            context = self.softmax(context)

            attended_value = context @ value
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=2)
        # aggregated_values = aggregated_values.transpose(1, 2)
        reprojected_value = self.reprojection(aggregated_values)
        reprojected_value = self.dropout(F.relu(reprojected_value))

        return reprojected_value

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,x_channels, nx, y_channels, ny):
        super(Block, self).__init__()
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.nx = nx
        self.ny = ny
        self.x_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=nx, d_hid=x_channels), requires_grad=False)
        self.y_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=ny, d_hid=y_channels), requires_grad=False)
        self.x2_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=nx, d_hid=x_channels),requires_grad=False)
        self.norm_layer = nn.LayerNorm(x_channels)
        # nn.LayerNorm

        self.self_attn = EfficientAttention(x_channels, x_channels, 4, x_channels)
        self.cross_attn = Multi_EfficientAttention(x_channels=x_channels, y_channels=y_channels, key_channels=x_channels, head_count=4, value_channels=x_channels)
        self.mlp = Mlp(in_features=x_channels, hidden_features=x_channels * 4,out_features= x_channels)
    def forward(self,x, y):
        x_atten = self.self_attn(x)
        Osa = self.norm_layer(x + x_atten)
        xy_attn = self.cross_attn(Osa, y)
        Oca = self.norm_layer(xy_attn + Osa)
        Of = self.mlp(Oca)
        Oo = self.norm_layer(Of + Oca)
        return Oo

class TransFuseModel(nn.Module):
    def __init__(self, num_blocks, x_channels, nx, y_channels, ny):
        super(TransFuseModel, self).__init__()
        assert x_channels == y_channels, "channel_X-{} should same as channel_Y-{}".format(x_channels, y_channels)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            # todo
            Block(x_channels=x_channels, nx=nx, y_channels=y_channels, ny=ny)for i in range(self.num_blocks)
        ])
        self.norm =nn.LayerNorm(x_channels)

    def forward(self,x,y):
        '''
        :param x: shape B,Nx,C
        :param y: shape B,Ny,C
        :return: shape B,Nx,c
        '''
        # Bx, Nx, Cx = x.shape
        # By, Ny, Cy = y.shape
        for block in self.blocks:
            x = block(x, y)
        x = self.norm(x)
        return x

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

        # Initialize the output tensor
        out_LL = torch.zeros((B, C, H // 2, W // 2), device=device)  # Approximation (LL)
        out_LH = torch.zeros((B, C, H // 2, W // 2), device=device)  # Horizontal (LH)
        out_HL = torch.zeros((B, C, H // 2, W // 2), device=device)  # Vertical (HL)
        out_HH = torch.zeros((B, C, H // 2, W // 2), device=device)  # Diagonal (HH)

        # Apply Haar wavelet transform to each channel independently
        for c in range(C):
            for b in range(B):
                out_LL[b, c, :, :] = F.conv2d(x[b, c, :, :].unsqueeze(0).unsqueeze(0), haar_matrix_LL, stride=2,
                                              padding=0).squeeze(0).squeeze(0)
                out_LH[b, c, :, :] = F.conv2d(x[b, c, :, :].unsqueeze(0).unsqueeze(0), haar_matrix_LH, stride=2,
                                              padding=0).squeeze(0).squeeze(0)
                out_HL[b, c, :, :] = F.conv2d(x[b, c, :, :].unsqueeze(0).unsqueeze(0), haar_matrix_HL, stride=2,
                                              padding=0).squeeze(0).squeeze(0)
                out_HH[b, c, :, :] = F.conv2d(x[b, c, :, :].unsqueeze(0).unsqueeze(0), haar_matrix_HH, stride=2,
                                              padding=0).squeeze(0).squeeze(0)

        return out_LL, out_LH, out_HL, out_HH

#########
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

class TransFuseModel_IR(nn.Module):
    def __init__(self, num_blocks, x_channels, nx, y_channels, ny):
        super(TransFuseModel, self).__init__()
        assert x_channels == y_channels, "channel_X-{} should same as channel_Y-{}".format(x_channels, y_channels)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            # todo
            Block(x_channels=x_channels, nx=nx, y_channels=y_channels, ny=ny) for i in range(self.num_blocks)
        ])
        self.norm = nn.LayerNorm(x_channels)
        self.wavel_channel_input = nn.Conv2d(x_channels, 1)
        self.har = HaarWaveletTransform()
        self.inversehar = InverseHaarWaveletTransform()

    def forward(self, x, y):
        '''
        :param x: shape B,Nx,C
        :param y: shape B,Ny,C
        :return: shape B,Nx,c
        '''
        # Bx, Nx, Cx = x.shape
        # By, Ny, Cy = y.shape
        x = self.wavel_channel_input(x)
        A1_1, H1_1, V1_1, D1_1 = self.har(x)





        for block in self.blocks:
            x = block(x, y)
        x = self.norm(x)
        return x
