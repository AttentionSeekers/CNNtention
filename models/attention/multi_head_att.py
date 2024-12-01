#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-01 22:28:35 Sunday

@author: Nikhil Kapila
"""

import torch
from torch import nn

class MultiHeadSelfAtt(nn.Module):
    def __init__(self, in_ch, num_heads=8): # typically as 8 basing it off Att is All You Need Paper
        super(MultiHeadSelfAtt, self).__init__()

        # should be able to divide input into different heads
        if in_ch % num_heads != 0: raise ValueError('Cannot equally divide input to different heads.')

        self.num_heads = num_heads
        self.channels_per_head = in_ch // num_heads

        self.query = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.key = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.val = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        self.linear = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.weights = nn.Parameter(torch.zeros(1))

    def forward(self, x, intermediate=False):
        N, ch, h, w = x.shape

        query = self.query(x).view(N, self.num_heads, self.channels_per_head, h*w) # N, N_h, C_h, h*w
        key = self.key(x).view(N, self.num_heads, self.channels_per_head, h*w) # N, N_h, C_h, h*w
        val = self.val(x).view(N, self.num_heads, self.channels_per_head, h*w) # N, N_h, C_h, h*w

        att = nn.Softmax(
            (key.mT@query)*self.channels_per_head**-0.5 # N, N_h, h*w, h*w
        )

        out = att@val.mT #N, N_h, h*w, h*w x N, N_h, C_h, h*w --> N, N_h, h*w, C_h
        out = out.mT # N, N_h, C_h, h*w
        out = out.view(N, ch, h, w)

        out = self.linear(out)
        out = self.weights*out + x

        if intermediate: return out, att, query, key
        else: return out


        