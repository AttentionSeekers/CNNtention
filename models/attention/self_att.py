#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-11-25 13:45:49 Monday

@author: Nikhil Kapila
"""

import torch
from torch import nn

class SelfAtt(nn.Module):
    def __init__(self, in_ch):
        super(SelfAtt, self).__init__()
        self.weights = nn.Parameter(torch.zeros(1))
        self.key = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.query = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.value = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, intermediate=False): 
        N, ch, h, w = x.shape
        # value = self.value(x).view(N, -1, h*w) # N, ch, h*w
        key = self.key(x).view(N, -1, h*w) # N, ch, h*w
        query = self.query(x).view(N, -1, h*w).permute(0, 2 ,1) # N, h*w, ch

        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        # torch.bmm(input, mat2, *, out=None) → Tensor
        # input - bxnxm and mat2 - bxmxp then out is bxnxp
        
        attention = self.softmax(
            # query is N, h*w, ch
            # key is N, ch, h*w
            # torch.bmm out -> N, h*w, h*w
            # att = smax(torch.bmm)
            torch.bmm(query, key))

        value = self.value(x).view(N, -1, h*w) # N, ch, h*w

        out = torch.bmm(value, # N, ch, h*w
                         attention.permute(0,2,1) # N, h*w, h*w
                         ).view( # output of bmm is N, ch, h*w
                             N, ch, h, w)
        
        # Residual connection
        # self.weights is a trainable param that further weighs contribution of attention 
        out = self.weights*out + x

        # return based on inference
        if intermediate: return out, attention, query, key
        else: return out