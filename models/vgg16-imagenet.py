#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-11-16 18:58:09 Saturday

@author: Nikhil Kapila
"""

import torch
import torch.nn as nn

# vgg16 info: https://www.geeksforgeeks.org/vgg-16-cnn-model/
# imagenet inputs are of size 224x224x3.

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()

        self.conv_layers =\
        nn.Sequential(
            # output size tracking in comments
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # 64x224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # 64x224x224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 64x112x112
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # 128x112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # 128x112x112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 128x56x56
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # 256x56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 256x56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 256x56x56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 256x28x28
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # 512x28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # 512x28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # 512x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 512x14x14
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # 512x14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # 512x14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # 512x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 512x7x7
        )
        # latent variable representation becomes of size 512x7x7 = 25,088
        # after 13 conv blocks

        self.linear =\
        nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        latent_variables = self.conv_layers(x)
        out = self.linear(latent_variables)
        return out