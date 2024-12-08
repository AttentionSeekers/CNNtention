#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-01 10:23:46 Sunday

@author: Nikhil Kapila
"""

# Entire file copied + adapted from source: https://raw.githubusercontent.com/pytorch/vision/refs/heads/main/torchvision/models/resnet.py
from typing import List, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck

from models.attention.cbam import CBAM
from models.original_basic_block import OriginalBasicBlock


class ResnetCBAM(nn.Module):
    def __init__(
            self,
            block: Type[Union[OriginalBasicBlock]],
            layers: List[int],
            num_classes: int = 10, 
            reduction: int = 8,
    ) -> None:
        super().__init__()
        # We adjust the entry convolution, therefore inplanes (number of input channels) needs to be adjusted too
        # Original: self.inplanes = 64
        self.inplanes = 16

        # Replace ImageNet entry convolution with CIFAR10 entry convolution
        # Original: self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # Quote: "The first layer is 3x3 convolutions"
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # Adapt to change of entry convolution above
        # Original: self.bn1 = norm_layer(self.inplanes)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()

        # Quote:
        # "The number of filters are {16, 32, 64} respectively".

        # NK: augmented self attention blocks between resnet blocks
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.att1 = CBAM(c_in=16, reduction=reduction, kernel_size=7)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.att2 = CBAM(c_in=32, reduction=reduction, kernel_size=7)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.att3 = CBAM(c_in=64, reduction=reduction, kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Adjust final layer
        # Original: self.fc = nn.Linear(512 * block.expansion, num_classes)
        # Quote: "The network ends with a global average pooling (above), a 10-way fully-connected layer and softmax."
        # I think softmax is not required here and will be done by loss
        self.fc = nn.Linear(64, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
    ) -> nn.Sequential:

        layers = [block(
            self.inplanes, planes, stride  #, downsample, self.groups, self.base_width, previous_dilation, norm_layer
        )]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Block 1
        x = self.layer1(x)
        x = self.att1(x)

        # Block 2
        x = self.layer2(x)
        x = self.att2(x)

        # Block 3
        x = self.layer3(x)
        x = self.att3(x)

        # Avg Pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Linear classifier
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
