# Entire file copied + adapted from source: https://raw.githubusercontent.com/pytorch/vision/refs/heads/main/torchvision/models/resnet.py
# we need some adaptions of the basic block (see comment below) to reproduce the results

import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
import torch
from models.cifar10resnet import conv3x3

class OriginalBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # This has to be changed to support Option A from the paper
        # Original:
        #    if self.downsample is not None:
        #      identity = self.downsample(x)
        #      out += identity
        # Quote:
        #     "we consider two options: (A) The shortcut still
        #     performs identity mapping, with extra zero entries padded
        #     for increasing dimensions. This option introduces no extra
        #     parameter [...] For both options, when the shortcuts go across feature maps,
        #     they are performed with a stride of 2."
        #
        # Note: For now I think Option (A) is closer to the paper results.
        #
        # Adapted from: https://github.com/a-martyn/resnet/blob/master/resnet.py#L80
        if x.shape != out.shape:
            downsampled = self.downsample(x)
            padded = torch.zeros_like(downsampled)
            out = out + torch.cat((downsampled, padded), dim=1)
        else:
            out = out + x

        out = self.relu2(out)

        return out

