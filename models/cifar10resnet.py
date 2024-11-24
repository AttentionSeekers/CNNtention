# Entire file copied + adapted from source: https://raw.githubusercontent.com/pytorch/vision/refs/heads/main/torchvision/models/resnet.py
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        #groups=groups,
        bias=False,
        #dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Cifar10ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        # not required: replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        # We adjust the entry convolution, therefore inplanes (number of input channels) needs to be adjusted too
        # Original: self.inplanes = 64
        self.inplanes = 16

        # This is not required (only leaving the dilation init to not make it undefined
        #
        self.dilation = 1
        # if replace_stride_with_dilation is None:
        #     # each element in the tuple indicates if we should replace
        #     # the 2x2 stride with a dilated convolution instead
        #     replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
        #     raise ValueError(
        #         "replace_stride_with_dilation should be None "
        #         f"or a 3-element tuple, got {replace_stride_with_dilation}"
        #     )

        self.groups = groups
        self.base_width = width_per_group

        # Replace ImageNet entry convolution with CIFAR10 entry convolution
        # Original: self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # Quote: "The first layer is 3x3 convolutions"
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)


        # Adapt to change of entry convolution above
        # Original: self.bn1 = norm_layer(self.inplanes)
        self.bn1 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)

        # Remove maxpooling
        # Original: self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # This is not mentioned in the paper


        # Adjust layers as specified in the paper
        # Original:
        #     self.layer1 = self._make_layer(block, 64, layers[0])
        #     self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        #     self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        #     self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # Quote:
        # "The number of filters are {16, 32, 64} respectively".

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Adjust final layer
        # Original: self.fc = nn.Linear(512 * block.expansion, num_classes)
        # Quote: "The network ends with a global average pooling (above), a 10-way fully-connected layer and softmax."
        # I think softmax is not required here and will be done by loss
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # This is not required
        #
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #    for m in self.modules():
        #        if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #            nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #        elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #            nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        # dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        # Not required as we do not use dilation (only leaving the dilation init)
        previous_dilation = self.dilation
        # if dilate:
        #     self.dilation *= stride
        #    stride = 1

        if stride != 1 or self.inplanes != planes: #  * block.expansion (not required as always == 1)
            # This changed is there to match Option A for shortcuts from the paper
            # Original:
            #    downsample = nn.Sequential(
            #        conv1x1(self.inplanes, planes * block.expansion, stride),
            #        norm_layer(planes * block.expansion),
            #    )
            #
            # Quote: "For both options, when the shortcuts go across feature maps of two sizes,
            # they are performed with a stride of 2."
            #
            # Note that average pooling is not explicity mentioned, but after there were still deviations
            # with the first implementation, I checked what they do
            # here and copied it: https://github.com/a-martyn/resnet/blob/master/resnet.py#L77
            downsample = nn.AvgPool2d(kernel_size=1, stride=2)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes # * block.expansion (not required as always == 1)
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Remove maxpooling (as above)
        # Original: self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Remove layer 4 (as above)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)