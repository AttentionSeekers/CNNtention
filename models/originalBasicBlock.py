# Entire file copied + adapted from source: https://raw.githubusercontent.com/pytorch/vision/refs/heads/main/torchvision/models/resnet.py
# we need some adaptions of the basic block (see comment below) to reproduce the results
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
class OriginalBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        # downsample: Optional[nn.Module] = None,
        # groups: int = 1,
        # base_width: int = 64,
        # dilation: int = 1,
        # norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # if norm_layer is None:
        #    norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #    raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # if dilation > 1:
        #    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
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
        # Adapted from: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
        if self.inplanes != self.planes or self.stride != 1:
            out += F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes//4, self.planes//4), "constant", 0)
        else:
            out = out + x

        out = self.relu2(out)

        return out

