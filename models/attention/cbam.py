import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d


class ChannelAttention(nn.Module):
    def __init__(self, c_in, reduction=8):
        """
        This module will calculate the channel attention. The shared network is composed of multi-layer perceptron (
        MLP) with one hidden layer. To reduce parameter overhead, the hidden activation size is set to C/r×1×1,
        where r is the reduction ratio Args: c_in: number of input channels reduction: reduction factor/ratio
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False)
        )

    def forward(self, f):
        """
        Using adaptive_max_pool2d(1) is equivalent to using max_pool2d(kernel_size=(H, W)) as both will generate the
        output of size C × 1 × 1
        Args:
            f: feature map to apply channel attention on

        Returns: Channel attention output
        """
        n, c, h, w = f.shape
        # Applying Max Pool on feature vector
        f_max = adaptive_max_pool2d(input=f, output_size=1)
        # Applying Average Pool on feature vector
        f_avg = adaptive_avg_pool2d(input=f, output_size=1)

        # Changing shape of f_max and f_avg from (N × C × 1 × 1) to (N, C) to pass through linear layers
        f_max, f_avg = f_max.view(n, c), f_avg.view(n, c)

        # Passing max pool through shared MLP
        f_max = self.mlp(f_max)
        # Passing average pool through shared MLP
        f_avg = self.mlp(f_avg)

        # Concatenate max_pool and avg_pool
        concatenated_pool = f_max + f_avg

        # Pass concatenated_pool through sigmoid function
        out = torch.sigmoid(concatenated_pool)

        # Change the shape of out from (N, C) to (N, C, 1, 1)
        out = out.view(n, c, 1, 1)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        This module will calculate the spatial attention. based on below equation:
         Ms(F)  = sigmoid(f([F_avg; F_max]))
        f represents a convolution operation with the filter size of kernel_size.
        Args:
            kernel_size: kernel size of convolution operation
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=int((kernel_size - 1) / 2))

    def forward(self, f):
        """
        Using adaptive_max_pool2d(1) is equivalent to using max_pool2d(kernel_size=(H, W)) as both will generate the
        output of size 1 × H × W

        Args:
            f: feature map to apply spatial attention on

        Returns: Spatial attention output
        """

        # Applying Max Pool on feature vector
        f_max, _ = torch.max(f, dim=1, keepdim=True)
        # Applying Average Pool on feature vector
        f_avg = torch.mean(f, dim=1, keepdim=True)

        # Concat the max_pool and avg_pool
        concatenated_pool = torch.cat([f_max, f_avg], dim=1)

        # Pass concatenated pool via convolution layer
        out = self.conv(concatenated_pool)

        # Pass concatenated_pool through sigmoid function
        out = torch.sigmoid(out)

        return out


class CBAM(nn.Module):
    def __init__(self, c_in, kernel_size, reduction=8):
        super().__init__()
        self.channel_attention = ChannelAttention(c_in=c_in, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, f):
        """
        Given an intermediate feature map F ∈ R(C×H×W) as input, CBAM sequentially
        infers a 1D channel attention map Mc ∈ R(C×1×1) and a 2D spatial attention map Ms ∈ R(1×H×W)
        This method calculates the channel attention followed by spatial attention over the input x.
        The attentions are calculated per below equations.
           f' = Mc(F) ⊗ f
           f" = Ms(F') ⊗ f'
        """
        channel_attention = self.channel_attention(f)  # channel_attention here equivalent to F'
        _f = channel_attention * f
        spatial_attention = self.spatial_attention(_f)  # spatial_attention here equivalent to F"
        __f = spatial_attention * _f
        return __f
