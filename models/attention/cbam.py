from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, c_in, reduction=8):
        """
        This module will calculate the channel attention.
        Args:
            c_in: number of input channels
            reduction: reduction factor/ratio
        """

        # Using nn.AdaptiveMaxPool2d(1) is equivalent to using nn.MaxPool(kernel_size=(H, W)) as both will generate the
        # output of size C × 1 × 1
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # The shared network is composed of multi-layer perceptron (MLP) with one hidden layer. To reduce
        # parameter overhead, the hidden activation size is set to C/r×1×1, where r is the reduction ratio
        self.mlp = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Applying Max Pool on feature vector
        max_pool = self.max_pool(x)
        # Applying Average Pool on feature vector
        avg_pool = self.avg_pool(x)

        # Passing max pool through shared MLP
        max_pool = self.mlp(max_pool)
        # Passing average pool through shared MLP
        avg_pool = self.mlp(avg_pool)

        # Concatenate max_pool and avg_pool
        concatenated_pool = max_pool + avg_pool

        # Pass concatenated_pool through sigmoid function
        return self.sigmoid(concatenated_pool)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        pass

    def forward(self, x):
        pass


class CBAM(nn.Module):
    def __init__(self, c_in, kernel_size, reduction=8):
        self.channel_attention = ChannelAttention(c_in=c_in, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, F):
        """
        Given an intermediate feature map F ∈ R(C×H×W) as input, CBAM sequentially
        infers a 1D channel attention map Mc ∈ R(C×1×1) and a 2D spatial attention map Ms ∈ R(1×H×W)
        This method calculates the channel attention followed by spatial attention over the input x.
        The attentions are calculated per below equations.
           F' = Mc(F) ⊗ F
           F" = Ms(F') ⊗ F'
        """
        out = self.channel_attention(F)     # out here equivalent to F'
        out = self.spatial_attention(out)   # out here equivalent to F"

        return out