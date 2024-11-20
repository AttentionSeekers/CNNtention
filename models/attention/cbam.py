from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class SpatialAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class CBAM:
    def __init__(self):
        self.channel_attention = ChannelAttention()
        self.spatial_attention = SpatialAttention()

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