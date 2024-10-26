import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F


class CrossScaleEmbedding(nn.Module):
    # Should be used in pairs
    def __init__(self, input_dim: int, output_dim: int,
                 kernel_size: List[int] = [2, 4],
                 stride: int = 2, reversed: bool = False):
        super(CrossScaleEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = [k for k in sorted(kernel_size)]
        self.stride = stride
        self.reversed = reversed
        self.convs = nn.ModuleList()

        if not reversed:
            conv = nn.Conv2d
        else:
            conv = nn.ConvTranspose2d
        token_size = self.token_size(self.kernel_size, output_dim)
        self.dim_list = token_size
        for i, k in enumerate(self.kernel_size):
            self.convs.append(conv(input_dim, token_size[i],
                                   kernel_size=k, stride=stride, padding=self.padding_size(k, stride)))

    def token_size(self, kernel_size, output_dim) -> List[int]:
        token_dim = []
        for i in range(1, len(kernel_size)):
            token_dim.append(output_dim // (2**i))
            # the largest token dim should equals to the
            # secondary largest token dim
        token_dim.append(output_dim // (2**(len(kernel_size) - 1)))
        return token_dim

    def padding_size(self, kernel_size, stride) -> int:
        """Calculate padding size for convolution

        Args:
            kernel_size (_type_): _description_
            stride (_type_): _description_

        Returns:
            _type_: _description_
        while dilation=1,
        y.shape = (x.shape + 2 * p.shape - k.shape) // stride + 1
        if we want y.shape = x.shape // stride
        then we get this function
        """
        if (kernel_size - stride) % 2 == True:
            return (kernel_size - stride) // 2
        else:
            return (kernel_size - stride + 1) // 2

    def forward(self, x):
        # from [B, C, H, W] to [B, H // stride, W // stride, C * stride]
        tokens = torch.cat([conv(x)
                            for conv in self.convs], dim=1)
        # a recursion to the deep layers
        return tokens


if __name__ == '__main__':
    input = torch.randn(1, 64, 32, 32)
    model = CrossScaleEmbedding(64, 128, [2, 4, 8], 2)
    output = model(input)
    print(output.shape)
