import torch.nn as nn
from .crossformer_layer import CrossformerPack
from .cross_embedding import CrossScaleEmbedding


class CrossformerEncoder(nn.Module):
    def __init__(self, in_channels, down_layers=3):
        super(CrossformerEncoder, self).__init__()
        channels = 64
        model = []
        model += [CrossScaleEmbedding(in_channels, channels,
                                      kernel_size=[4, 8, 16, 32], stride=2),
                  nn.InstanceNorm2d(channels * 2),
                  nn.SiLU(inplace=True)]

        for i in range(1, down_layers):
            model += [CrossScaleEmbedding(channels, channels * 2,
                                          kernel_size=[2, 4], stride=2),
                      nn.InstanceNorm2d(channels * 2),
                      nn.SiLU(inplace=True),
                      CrossformerPack(channels * 2, group=7, n_layer=i)]
            channels *= 2
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CrossformerDecoder(nn.Module):
    def __init__(self, out_channels, up_layers=3):
        super(CrossformerDecoder, self).__init__()
        in_channels = 64 * 2 ** (up_layers - 1)
        model = []
        for i in range(up_layers - 1, 0, -1):
            model += [CrossScaleEmbedding(in_channels, in_channels // 2,
                                          kernel_size=[2, 4], stride=2, reversed=True),
                      nn.InstanceNorm2d(in_channels // 2),
                      nn.SiLU(inplace=True),
                      CrossformerPack(in_channels // 2, group=7, n_layer=i)]
            in_channels //= 2
        model += [CrossScaleEmbedding(in_channels, 16,
                                      kernel_size=[4, 8, 16, 32], stride=2, reversed=True),
                  nn.InstanceNorm2d(out_channels),
                  nn.Conv2d(16, out_channels, 1),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
