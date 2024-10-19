import numpy as np
import torch
import torch.nn as nn
from .unet_parts import DoubleConv, Down, OutConv, Up


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, layers=5):
        super(UNetEncoder, self).__init__()
        self.model = nn.ModuleList()
        in_features = 32
        upper_bound = 512
        self.model += [DoubleConv(in_channels, in_features)]
        for layer in range(layers):
            self.model.append(
                Down(min(2 ** (layer + 5), upper_bound), min(2 ** (layer + 6), upper_bound)))
        print(f"encoder: {len(self.model)} layers: {self}")

    def forward(self, x):
        features = []
        for layer in self.model:
            x = layer(x)
            features.append(x)
        return features


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, layers=5, out_activation=None):
        super(UNetDecoder, self).__init__()
        self.model = nn.ModuleList()
        upper_bound = 512
        for layer in range(layers, 0, -1):
            self.model.append(Up(min(upper_bound, 2 ** (layer + 5)),
                              min(2 ** (layer + 4), upper_bound), layer))
        self.out_conv = OutConv(32, out_channels, out_activation)
        print(f"decoder: {len(self.model) + 1} layers: {self}")

    def forward(self, features, x):
        for layer, feature in zip(self.model, features[::-1]):
            x = layer(x, feature)
        return self.out_conv(x)
