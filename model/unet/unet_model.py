import numpy as np
import torch
import torch.nn as nn
from .unet_parts import DoubleConv, Down, OutConv, Up


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1024):
        super(UNetEncoder, self).__init__()
        self.model = nn.ModuleList()
        in_features = 32
        self.model += [DoubleConv(in_channels, in_features)]
        while in_features * 2 <= out_channels:
            self.model.append(Down(in_features, in_features * 2))
            in_features *= 2
        self.out_features = in_features
        print(f"encoder: {len(self.model)} layers")

    def forward(self, x):
        features = []
        for layer in self.model:
            x = layer(x)
            features.append(x)
        return features


class UNetDecoder(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1):
        super(UNetDecoder, self).__init__()
        self.model = nn.ModuleList()
        in_features = in_channels
        while in_features // 2 >= 32:
            self.model.append(Up(in_features, in_features // 2))
            in_features //= 2
        self.out_conv = OutConv(in_features, out_channels)
        print(f"decoder: {len(self.model) + 1} layers")

    def forward(self, features, x):
        for layer, feature in zip(self.model, features[::-1]):
            x = layer(x, feature)
        return self.out_conv(x)
