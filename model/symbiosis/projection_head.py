import torch
import torch.nn as nn

from model.unet.unet_parts import DoubleConv


class ProjectionHead(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super(ProjectionHead, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.model(x)
