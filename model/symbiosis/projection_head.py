import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024):
        super(ProjectionHead, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.model(x).squeeze()
