import torch
import torch.nn as nn
from model.residual.res_net import GeneratorResNet
from model.unet.unet_model import UNetDecoder, UNetEncoder


class SegmentationNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.encoder = UNetEncoder(input_channels, 5)
        self.decoder = UNetDecoder(output_channels, 5)

    def forward(self, data):
        x = torch.cat([data["t1"],
                      data["t2"]], dim=1)
        features = self.encoder(x)
        x = features.pop()
        return self.decoder(features, x)
