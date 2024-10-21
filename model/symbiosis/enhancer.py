import torch
import torch.nn as nn
from model.residual.res_net import GeneratorResNet
from model.unet.unet_model import UNetDecoder, UNetEncoder


class Enhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder(2, 5)
        self.decoder = UNetDecoder(2, 5)

    def forward(self, data):
        x = torch.cat([data["t1"],
                      data["t2"]], dim=1)
        features = self.encoder(x)
        x = features.pop()
        return self.decoder(features, x)