import torch
import torch.nn as nn
from model.residual.res_net import GeneratorResNet
from model.unet.unet_model import UNetDecoder, UNetEncoder


class Enhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enhancer = GeneratorResNet(2, 2)

    def forward(self, data):
        x = torch.cat([data["t1"]["mapping"],
                      data["t2"]["mapping"]], dim=1)
        return self.enhancer(x)
