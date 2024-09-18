import torch
import torch.nn as nn
from model.residual.res_net import GeneratorResNet


class Enhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enhancer = GeneratorResNet(1, 1)

    def forward(self, data):
        mapping = data["t1"]["mapping"]
        return self.enhancer(mapping)
