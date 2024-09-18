import torch
import torch.nn as nn
from model.residual.res_net import GeneratorResNet


class Decomposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = GeneratorResNet(1, 2)

    def forward(self, data, mode):
        result = {}
        for modal_name, modal in data.items():
            if mode == 'test' and modal_name != 't1':
                continue
            pd, mapping = torch.split(self.generator(modal), 1, dim=1)
            result[modal_name] = {
                "pd": pd,
                "mapping": mapping
            }
        return result
