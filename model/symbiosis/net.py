import torch
import torch.nn as nn
from .enhancer import Enhancer
from .decomposer import Decomposer
from itertools import combinations
from tools.accuracy_tool import general_image_metrics


class Symbiosis(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        self.enhancer = Enhancer()
        self.decomposer = Decomposer()
        self.l1_loss = nn.L1Loss()

    def multi_gpu_list(self, device, config, *args, **kwargs):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        decomposed = self.decomposer(data, mode)
        enhanced = self.enhancer(decomposed)
        loss = 0
        if mode != "test":
            for modal_name, component in decomposed.items():
                pd = component["pd"]
                mapping = component["mapping"]
                if modal_name == 't2':
                    loss += self.l1_loss(data[modal_name],
                                         pd * torch.exp(-mapping))
                elif modal_name in ['t1', 't1ce']:
                    loss += self.l1_loss(data[modal_name],
                                         pd * (1 - torch.exp(-mapping)))
                else:
                    raise ValueError(
                        "Unknown modal name: {}".format(modal_name))
            for components in combinations(decomposed.values(), 2):
                pd1 = components[0]["pd"]
                pd2 = components[1]["pd"]
                loss += self.l1_loss(pd1, pd2)
            loss += self.l1_loss(decomposed["t1ce"]["mapping"],
                                 enhanced)

        pred = decomposed["t1"]["pd"] * (1 - torch.exp(-enhanced))
        loss += self.l1_loss(data["t1ce"], pred)
        acc_result = general_image_metrics(
            pred, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [loss],
            "pred": pred
        }
