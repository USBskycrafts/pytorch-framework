import torch
import torch.nn as nn

from model.symbiosis.projection_head import ProjectionHead
from .enhancer import Enhancer
from .decomposer import Decomposer
from itertools import combinations
from tools.accuracy_tool import general_image_metrics
from model.loss import SobelLoss


class Symbiosis(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        self.enhancer = Enhancer()
        self.decomposer = Decomposer()
        self.project_head = ProjectionHead()
        self.l1_loss = nn.L1Loss()
        self.cos = nn.CosineEmbeddingLoss(margin=0.5)
        self.sobel_loss = SobelLoss()

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        data = {
            "t1": data["t1"],
            "t1ce": data["t1ce"],
            "t2": data["t2"],
        }
        decomposed = self.decomposer(data, mode)
        t1_weight, t1_bias = torch.split(
            self.enhancer(decomposed), 1, dim=1)
        enhanced = decomposed['t1']['mapping'] * t1_weight + t1_bias
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
                bs, *_ = component["pd_vector"].shape
                loss += self.cos(component["pd_vector"],
                                 component["mapping_vector"], -1 * torch.ones(bs))
            for components in combinations(decomposed.values(), 2):
                pd1 = components[0]["pd_vector"]
                pd2 = components[1]["pd_vector"]
                loss += self.l1_loss(pd1, pd2)
            loss += self.l1_loss(decomposed["t1ce"]["mapping"],
                                 enhanced) \
                + self.sobel_loss(decomposed["t1ce"]["mapping"], enhanced)

        pred = decomposed["t1"]["pd"] * (1 - torch.exp(-enhanced))
        loss += self.l1_loss(data["t1ce"], pred)
        acc_result = general_image_metrics(
            pred, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred
        }
