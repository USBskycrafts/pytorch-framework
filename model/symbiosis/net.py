import torch
import torch.nn as nn

from model.symbiosis.projection_head import ProjectionHead
from .enhancer import Enhancer
from .decomposer import Decomposer
from itertools import combinations
from tools.accuracy_tool import general_accuracy, general_image_metrics
from model.loss import BCELoss2d, DiceLoss, SobelLoss, FocalLoss2d
from .taylor import taylor_approximation


class Symbiosis(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        self.enhancer = Enhancer()
        self.decomposer = Decomposer()
        self.project_head = ProjectionHead(1, 16)
        self.l1_loss = nn.L1Loss()
        self.sobel_loss = SobelLoss()
        self.focal_loss = FocalLoss2d()
        self.dice_loss = DiceLoss(multiclass=False)

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        mask = data["mask"]
        data = {
            "t1": data["t1"],
            "t1ce": data["t1ce"],
            "t2": data["t2"],
        }
        decomposed = self.decomposer(data, mode)
        mask_pred = self.enhancer(data)
        # \frac{1}{t_{1, obs}} = \frac{1}{t_{1,d}} + r[Gd]
        coefficients = self.project_head(mask_pred)
        enhanced = decomposed['t1']['mapping'] + \
            taylor_approximation(mask_pred, coefficients)
        loss = self.dice_loss(mask.squeeze(dim=1), mask_pred.squeeze(dim=1))
        acc_result = general_accuracy(
            1 - loss.detach().item(), acc_result, "DICE")
        loss += self.focal_loss(mask_pred, mask)
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
        loss += self.l1_loss(data["t1ce"], pred) \
            + self.sobel_loss(pred, data["t1ce"])
        acc_result = general_image_metrics(
            pred, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred
        }
