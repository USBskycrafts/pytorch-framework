import torch
import torch.nn as nn

from model.symbiosis.projection_head import ProjectionHead
from .segmentation import SegmentationNet
from itertools import combinations
from tools.accuracy_tool import general_accuracy, general_image_metrics
from model.loss import DiceLoss, WeightedBCELoss
from .taylor import taylor_approximation


class Symbiosis(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        self.seg = SegmentationNet()
        self.project = ProjectionHead(2, 32)
        self.dice_loss = DiceLoss(multiclass=False)
        self.wbce = WeightedBCELoss(local_size=9)
        self.l1_loss = nn.L1Loss()

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        mask = data["mask"]
        data = {
            "t1": data["t1"],
            "t1ce": data["t1ce"],
            "t2": data["t2"],
        }
        mask_pred = self.seg(data)
        coefficients = self.project(torch.cat([data['t1'], mask_pred], dim=1))
        bias = taylor_approximation(mask_pred, coefficients)
        pred = data['t1'] + bias

        loss = self.dice_loss(mask_pred.squeeze(dim=1), mask.squeeze(dim=1)) + \
            self.l1_loss(pred, data['t1ce']) + \
            self.wbce(mask_pred, mask)

        acc_result = general_image_metrics(
            pred, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred
        }
