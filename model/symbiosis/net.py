import torch
import torch.nn as nn

from model.residual.res_net import GeneratorResNet
from .segmentation import SegmentationNet
from tools.accuracy_tool import general_accuracy, general_image_metrics
from model.loss import DiceLoss, FocalLoss2d
from .taylor import taylor_approximation

class Symbiosis(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        self.seg = SegmentationNet()
        self.project = GeneratorResNet(2, 1)
        self.dice_loss = DiceLoss(multiclass=False)
        self.focal_loss = FocalLoss2d()
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
        logits = self.seg(data)
        weight = self.project(torch.cat([logits, data['t1']], dim=1))
        pred = data['t1'] * weight

        dice_loss = self.dice_loss(logits.squeeze(dim=1), mask.squeeze(dim=1))
        focal_loss = self.focal_loss(logits, mask)
        l1_loss = self.l1_loss(pred, data['t1ce']) * 8
        loss = dice_loss + focal_loss + l1_loss

        acc_result = general_accuracy(
            1 - dice_loss.item(), acc_result, "DICE"
        )
        acc_result = general_image_metrics(
            pred, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred
        }
