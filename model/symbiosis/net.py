import torch
import torch.nn as nn

from model.residual.res_net import GeneratorResNet
from .segmentation import SegmentationNet
from tools.accuracy_tool import general_accuracy, general_image_metrics
from model.loss import DiceLoss, FocalLoss2d, WeightedBCELoss
from .taylor import taylor_approximation

class Symbiosis(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        self.seg = SegmentationNet(2, 3)
        self.project = GeneratorResNet(4, 1, num_residual_blocks=18)
        self.dice_loss = DiceLoss(multiclass=True)
        self.wbce_loss = WeightedBCELoss(15)
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
        pred = self.project(torch.cat([logits, data['t1']], dim=1))
        
        dice_loss = self.dice_loss(logits, mask)
        wbce_loss = self.wbce_loss(logits, mask)
        l1_loss = self.l1_loss(pred, data['t1ce']) * 8
        loss = l1_loss + dice_loss + wbce_loss

        acc_result = general_accuracy(
            dice_loss.item(), acc_result, "DICE↓"
        )
        acc_result = general_accuracy(
            wbce_loss.item(), acc_result, "WBCE↓"
        )
        acc_result = general_image_metrics(
            pred, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred
        }
