import torch
import torch.nn as nn

from tools.accuracy_tool import general_accuracy, general_image_metrics
from model.residual.res_net import GeneratorResNet


class Symbiosis(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        self.model = GeneratorResNet(2, 1, 12)
        self.l1_loss = nn.L1Loss()
        print(self)

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        data = {
            "t1": data["t1"],
            "t1ce": data["t1ce"],
            "t2": data["t2"],
        }
        pred = self.model(torch.cat([data["t1"], data["t2"]], dim=1))
        loss = self.l1_loss(pred, data['t1ce'])
        acc_result = general_image_metrics(
            pred, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred
        }
