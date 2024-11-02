import torch
import torch.nn as nn
from tools.accuracy_tool import general_accuracy, general_image_metrics


class TemplateNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        print(f"the model structure:\n{self}")
        print(f"parameter size: {sum(p.numel() for p in self.parameters())}")

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        pred = ...
        loss = ...
        acc_result = general_image_metrics(
            pred, data[...], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred
        }
