import torch
import torch.nn as nn
from .backbone import Backbone
from tools.accuracy_tool import general_accuracy, general_image_metrics


class Symbiosis(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = Backbone(
            in_channels=2,
            out_channels=2,
            num_layers=18,
        )
        self.l1_loss = nn.L1Loss()
        print(self)

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        data = {
            "t1": data["t1"],
            "t1ce": data["t1ce"],
        }
        t1_freq = torch.fft.fft2(data["t1"])
        t1_freq = torch.fft.fftshift(t1_freq, dim=(2, 3))
        t1ce_freq = torch.fft.fft2(data["t1ce"])
        t1ce_freq = torch.fft.fftshift(t1ce_freq, dim=(2, 3))
        filter = self.model(
            torch.cat([t1_freq.abs(), t1_freq.angle()], dim=1))
        pred_abs, pred_angle = torch.split(filter, 1, dim=1)
        pred_angle = 2 * torch.pi * pred_angle
        loss = self.l1_loss(pred_abs, t1ce_freq.abs()) \
            + self.l1_loss(pred_angle, t1ce_freq.angle())
        pred = torch.fft.ifftshift(pred_abs *
                                   torch.exp(1j * pred_angle), dim=(2, 3))
        pred = torch.fft.ifft2(pred)
        acc_result = general_image_metrics(
            pred.real, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred.real
        }
