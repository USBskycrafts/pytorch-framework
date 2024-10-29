import torch
import torch.nn as nn

from tools.accuracy_tool import general_accuracy, general_image_metrics
from model.residual.res_net import GeneratorResNet


class Symbiosis(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        self.model = GeneratorResNet(2, 2, 14)
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
        t1_r, t1_imag = torch.real(t1_freq), torch.imag(t1_freq)
        pred_freq = self.model(torch.cat([t1_r, t1_imag], dim=1))
        pred_freq = torch.complex(pred_freq[:, :1, :, :], pred_freq[:, 1:, :, :])
        pred = torch.fft.ifft2(pred_freq) 
        loss = self.l1_loss(pred.real, data['t1ce']) + self.l1_loss(pred.imag, torch.zeros_like(data['t1ce']))
        acc_result = general_image_metrics(
            pred.real, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred
        }
