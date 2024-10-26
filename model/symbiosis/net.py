import torch
import torch.nn as nn

from tools.accuracy_tool import general_accuracy, general_image_metrics
from model.user.crossformer import CrossformerEncoder, CrossformerDecoder


class Symbiosis(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        self.similarity_encoder = CrossformerEncoder(1, down_layers=3)
        self.difference_encoder = CrossformerEncoder(1, down_layers=3)
        self.decoder = CrossformerDecoder(1, up_layers=3)
        self.cossine_loss = nn.CosineEmbeddingLoss(margin=0.5)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.InstanceNorm2d(256),
            nn.SiLU(inplace=True),
        )
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
        batch_size = config.getint('train', 'batch_size')
        t1_similarity = self.similarity_encoder(data["t1"])
        t1_difference = self.difference_encoder(data["t1"])
        fused_feature = self.fuse_conv(
            torch.cat([t1_similarity, t1_difference], dim=1))
        pred = self.decoder(fused_feature)
        loss = self.l1_loss(pred, data["t1ce"])
        if mode != 'test':
            t1_similarity = t1_similarity.reshape(batch_size, -1)
            t1_difference = t1_difference.reshape(batch_size, -1)
            t2_similarity = self.similarity_encoder(
                data["t2"]).reshape(batch_size, -1)
            t2_difference = self.difference_encoder(
                data["t2"]).reshape(batch_size, -1)
            t1ce_similarity = self.similarity_encoder(
                data["t1ce"]).reshape(batch_size, -1)
            t1ce_difference = self.difference_encoder(
                data["t1ce"]).reshape(batch_size, -1)
            loss += self.cossine_loss(t1_similarity,
                                      t1ce_similarity,
                                      torch.ones(batch_size, device=t1_similarity.device)) \
                + self.cossine_loss(t1_similarity,
                                    t2_similarity,
                                    -torch.ones(batch_size, device=t1_similarity.device)) \
                + self.cossine_loss(t1ce_similarity,
                                    t2_similarity,
                                    -torch.ones(batch_size,
                                                device=t1_similarity.device)) \
                + self.cossine_loss(t1_difference,
                                    t2_difference,
                                    torch.ones(batch_size,
                                               device=t1_similarity.device)) \
                + self.cossine_loss(t1_difference,
                                    t1ce_difference,
                                    -torch.ones(batch_size,
                                                device=t1_similarity.device)) \
                + self.cossine_loss(t1ce_difference,
                                    t2_difference,
                                    torch.ones(batch_size,
                                               device=t1_similarity.device))

        acc_result = general_image_metrics(
            pred, data["t1ce"], config, acc_result)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": [acc_result["PSNR"]],
            "pred": pred
        }
