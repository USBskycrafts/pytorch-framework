import torch
import torch.nn as nn
from model.symbiosis.projection_head import ProjectionHead
from model.unet.unet_model import UNetDecoder, UNetEncoder
from model.unet.unet_parts import DoubleConv


class Decomposer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.generator = GeneratorResNet(1, 2)
        self.encoder = UNetEncoder(1, 1024)
        self.pd_decoder = UNetDecoder(1024, 1)
        self.mapping_decoder = UNetDecoder(1024, 1)
        self.conv = DoubleConv(512, 1024)
        self.projection_head = ProjectionHead(512, 1024)

    def forward(self, data, mode):
        result = {}
        for modal_name, modal in data.items():
            if mode == 'test' and modal_name == 't1ce':
                continue
            features = self.encoder(modal)

            out_features = features.pop()
            bs, c, *_ = out_features.shape
            pd_features, mapping_features = torch.split(
                out_features, c // 2, dim=1)
            pd_vector, mapping_vector = map(
                self.projection_head, [pd_features, mapping_features])
            pd_features, mapping_features = self.conv(
                pd_features), self.conv(mapping_features)
            pd, mapping = self.pd_decoder(features, pd_features), self.mapping_decoder(
                features, mapping_features)
            result[modal_name] = {
                "pd": pd,
                "mapping": mapping,
                "pd_vector": pd_vector,
                "mapping_vector": mapping_vector
            }
        return result
