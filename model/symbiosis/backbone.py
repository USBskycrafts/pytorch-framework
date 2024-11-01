import torch
import torch.nn as nn
import math
from model.transformer.encoder import TransformerEncoder


class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, d_model=256, num_heads=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model * in_channels,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, d_model * 2))

    def forward(self, x):
        bs, _, seq_len, d_model = x.shape
        tokens = x.transpose(1, 2).contiguous().reshape(bs, seq_len, -1)
        tokens = tokens + self.pos_embedding
        tokens = self.encoder(tokens)
        return tokens.reshape(bs, seq_len, 2, d_model).transpose(1, 2).contiguous()
