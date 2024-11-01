import torch
import torch.nn as nn
import math
from model.transformer.encoder import TransformerEncoder


class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, d_model=1024, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.tokenizer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=(1, 256)
        )
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        tokens = self.tokenizer(x).squeeze().transpose(1, 2).contiguous()
        tokens = tokens + self.pos_embedding
        tokens = self.encoder(tokens)
        bs, seq_len, d_model = tokens.shape
        return self.out_proj(tokens).view(bs, seq_len, 2, seq_len) \
            .transpose(1, 2).contiguous()
