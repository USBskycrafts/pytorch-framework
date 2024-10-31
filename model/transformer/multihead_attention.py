import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "Model dimension must be divisible by number of heads"
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model, "Input dimension must match model dimension"
        assert d_model % self.num_heads == 0, "Model dimension must be divisible by number of heads"

        # Project input to query, key, and value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)

        # Apply attention to value
        x = torch.matmul(attention, v)

        # Reshape and concatenate heads

        x = x.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        x = self.out_proj(x)
        return x
