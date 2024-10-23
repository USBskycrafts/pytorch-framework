import torch
import torch.nn as nn
from torch.nn import functional as F


def taylor_approximation(x, coefficients: torch.Tensor):
    bs, order, *_ = coefficients.shape
    # [bs, order, 1, 1, 1]
    y = torch.zeros_like(x)
    y += coefficients[:, 0:1, :, :]
    for n in range(1, order):
        a = x.unsqueeze(1).repeat(1, n, 1, 1, 1)
        b = torch.arange(
            1, n + 1, device=x.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(bs, 1, 1, 1, 1)
        # print(a.shape, b.shape)
        # print(torch.prod(a / b, dim=1).shape,
        #       coefficients[:, n:n+1, :, :].shape, y.shape)
        y += coefficients[:, n:n+1, :, :] * torch.prod(a / b, dim=1)
    return y


if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(9, 1, 1, 1)
    coefficients = torch.arange(1, 6, 2).unsqueeze(
        0).unsqueeze(-1).unsqueeze(-1).repeat(9, 1, 1, 1)
    y = taylor_approximation(x, coefficients)
    print(y)
