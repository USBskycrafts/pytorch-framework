import torch
import torch.nn as nn
from torch.nn import functional as F


def taylor_approximation(x, coefficients: torch.Tensor):
    offset = coefficients[:, 0:1]
    coefficients = coefficients[:, 1:]
    y = torch.zeros_like(x)
    for order in range(coefficients.shape[1]):
        # print(coefficients[:, order:order+1].shape,
        #   x.shape, offset.shape, y.shape)
        y += coefficients[:, order:order+1] * (x - offset) ** order
    return y


if __name__ == "__main__":
    x = torch.randn(32, 1, 224, 224)
    coefficients = torch.randn(32, 512, 1, 1)
    print(taylor_approximation(x, coefficients).shape)
