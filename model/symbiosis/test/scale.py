import unittest
import torch
import torch.nn as nn
from model.symbiosis.decomposer import Decomposer
from model.symbiosis.enhancer import Enhancer
from thop import profile


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.decomposer = Decomposer()
        self.enhancer = Enhancer()

    def forward(self, x, mode):
        x = self.decomposer(x, mode)
        x = self.enhancer(x)
        return x


class TestScale(unittest.TestCase):
    def test_scale(self):
        # TODO: Implement test for scale
        flops, params = profile(Net(), inputs=(
            {
                "t1": torch.randn(1, 1, 224, 224),
                "t2": torch.randn(1, 1, 224, 224),
                "t1ce": torch.randn(1, 1, 224, 224)
            }, "train"))
        print(f"flops: {flops / 1e9}G, params: {params / 1e6}M")
        flops, params = profile(Net(), inputs=(
            {
                "t1": torch.randn(1, 1, 224, 224),
                "t2": torch.randn(1, 1, 224, 224),
                "t1ce": torch.randn(1, 1, 224, 224)
            }, "test"))
        print(f"flops: {flops / 1e9}G, params: {params / 1e6}M")
