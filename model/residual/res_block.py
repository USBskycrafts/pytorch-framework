import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(  # block = [pad + conv + norm + relu + pad + conv + norm]
            nn.ReflectionPad2d(1),  # ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),  # 卷积
            # InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),  # 非线性激活
            nn.ReflectionPad2d(1),  # ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),  # 卷积
            # InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):  # 输入为 一张图像
        return x + self.block(x)  # 输出为 图像加上网络的残差输出
