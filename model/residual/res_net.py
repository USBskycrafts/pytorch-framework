import torch
import torch.nn as nn
from .res_block import ResidualBlock


class GeneratorResNet(nn.Module):
    # (input_shape = (3, 256, 256), num_residual_blocks = 9)
    def __init__(self, input_dim, output_dim, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        channels = input_dim  # 输入通道数channels = 3

        # 初始化网络结构
        out_features = 64  # 输出特征数out_features = 64
        model = [  # model = [Pad + Conv + Norm + ReLU]
            # ReflectionPad2d(3):利用输入边界的反射来填充输入张量
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),  # Conv2d(3, 64, 7)
            # InstanceNorm2d(64):在图像像素上对HW做归一化，用在风格化迁移
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(inplace=True),  # 非线性激活
        ]
        in_features = out_features  # in_features = 64

        # 下采样，循环2次
        for _ in range(2):
            out_features *= 2  # out_features = 128 -> 256
            model += [  # (Conv + Norm + ReLU) * 2
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(inplace=True),
            ]
            in_features = out_features  # in_features = 256

        # 残差块儿，循环9次
        for _ in range(num_residual_blocks):
            # model += [pad + conv + norm + relu + pad + conv + norm]
            model += [ResidualBlock(out_features)]

        # 上采样两次
        for _ in range(2):
            out_features //= 2  # out_features = 128 -> 64
            model += [  # model += [Upsample + conv + norm + relu]
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(inplace=True),
            ]
            in_features = out_features  # out_features = 64

        # 网络输出层                                                            ## model += [pad + conv + tanh]
        # 将(3)的数据每一个都映射到[-1, 1]之间
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(out_features, output_dim, 7),
                  nn.Softplus()]

        self.model = nn.Sequential(*model)

    def forward(self, x):  # 输入(1, 3, 256, 256)
        return self.model(x)  # 输出(1, 3, 256, 256)
