import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossformerPack(nn.Module):
    def __init__(self, input_dim: int, group=3, n_layer=3):
        super(CrossformerPack, self).__init__()
        self.input_dim = input_dim
        self.group = group
        self.layer_num = n_layer

        layers = []
        for _ in range(n_layer):
            layers.append(CrossformerLayer(
                input_dim, group=group, attention_type="local"))
            layers.append(CrossformerLayer(
                input_dim, group=group, attention_type="long"))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CrossformerLayer(nn.Module):
    def __init__(self, input_dim: int, group=3, attention_type="long"):
        super(CrossformerLayer, self).__init__()
        self.input_dim = input_dim
        self.group = group
        self.attention_type = attention_type

        self.position_embedding = nn.parameter.Parameter(torch.randn(
            (1, group * group, input_dim)
        ))
        self.encoder = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            norm_first=True,
        )
        self.groups = 0

    def padding(self, x):
        # WARN: the image should be padded to the multiple of group

        # first, calculate the size of the padded image
        x_diff = (x.shape[2] + self.group -
                  1) // self.group * self.group - x.shape[2]
        y_diff = (x.shape[3] + self.group -
                  1) // self.group * self.group - x.shape[3]
        assert x_diff >= 0 and y_diff >= 0, "the image should be padded to the multiple of group"
        self.padding_size = (
            y_diff // 2,
            y_diff // 2 + (1 if y_diff % 2 == 1 else 0),
            x_diff // 2,
            x_diff // 2 + (1 if x_diff % 2 == 1 else 0),
        )
        x = F.pad(x, self.padding_size, mode='reflect')
        assert x.shape[2] % self.group == 0, f"{x.shape}"
        assert x.shape[3] % self.group == 0, f"{x.shape}"
        return x

    def forward(self, x):
        x_ = self.padding(x)
        bs, c, h, w = x_.shape
        if self.attention_type == "local":
            groups = F.unfold(x_, kernel_size=(self.group, self.group),
                              stride=(self.group, self.group))
            groups = groups.reshape(bs, c, self.group * self.group, -1)
            groups = groups.permute(0, 3, 2, 1)
            groups = groups.reshape(-1, self.group * self.group, c)
            groups += self.position_embedding
            groups = self.encoder(groups)
            groups = groups.reshape(bs, -1, self.group * self.group, c)
            groups = groups.permute(0, 3, 2, 1)
            groups = groups.reshape(bs, c * self.group * self.group, -1)
            y = F.fold(groups, output_size=(h, w),
                       kernel_size=(self.group, self.group),
                       stride=(self.group, self.group))
        elif self.attention_type == "long":
            stride = (h // self.group, w // self.group)
            # project the groups
            groups = F.unfold(x_, kernel_size=stride,
                              stride=stride)
            groups = groups.reshape(
                bs, c, stride[0] * stride[1], self.group * self.group)
            groups = groups.permute(0, 2, 3, 1)
            groups = groups.reshape(-1, self.group * self.group, c)
            groups += self.position_embedding
            groups = self.encoder(groups)
            groups = groups.reshape(bs, stride[0] * stride[1],
                                    self.group * self.group, c)
            groups = groups.permute(0, 3, 1, 2)
            groups = groups.reshape(bs, c * stride[0] * stride[1],
                                    self.group * self.group)
            y = F.fold(groups, output_size=(h, w), kernel_size=stride,
                       stride=stride)
        else:
            raise NotImplementedError("please check the attention type")
        # remove the padding
        h, w = x_.shape[2:]
        left, right, top, bottom = self.padding_size
        y = y[:, :, top:h-bottom, left:w-right]
        assert y.shape == x.shape, f"y's shape: {y.shape}, x's shape: {x.shape}, padding size: {self.padding_size}"
        return y
