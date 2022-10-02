import torch.nn as nn
import torch
import torch.nn.functional as F

"""
1. Модель - это класс, который наследуется от nn.Module
для которой надо написать прямой проход (forward), на вход берёт объекты (x)

2. Слои с параметрами должны быть в конструкторе!
"""


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2):
        """
        :param in_channels: (Int), number of channels in input image
               out_channels: (Int), number of channels in output image
        """
        super(Residual, self).__init__()
        self.in_channels = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels1, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.out_channels1, self.out_channels2, 3, 1, 1)

        self.bnorm1 = nn.BatchNorm2d(in_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels1)

        if in_channels == out_channels2:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(self.in_channels, self.out_channels2, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bnorm1(x)))
        out = self.conv2(torch.relu(self.bnorm2(out)))
        out = out + self.skip(x)

        return out


class FinalLayer(nn.Module):
    def __init__(self, img_size):
        super(FinalLayer, self).__init__()

        self.w1 = nn.Parameter(torch.randn(1, 1, img_size, img_size))
        self.w2 = nn.Parameter(torch.randn(1, 1, img_size, img_size))

        self.w1.requires_grad = True
        self.w2.requires_grad = True

    def forward(self, x1, x2):
        weighted_sum = (self.w1 * x1 + self.w2 * x2) / (self.w1.sum() + self.w2.sum())
        return weighted_sum


class Model(nn.Module):
    def __init__(self, in_channels, channels):
        """
        :param in_channels: (Int), number of channels in input image
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.channels = channels

        self.down_model = self.make_down()
        self.middle_model = self.make_middle()
        self.top_model = self.make_top()

        self.max_pool = F.max_pool2d
        self.upsample = F.upsample

        self.x_final = nn.Conv2d(self.channels, 1, 1, 1)
        self.w_final = FinalLayer(256)
        self.b_final = nn.BatchNorm2d(in_channels)
        self.final = nn.Conv2d(in_channels, 1, 3, 1, 1)

    def make_down(self):
        model = nn.ModuleList()

        in_channels = self.in_channels
        out_channels = self.channels

        for _ in range(4):
            model.append(Residual(in_channels, out_channels, out_channels))
            in_channels = out_channels
            out_channels *= 2

        return model

    def make_middle(self):
        model = Residual(self.channels * 8, self.channels * 16, self.channels * 8)

        return model

    def make_top(self):
        model = nn.ModuleList()

        in_channels = self.channels * 16
        out_channels = self.channels * 4

        for i in range(4):
            model.append(nn.Sequential(
                Residual(in_channels, in_channels // 2, in_channels // 2),
                nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1)
            ))
            if i == 2:
                in_channels //= 2
                out_channels = out_channels
            else:
                in_channels //= 2
                out_channels //= 2

        return model

    def forward(self, x, y):
        """
        Forward propagation
        :param x: (Tensor), [b_size x C_in x W x H], input object
        :return: (Tensor), [b_size x C_in x W x H], output object
        """
        down_outputs = []

        # [B x C_in x W x H] -> [B x 8*C x W/16 x H/16]
        for i in range(4):
            x = self.down_model[i](x)
            down_outputs.append(x)
            x = self.max_pool(x, kernel_size=3, stride=2, padding=1)

        # [B x 8*C x W/16 x H/16] -> [B x 16*C x W/8 x H/8]
        x = self.upsample(self.middle_model(x), scale_factor=2)

        # [B x 16*C x W/8 x H/8] -> [B x C x W x H]
        for i in range(4):
            x = torch.cat((x, down_outputs[-i - 1]), dim=1)
            x = self.top_model[i](x)
            if i == 3:
                continue
            else:
                x = self.upsample(x, scale_factor=2, mode='nearest')
        # a = self.w_final(self.x_final(x), y)
        out = self.final(x)

        return F.sigmoid(out)
