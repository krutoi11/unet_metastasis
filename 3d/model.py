import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, in_channels, channels):
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.channels = channels

        self.block = self.make_block()
        self.final = nn.Linear(16128, 1)

    def make_block(self):
        list_ = []
        for i in range(6):
            list_.append(nn.Conv3d(self.in_channels, self.channels, 3, 2))
            list_.append(nn.BatchNorm3d(self.channels))
            list_.append(nn.ReLU())

            self.in_channels = self.channels
            self.channels = self.channels * 2
        return nn.Sequential(*list_)

    def forward(self, x):
        y = self.block(x)
        return self.final(torch.flatten(y, start_dim=1, end_dim=4))
