#model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FireModule(nn.Module):
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=s1x1, kernel_size=1, stride=1)
        self.expand1x1 = nn.Conv2d(in_channels=s1x1, out_channels=e1x1, kernel_size=1)
        self.expand3x3 = nn.Conv2d(in_channels=s1x1, out_channels=e3x3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        x1 = self.expand1x1(x)
        x2 = self.expand3x3(x)
        x = F.relu(torch.cat((x1, x2), dim=1))
        return x


class SqueezeNet(nn.Module):
    def __init__(self, out_channels):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire2 = FireModule(in_channels=96, s1x1=16, e1x1=64, e3x3=64)
        self.fire3 = FireModule(in_channels=128, s1x1=16, e1x1=64, e3x3=64)
        self.fire4 = FireModule(in_channels=128, s1x1=32, e1x1=128, e3x3=128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire5 = FireModule(in_channels=256, s1x1=32, e1x1=128, e3x3=128)
        self.fire6 = FireModule(in_channels=256, s1x1=48, e1x1=192, e3x3=192)
        self.fire7 = FireModule(in_channels=384, s1x1=48, e1x1=192, e3x3=192)
        self.fire8 = FireModule(in_channels=384, s1x1=64, e1x1=256, e3x3=256)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire9 = FireModule(in_channels=512, s1x1=64, e1x1=256, e3x3=256)
        self.dropout = nn.Dropout(p=0.5)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=12, stride=1)
        # We don't have FC Layers, inspired by NiN architecture.

    def forward(self, x):
        # First max pool after conv1
        x = self.max_pool1(self.conv1(x))
        # Second max pool after fire4
        x = self.max_pool2(self.fire4(self.fire3(self.fire2(x))))
        # Third max pool after fire8
        x = self.max_pool3(self.fire8(self.fire7(self.fire6(self.fire5(x)))))
        # Final pool (avg in this case) after conv10
        x = self.avg_pool(self.conv10(self.fire9(x)))
        return torch.flatten(x, start_dim=1)