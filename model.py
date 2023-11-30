import torch
import torch.nn as nn
import constants as c

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super(Yolov1, self).__init__()
        self.in_channels = in_channels
        self.convs = self._create_conv_layers()
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.convs(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self):
        layers = []
        layers.append(nn.Conv2d(1, 64, kernel_size=7, bias=False, stride=2, padding=3))
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        layers.append(CNNBlock(64, 64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        return nn.Sequential(*layers)

    def _create_fcs(self):
        return nn.Sequential(
            nn.Flatten(),
            # nn.Linear(64 * c.SR * c.SC, 496),
            nn.Linear(c.IMAGE_HEIGHT/8 * c.IMAGE_WIDTH/8 * 64, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, c.SR * c.SC * (c.C + c.B * 5)),
        )
