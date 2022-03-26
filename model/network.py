import torch.nn as nn
import torch.nn.functional as F


# simple model demo
class Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # mode 1
        self.layer1_conv = nn.Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer1_relu = nn.ReLU()

        # mode 2
        self.layer2 = nn.Sequential(nn.Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU())

        self.layer3 = nn.Conv2d(10, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = self.layer1_conv(x)
        x = self.layer1_relu(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
