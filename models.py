from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1, residual=True):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, x):
        out = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))
        return x + out if self.residual else out
