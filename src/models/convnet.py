import torch
import torch.nn as nn
from a2c_ppo_acktr.utils import init


class ConvNetEncoder(nn.Module):
    """ConvNet intended to encode 26x26 input."""

    def __init__(self, n_features=128):
        super().__init__()

        # trunk-ignore(flake8/E731)
        init_orthogonal = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.conv1 = init_orthogonal(nn.Conv2d(3, 32, 4, stride=2))
        self.conv2 = init_orthogonal(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_orthogonal(nn.Conv2d(64, 64, 4, stride=1))
        self.conv4 = init_orthogonal(nn.Conv2d(64, n_features, 2, stride=1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x - 0.5  # Shift to [-0.5, 0.5]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = torch.flatten(x, 1)

        return x
