import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    LAYERS = 4

    def __init__(self, input_size=120, output_size=1):
        super(Model, self).__init__()

        layers = [
            nn.Linear(input_size, input_size * 4),
            nn.ReLU(),
            nn.Linear(input_size * 4, input_size * 2),
            nn.ReLU(),
            nn.Linear(input_size * 2, input_size * 1),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
            nn.ReLU(),
        ]
        # for i in range(self.LAYERS):
        #     layers.append(nn.Linear(input_size // (2**i), input_size // (2 ** (i + 1))))
        #     layers.append(nn.ReLU())

        # layers.append(nn.Linear(input_size // (2**self.LAYERS), output_size))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class SpatialModel(nn.Module):
    LAYERS = 4

    def __init__(self, spatial_radius: int, channel_size=120, output_size=1):
        super(SpatialModel, self).__init__()

        self.conv_layer = nn.Conv2d(
            channel_size, channel_size * 2, spatial_radius * 2 + 1
        )

        lin_layers = [
            nn.Linear(channel_size * 2, channel_size * 4),
            nn.ReLU(),
            nn.Linear(channel_size * 4, channel_size * 2),
            nn.ReLU(),
            nn.Linear(channel_size * 2, channel_size * 1),
            nn.ReLU(),
            nn.Linear(channel_size, output_size),
            nn.ReLU(),
        ]
        # for i in range(self.LAYERS):
        #     layers.append(nn.Linear(input_size // (2**i), input_size // (2 ** (i + 1))))
        #     layers.append(nn.ReLU())

        # layers.append(nn.Linear(input_size // (2**self.LAYERS), output_size))
        self.seq = nn.Sequential(*lin_layers)

    def forward(self, x):
        batch_size = x.shape[0]
        conv_res = F.relu(self.conv_layer(x))
        conv_res_flat = conv_res.view(batch_size, -1)
        return self.seq(conv_res_flat)
