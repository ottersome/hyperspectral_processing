import torch
from torch import nn


class Model(nn.Module):
    LAYERS = 4

    def __init__(self, input_size=120, output_size=1):
        super(Model, self).__init__()

        layers = [
            nn.Linear(input_size,input_size*2),
            nn.ReLU(),
            nn.Linear(input_size*2,input_size*4),
            nn.ReLU(),
            nn.Linear(input_size*4,input_size*2),
            nn.ReLU(),
            nn.Linear(input_size*2,input_size*1),
            nn.ReLU(),
            nn.Linear(input_size,1)
            
            ]
        # for i in range(self.LAYERS):
        #     layers.append(nn.Linear(input_size // (2**i), input_size // (2 ** (i + 1))))
        #     layers.append(nn.ReLU())
        

        #layers.append(nn.Linear(input_size // (2**self.LAYERS), output_size))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
