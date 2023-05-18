import torch.nn as nn
from custom import Linear, ReLU, Sequential, Flatten

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.layers = Sequential(
            Linear(28*28, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x