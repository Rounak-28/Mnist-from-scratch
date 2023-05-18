import torch
import torch.nn as nn

from variables import device

class ReLU(nn.Module):

    def relu(self, x):
        zero = torch.tensor([0]).to(device)
        x = x.to(device)
        return torch.max(zero, x)
    
    def __call__(self, x):
        return self.relu(x)