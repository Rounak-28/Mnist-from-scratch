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


class CrossEntropyLoss:
    
    def log_softmax(self, x, dim):
        softmax = torch.exp(x) / torch.exp(x).sum(axis=dim, keepdims=True)
        return torch.log(softmax)

    def __call__(self, pred, y):
        '''custom cross-entropy loss'''
        batch_size = y.size(0)
        log_softmax = self.log_softmax(pred, dim=1)
        per_batch_ce = [log_softmax[i][y[i]] for i in range(batch_size)]
        summed = sum(per_batch_ce)
        ce = -summed / batch_size
        return ce