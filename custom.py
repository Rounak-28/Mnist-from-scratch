import torch
import torch.nn as nn
from variables import device


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_k = (1 / in_features) ** 0.5
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.w = nn.Parameter(torch.empty(in_features, out_features, device=device))
        self.b = nn.Parameter(torch.empty(out_features, device=device)) if self.bias else None
        
        nn.init.uniform_(self.w, -root_k, root_k)
        nn.init.uniform_(self.b, -root_k, root_k)
    
    def forward(self, x):
        x = x @ self.w
        if self.b is not None:
            x += self.b
        return x
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"


class ReLU(nn.Module):

    def relu(self, x):
        zero = torch.tensor([0]).to(device)
        x = x.to(device)
        return torch.max(zero, x)
    
    def forward(self, x):
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