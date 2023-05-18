import torch

lr = 0.001
epochs = 2
device = "cuda" if torch.cuda.is_available() else "cpu"