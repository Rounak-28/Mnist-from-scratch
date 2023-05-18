import torch

epochs = 2
device = "cuda" if torch.cuda.is_available() else "cpu"