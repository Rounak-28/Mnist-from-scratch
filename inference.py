import torch

from dataset import test_dataloader
from variables import device
from model import Net

model = Net()
model.load_state_dict(torch.load("models/model.pth"))
model.to(device)

total_correct = 0
total_no = 0
for x, y in test_dataloader:
    x = x.to(device)
    pred = model(x).argmax(1).cpu()
    correct = (pred == y).long().sum().item()
    total_correct += correct
    total_no += x.shape[0]

accuracy = total_correct / total_no
print(f"Accuracy is: {accuracy*100:.2f}%")