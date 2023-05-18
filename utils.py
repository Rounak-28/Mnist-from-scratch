import torch
import random
import numpy as np
from PIL import Image
from variables import device, lr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_image(path):
    img_pil = Image.open(path)
    img = np.array(img_pil)
    return img


def custom_log_softmax(x, dim):
    softmax = torch.exp(x) / torch.exp(x).sum(axis=dim, keepdims=True)
    return torch.log(softmax)


def custom_loss(pred, y):
    '''custom cross entropy loss'''
    batch_size = y.size(0)
    log_softmax = custom_log_softmax(pred, dim=1)
    per_batch_ce = [log_softmax[i][y[i]] for i in range(batch_size)]
    summed = sum(per_batch_ce)
    ce = -summed / batch_size
    return ce


def train(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        for p in model.parameters():
            p.grad = None

        loss.backward()

        for p in model.parameters():
            p.data += - lr * p.grad

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def save_model(model, path):
    torch.save(model.state_dict(), path)
