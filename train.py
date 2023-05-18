from model import Net
from variables import device, epochs
from dataset import train_dataloader, test_dataloader
from utils import train, test, save_model
from custom import CrossEntropyLoss

model = Net().to(device)

loss_fn = CrossEntropyLoss()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn)
    test(test_dataloader, model, loss_fn)
print("Done!")

save_model(model, "models/model.pth")