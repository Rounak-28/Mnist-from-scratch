from utils import train, test, custom_loss, save_model
from dataset import train_dataloader, test_dataloader
from model import Net
from variables import device, epochs

model = Net().to(device)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, custom_loss)
    test(test_dataloader, model, custom_loss)
print("Done!")


save_model(model, "models/model.pth")