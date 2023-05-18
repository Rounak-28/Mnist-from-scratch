import glob
import random
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from utils import set_seed, read_image
set_seed()

class MNISTDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = read_image(img_path)
        label = int(img_path.split("/")[-2])

        if self.transform:
            image = self.transform(image)

        return image, label


train_images = glob.glob("data/training/*/*")
test_images = glob.glob("data/testing/*/*")

random.shuffle(train_images)
random.shuffle(test_images)

train_images = train_images[:500]
test_images = test_images[:500]

training_data = MNISTDataset(train_images, ToTensor())
test_data = MNISTDataset(test_images, ToTensor())

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)
