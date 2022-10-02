import matplotlib.pyplot as plt
from dataset import BrainDataset
from torch.utils.data import DataLoader
from unet import UNet
from model import Model
import torchvision

img_size = 256

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize([img_size, img_size])])

dataset_train = BrainDataset(root_dir='../datas/images_no_patients_no_empty_train',
                             transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=32,
                              shuffle=True, num_workers=0)
dataset_test = BrainDataset(root_dir='../datas/images_no_patients_no_empty_test',
                            transform=transform)
dataloader_test = DataLoader(dataset_test, batch_size=32,
                             shuffle=True, num_workers=0)

model = Model(in_channels=1, channels=32)
unet = UNet(device='cpu',
            model=model)

unet.fit(10, dataloader_train, dataloader_test)
