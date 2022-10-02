from dataset import BrainDataset
from model import Model
from train import Unknown
import torchvision
from torch.utils.data import DataLoader

img_size = 256

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize([img_size, img_size])])

dataset_train = BrainDataset(root_dir='../datas/images_28patients_train',
                             volume_dir='../datas/volume_28patients_train.txt',
                             transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=3,
                              shuffle=True, num_workers=0)

dataset_test = BrainDataset(root_dir='../datas/images_6patients_test',
                             volume_dir='../datas/volume_6patients_test.txt',
                             transform=transform)
dataloader_test = DataLoader(dataset_test, batch_size=3,
                              shuffle=True, num_workers=0)

model = Model(in_channels=1, channels=8)

print(sum(p.numel() for p in model.parameters()))

unknown = Unknown(model=model)
unknown.fit(dataloader_train, dataloader_test)
