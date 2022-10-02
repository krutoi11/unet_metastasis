import os
import torch
from skimage import io, transform
import PIL
from torch.utils.data import Dataset
import shutil
import numpy

class BrainDataset(Dataset):

    def __init__(self, root_dir, volume_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.volume = numpy.loadtxt(volume_dir)

    def __len__(self):
        patient_number = len(self.volume)
        return patient_number

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        list_before = []
        list_after = []

        for i in range(256):
            label_before = f'{self.root_dir}/patient{idx+1}/before/seg{i}.png'
            label_fu = f'{self.root_dir}/patient{idx+1}/fu/seg{i}.png'

            label_before = self.transform(PIL.Image.open(label_before).convert('L')).unsqueeze(3)
            label_after = self.transform(PIL.Image.open(label_fu).convert('L')).unsqueeze(3)

            list_before.append(label_before)
            list_after.append(label_after)

        list = torch.cat([torch.cat(list_before, 3), torch.cat(list_after, 3)], 3)

        sample = {'label': list,
                  'volume': self.volume[idx]}

        return sample
