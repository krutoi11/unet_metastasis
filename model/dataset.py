import os
import torch
from skimage import io, transform
import PIL
from torch.utils.data import Dataset
import shutil
import numpy

"""
class BrainDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        
         Args:
             root_dir (string): Directory with all the images.
             transform (callable, optional): Optional transform to be applied
                 on a sample.
         

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        patient_number = len(os.listdir(self.root_dir)) // 3
        return patient_number

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f'{self.root_dir}/{idx}_image.png'
        label_before = f'{self.root_dir}/{idx}_label_before.png'
        label_fu = f'{self.root_dir}/{idx}_label_fu.png'

        image = PIL.Image.open(img_name).convert('L')
        label_before = PIL.Image.open(label_before).convert('L')
        label_after = PIL.Image.open(label_fu).convert('L')

        sample = {'image': image,
                  'label_before': label_before,
                  'label_after': label_after}

        if self.transform:
            sample = {'image': self.transform(image),
                      'label_before': self.transform(label_before),
                      'label_after': self.transform(label_after)}

        return sample
"""

"""

убираем разбивку по пациентам 

os.mkdir('../images_train')
os.mkdir('../images_test')

k = 0
for i in range(1, 29):
    for j in range(256):
        shutil.copy(f'../images_28/patient{i}/before/image{j}.png', '../images_train')
        old_name = f"../images_train/image{j}.png"
        new_name = f"../images_train/{k}_image.png"
        os.rename(old_name, new_name)
        shutil.copy(f'../images_28/patient{i}/before/seg{j}.png', '../images_train')
        old_name = f"../images_train/seg{j}.png"
        new_name = f"../images_train/{k}_label_before.png"
        os.rename(old_name, new_name)
        shutil.copy(f'../images_28/patient{i}/fu/seg{j}.png', '../images_train')
        old_name = f"../images_train/seg{j}.png"
        new_name = f"../images_train/{k}_label_fu.png"
        os.rename(old_name, new_name)
        k = k+1


n = 0
for l in range(1,7):
    for m in range(256):
        shutil.copy(f'../images_6/patient{l}/before/image{m}.png', '../images_test')
        old_name = f"../images_test/image{m}.png"
        new_name = f"../images_test/{n}_image.png"
        os.rename(old_name, new_name)
        shutil.copy(f'../images_6/patient{l}/before/seg{m}.png', '../images_test')
        old_name = f"../images_test/seg{m}.png"
        new_name = f"../images_test/{n}_label_before.png"
        os.rename(old_name, new_name)
        shutil.copy(f'../images_6/patient{l}/fu/seg{m}.png', '../images_test')
        old_name = f"../images_test/seg{m}.png"
        new_name = f"../images_test/{n}_label_fu.png"
        os.rename(old_name, new_name)
        n = n+1
"""

# убираем слайсы без опухоли

PATH_TO_OLD_TRAIN = '../datas/images_28patients_train'
PATH_TO_OLD_TEST = '../datas/images_6patients_test'
PATH_TO_NEW_TRAIN = '../datas/comb_no_patients_no_empty_train'
PATH_TO_NEW_TEST = '../datas/comb_no_patients_no_empty_test'

os.mkdir(PATH_TO_NEW_TRAIN)
os.mkdir(PATH_TO_NEW_TEST)

k = 0

for i in range(1, 29):
    for j in range(256):
        I_bef = numpy.asarray(PIL.Image.open(PATH_TO_OLD_TRAIN + f'/patient{i}/before/seg{j}.png').convert('L'))
        I_after = numpy.asarray(PIL.Image.open(PATH_TO_OLD_TRAIN + f'/patient{i}/fu/seg{j}.png').convert('L'))
        if numpy.sum(I_bef) != 0 or numpy.sum(I_after) != 0:
            shutil.copy(PATH_TO_OLD_TRAIN + f'/patient{i}/before/comb{j}.png', PATH_TO_NEW_TRAIN)
            old_name = PATH_TO_NEW_TRAIN + f"/comb{j}.png"
            new_name = PATH_TO_NEW_TRAIN + f"/{k}_comb.png"
            os.rename(old_name, new_name)
            shutil.copy(PATH_TO_OLD_TRAIN + f'/patient{i}/before/seg{j}.png', PATH_TO_NEW_TRAIN)
            old_name = PATH_TO_NEW_TRAIN + f"/seg{j}.png"
            new_name = PATH_TO_NEW_TRAIN + f"/{k}_label_before.png"
            os.rename(old_name, new_name)
            shutil.copy(PATH_TO_OLD_TRAIN + f'/patient{i}/fu/seg{j}.png', PATH_TO_NEW_TRAIN)
            old_name = PATH_TO_NEW_TRAIN + f"/seg{j}.png"
            new_name = PATH_TO_NEW_TRAIN + f"/{k}_label_fu.png"
            os.rename(old_name, new_name)
            k = k + 1

n = 0

for l in range(1, 7):
    for m in range(256):
        L_bef = numpy.asarray(PIL.Image.open(PATH_TO_OLD_TEST + f'/patient{l}/before/seg{m}.png').convert('L'))
        L_after = numpy.asarray(PIL.Image.open(PATH_TO_OLD_TEST + f'/patient{l}/fu/seg{m}.png').convert('L'))
        if numpy.sum(L_bef) != 0 or numpy.sum(L_after) != 0:
            shutil.copy(PATH_TO_OLD_TEST + f'/patient{l}/before/comb{m}.png', PATH_TO_NEW_TEST)
            old_name = PATH_TO_NEW_TEST + f"/comb{m}.png"
            new_name = PATH_TO_NEW_TEST + f"/{n}_comb.png"
            os.rename(old_name, new_name)
            shutil.copy(PATH_TO_OLD_TEST + f'/patient{l}/before/seg{m}.png', PATH_TO_NEW_TEST)
            old_name = PATH_TO_NEW_TEST + f"/seg{m}.png"
            new_name = PATH_TO_NEW_TEST + f"/{n}_label_before.png"
            os.rename(old_name, new_name)
            shutil.copy(PATH_TO_OLD_TEST + f'/patient{l}/fu/seg{m}.png', PATH_TO_NEW_TEST)
            old_name = PATH_TO_NEW_TEST + f"/seg{m}.png"
            new_name = PATH_TO_NEW_TEST + f"/{n}_label_fu.png"
            os.rename(old_name, new_name)
            n = n + 1

