import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader 


from albumentations.pytorch import ToTensorV2
from albumentations import *


class CustomDataset(Dataset): 
    def __init__(self, data_paths, labels, is_test=False, transforms=None): 
        self.data_paths = data_paths
        self.labels = labels
        self.is_test = is_test
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.data_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        if self.is_test:
            return img, img_path
        else:
            return img, self.labels[index], img_path

    def __len__(self): 
        return len(self.data_paths)

# 사용될 augmentations
def get_transforms(need=('train', 'val')):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transforms = {}
    if 'train' in need:
        transforms['train'] = Compose(
            [
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                CoarseDropout(p=0.3),
                Cutout(p=0.3),
                ToTensorV2(p=1.0)
            ],
            p=1.0
        )
    if 'val' in need:
        transforms['val'] = Compose(
            [
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)
            ],
            p=1.0
        )
    return transforms

