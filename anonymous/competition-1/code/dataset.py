from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2

"""
AnonymousDataset: 18개의 레이블 또는 각각의 레이블을 반환하는 데이터셋 -> 적합한 loss는 crossentropy 계열
AnonymousMultilabelDataset: 8개의 레이블을 반환하는 데이터셋 -> 적합한 loss는 multilabel margin loss 계열

필수요소:
    - datalist: 데이터셋에 호출할 파일의 경로
    - transforms: 데이터 호출시 적용한 augmentation
    - label_type: 데이터에 대응하는 레이블의 종류
    - age_range: age labeling rule

A: 나이
B: 마스크
C: 성별

"""

# filepath를 받아 B label 반환
def get_B_label(filepath):
    image_name = filepath.split("/")[-1]
    if "incorrect" in image_name:
        return 1
    elif "normal" in image_name:
        return 2
    elif "mask" in image_name:
        return 0
    else:
        raise ValueError(f"No class for {image_name}")

# filepath를 받아 C label 반환
def get_C_label(filepath):
    profile = filepath.split("/")[-2]
    _, B, _, _ = profile.split("_")
    return 0 if "fe" in B else 1

# A range를 받아 A label 반환하는 함수를 반환
def get_A_label(A_range = [(0, 29), (30, 59), (60, 200)]):
    def A_label(filepath):
        profile = filepath.split("/")[-2]
        _, _, _, A = profile.split("_")
        A = int(A)
        label = -1
        for idx, A_bound in enumerate(A_range):
            if (A_bound[0]<=A) and (A<=A_bound[1]):
                label = idx
                break
        if label==-1:
            raise Exception("범위에 포함되지 않은 값")
        return label
    return A_label

# A range를 받아 원래 18개 클래스 계열의 label 반환하는 함수를 반환
def get_label_by_18(A_range = [(0, 29), (30, 59), (60, 200)]):
    def label_by_18(filepath):
        B = get_B_label(filepath)
        C = get_C_label(filepath)
        A = get_A_label(A_range)(filepath)
        return B*(2*len(A_range)) + C*len(A_range) + A
    return label_by_18

# age range를 받아 원래 multilabel 반환하는 함수를 반환
def get_multilabel(A_range = [(0, 29), (30, 59), (60, 200)]):
    def multilabel(filepath):
        B = get_B_label(filepath)
        C = get_C_label(filepath)
        A = get_A_label(A_range)(filepath)
        res = [0]*(4+len(A_range))
        res[0] = C
        res[1+B] = 1
        res[4+A] = 1
        return torch.Tensor(res)
    return multilabel

# A range를 받아 A multilabel 반환하는 함수를 반환
def get_A_multilabel(A_range = [(0, 29), (30, 59), (60, 200)]):
    def A_multilabel(filepath):
        age = get_A_label(A_range)(filepath)
        res = [0]*len(A_range)
        res[A] = 1
        return torch.Tensor(res)
    return A_multilabel

# 사용될 augmentations
def get_transforms(need=('train', 'val')):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformations = {}
    if 'train' in need:
        transformations['train'] = Compose(
            [
                CenterCrop(384, 384, p=1.0),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(p=0.7),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.7),
                RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.7),
                Normalize(mean=mean, std=std, max_pixel_value=225.0, p=1.0),
                OneOf(
                    [
                        RandomGridShuffle((2, 2), p = 1.0),
                        RandomGridShuffle((4, 2), p = 1.0),
                    ],
                    p=0.8
                ),
                CoarseDropout(p=0.3),
                Cutout(p=0.7),
                ToTensorV2(p=1.0)
            ],
            p=1.0
        )
    if 'val' in need:
        transformations['val'] = Compose(
            [
                CenterCrop(384, 384, p=1.0),
                Normalize(mean=mean, std=std, max_pixel_value=225.0, p=1.0),
                ToTensorV2(p=1.0)
            ],
            p=1.0
        )
    return transformations


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


class AnonymousDataset(Dataset):
    """
    data_paths, label_type, transformations
    """
    def __init__(self, data_paths, label_type, transformations = None):
        self.data_paths = data_paths
        self.transformations = transformations
        self.labels = [label_type(filepath) for filepath in self.data_paths]
    
    def __getitem__(self, idx):
        image_path = self.data_paths[idx]
        image = get_img(image_path)

        if self.transformations:
            image = self.transformations(image=image)["image"]
        label = self.labels[idx]

        return image, label
    
    def __len__(self):
        return len(self.data_paths)


class AnonymousMultilabelDataset(Dataset):
    """
    data_paths, label_type, transformations
    """
    def __init__(self, data_paths, label_type, transformations = None):
        self.transformations = transformations
        self.data_paths = data_paths
        self.label_type = label_type
        
    def __getitem__(self, idx):
        image_path = self.data_paths[idx]
        image = get_img(image_path)
        
        if self.transforms:
            image = self.transformations(image=image)["image"]
        label = self.label_type(self.data_paths[idx])
        return image, label
    
    def __len__(self):
        return len(self.data_paths)

