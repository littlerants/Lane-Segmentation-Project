# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 20:10:30 2019

@author: Administrator
"""
import configs as cf
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, filters, util
import numpy as np
import torch
from torchvision import transforms, utils
import cv2
from label_process import encode_labels, decode_labels, decode_color_labels

import torchvision
import os
from tqdm import tqdm
# from imgaug import augmenters as iaa
def crop_resize_data(image, label = None, image_size=(768,256), offset=690):
    re_image = image[offset:, :]
    if label is not None:
        re_label = label[offset:, :]
        train_image = cv2.resize(re_image, image_size, interpolation=cv2.INTER_LINEAR)
        train_label = cv2.resize(re_label, image_size, interpolation=cv2.INTER_NEAREST)
        return train_image,train_label
    else:
        train_image = cv2.resize(re_image, image_size, interpolation=cv2.INTER_LINEAR)
        return train_image
class MyData(Dataset):
    def __init__(self, dir, csv_file, transforms=None):
        self.full_name = dir + csv_file
        self.landmarks_frame = pd.read_csv(dir + csv_file)
        self.root_dir1 = dir
        self.transforms = transforms

    def __len__(self):
        return len(self.landmarks_frame)
    def getitem(self, idx):
        image_name = self.landmarks_frame.iloc[idx, 1]
        label_name = self.landmarks_frame.iloc[idx, 2]
        image = io.imread(image_name)
        label = io.imread(label_name)
        image, label = crop_resize_data(image, label)
        #plt.figure('image')
        #plt.imshow(label)
        #plt.show()
        label = encode_labels(label)
        sample = {'image': image, 'label': label}
        if self.transforms:
            sample = self.transforms(sample)
        return sample
    def __getitem__(self, idex):
        sample = self.getitem(idex)
        return sample
class ImageAug(object):
    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        if np.random.uniform(0, 1) > 0.7:

            image = filters.gaussian(image, sigma=0.3, multichannel=False)
            #image = filters.sobel(image)
            image = util.random_noise(image, mode='gaussian')
            sample = {'image': image, 'label': mask}
        return sample

class DeformAug(object):
    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        sample = {'image': image, 'label': mask}
        return sample

class ScaleAug(object):
    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        scale = np.random.uniform(0.6, 1.5)
        h, w, _ = image.shape
        aug_image = image.copy()
        aug_mask = mask.copy()

        aug_image = cv2.resize(aug_image, (int(w * scale),int(h * scale)))
        aug_mask = cv2.resize(aug_mask, (int(w * scale),int(h * scale)))

        if scale < 1.0:
            new_h, new_w = aug_image.shape[:2]
            pre_h_crop = int((h - new_h) / 2)
            pre_w_crop = int((w - new_w)/2)
            pad_list = [[pre_h_crop, h - new_h - pre_h_crop], [pre_w_crop, w - new_w - pre_w_crop],[0,0]]
            aug_image = np.pad(aug_image, pad_list, mode='constant')
            aug_mask = np.pad(aug_mask, pad_list[:2], mode='constant')
        if scale > 1.0:
            new_h, new_w,_ = aug_image.shape
            pre_h_crop = int((new_h - h)/2)
            pre_w_crop = int((new_w - w)/2)
            host_h = pre_h_crop + h
            host_w = pre_w_crop + w
            aug_image = aug_image[pre_h_crop:host_h, pre_w_crop:host_w]
            aug_mask = aug_mask[pre_h_crop:host_h, pre_w_crop:host_w]
        sample = {'image': aug_image, 'label': aug_mask}

        return sample

class CutOut(object):
    def __init__(self, mask_size, p):
        self.mask_size = mask_size
        self.p = p
    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        mask_size_half = self.mask_size//2
        offset = 1 if self.mask_size % 2 == 0 else 0
        h, w, _ = image.shape
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax) #中心点
        xmin, ymin = cx - mask_size_half, cy - mask_size_half
        xmax, ymax = xmin + self.mask_size, ymin + self.mask_size
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), max(0, xmax), max(0, ymax)

        if (np.random.uniform(0, 1) < self.p):
            image[ymin:ymax, xmin:xmax] = (0, 0, 0)
        sample = {'image': image, 'label': mask}

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.int32)
        mask = mask.astype(np.uint8)
        sample = {'image': torch.from_numpy(image.copy()),
                  'label': torch.from_numpy(mask.copy())}
        return sample

# the 'main()' function is test program, you can ignore
def main():
    dataset = MyData(cf.params['root_dir'], cf.params['train_csv'], transforms=transforms.Compose([ImageAug(),ScaleAug(),
                                                                    CutOut(64, 0.8)]))
    for i in range(10,20):
        print(i)
        sample = dataset[i]
        img = sample['image']
        mask = sample['label']
        plt.figure('image')
        plt.imshow(img)
        plt.show()
        plt.figure('image')
        plt.imshow(mask)
        plt.show()

if __name__ == '__main__':
    main()

