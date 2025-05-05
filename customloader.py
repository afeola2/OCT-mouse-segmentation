import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

import random
import numpy as np 
import cv2

from utils import *



class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir_c, mask_dir_tt, img_size, train=True, return_name=False):
        self.img_size = img_size
        self.image_files = image_dir
        self.mask_files_c = mask_dir_c
        self.mask_files_tt = mask_dir_tt
        self.train = train
        self.return_name=return_name

        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize((img_size, img_size)),
         #   transforms.RandomCrop((img_size, img_size)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda x: torch.stack([x, x, x]))  # Stack grayscale image to get 3 channels
        ])

        self.transform_mask= transforms.Compose([
            transforms.ToTensor()]) 

    def __len__(self):
        return len(self.image_files)
    


    def __getitem__(self, idx):

        image = np.asarray(cv2.imread(self.image_files[idx], cv2.IMREAD_UNCHANGED)).astype(np.float32 )[1536:]
        mask_choroid = np.asarray(cv2.imread(self.mask_files_c[idx], cv2.IMREAD_UNCHANGED)).astype(np.int64)[1536:]
        mask_TT = np.asarray(cv2.imread(self.mask_files_tt[idx], cv2.IMREAD_UNCHANGED)).astype(np.int64)[1536:]
        #Change mask valeus for CrossEntropy
        mask_choroid = np.where(mask_choroid == 7, 2, mask_choroid)
        mask_choroid = np.where(mask_choroid == 8, 3, mask_choroid)
        mask_TT = np.where(mask_TT == 9, 1, mask_TT)

        mask = mask_choroid + mask_TT

        
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        mask = mask[0, :, :].long()

        if self.train:
            # Apply the same random horizontal flip to both image and mask
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Apply the same random vertical flip to both image and mask
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            # Apply consistent random crop to both image and mask
            #i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.img_size, self.img_size))
            #image = TF.crop(image, i, j, h, w)
            #mask = TF.crop(mask, i, j, h, w)

            #print(image.shape)
            #print(mask.shape)

        # Masks should not have three channels; reduce to one channel
        

        if self.return_name:
            return image.squeeze(1), mask.squeeze(), self.image_files[idx], self.mask_files_tt[idx], idx #Mask needs to be W,H , Image needs to be 3, W,H



        return image.squeeze(1), mask.squeeze() #Mask needs to be W,H , Image needs to be 3, W,H
    




class InferenceDataset(Dataset):
    def __init__(self, image_dir):
        self.image_files = image_dir
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda x: torch.stack([x, x, x]))  # Stack grayscale image to get 3 channels
        ])



    def __len__(self):
        return len(self.image_files)
    


    def __getitem__(self, idx):

        image = np.asarray(cv2.imread(self.image_files[idx], cv2.IMREAD_UNCHANGED)).astype(np.float32)[1536:]     
        image = self.transform_image(image).squeeze(1)
        
        #print(f'Loader {image.shape}')
  
        return image, self.image_files[idx]
