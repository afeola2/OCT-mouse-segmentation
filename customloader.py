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
    


class AngioSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size, train=True, return_name=False, resize=True):
        self.img_size = img_size
        self.image_files = image_dir
        self.mask_files= mask_dir
        self.train = train
        self.return_name=return_name
        self.resize=resize

        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize((img_size)),
         #   transforms.RandomCrop((img_size, img_size)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            #transforms.Lambda(lambda x: torch.stack([x, x, x]))  # Stack grayscale image to get 3 channels
        ])

        self.transform_mask= transforms.Compose([
            #transforms.Resize((img_size)),
            transforms.ToTensor()]) 

    def __len__(self):
        return len(self.image_files)
    
    def resize_with_aspect_ratio(self, image, target_size, interpolation=cv2.INTER_CUBIC):
        """
        Resize an image while maintaining its aspect ratio.

        Args:
        - image: Input image (NumPy array).
        - target_size: Desired size for the smaller dimension (int).
        - interpolation: Interpolation method for resizing (default: cv2.INTER_CUBIC).

        Returns:
        - Resized image (NumPy array).
        """
        h, w = image.shape[:2]

        # Calculate aspect ratio
        if h > w:
            new_height = target_size
            new_width = int(w * (target_size / h))
        else:
            new_width = target_size
            new_height = int(h * (target_size / w))
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        return resized_image

    def __getitem__(self, idx):

        image = np.asarray(cv2.imread(self.image_files[idx], cv2.IMREAD_UNCHANGED)).astype(np.float32 )
        mask = np.asarray(cv2.imread(self.mask_files[idx], cv2.IMREAD_UNCHANGED)).astype(np.int64)
        mask = np.where(mask !=0, 1, mask)
        
        if self.resize:
            #Resize with cv2 not torch
            image = self.resize_with_aspect_ratio(image, (self.img_size), interpolation=cv2.INTER_CUBIC)
            h, w = image.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            #mask = self.resize_with_aspect_ratio(mask, (self.img_size), interpolation=cv2.INTER_NEAREST)        
           # print('Here')
           # print(image.shape)
            #print(mask.shape)
        image = self.transform_image(image)
        mask = self.transform_mask(mask)

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
           # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.img_size, self.img_size))
         #   image = TF.crop(image, i, j, h, w)
          #  mask = TF.crop(mask, i, j, h, w)

            #print(image.shape)
            #print(mask.shape)

        # Masks should not have three channels; reduce to one channel
        mask = mask[0, :, :].long()

        if self.return_name:
            return image.squeeze(1), mask.squeeze(), self.image_files[idx], self.mask_files[idx], idx #Mask needs to be W,H , Image needs to be 3, W,H



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
