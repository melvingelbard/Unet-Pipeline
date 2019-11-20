from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os, sys
import numpy as np
import random


class Nuclei_Dataset(Dataset):
    def __init__(self, image_paths, target_paths, params=None, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.params = params
        self.train = train
        
        self.files = os.listdir(self.image_paths)
        self.labels = os.listdir(self.target_paths)
        self.c = 3
        
        self.cropping_width, self.cropping_height, self.h_flip, self.v_flip, self.normalise = self.unpack_params()
        

    def transform_train(self, inputs, labels):
        
        # Randomly crop the image and annotation in the same place.
        i, j, h, w = transforms.RandomCrop.get_params(inputs, (self.cropping_width, self.cropping_height))
        inputs = TF.crop(inputs, i, j, h, w)
        labels = TF.crop(labels, i, j, h, w)
        
        # Colour transform?
        colour_augmentation=transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        inputs = colour_augmentation(inputs)
        
        
        
        # Flip both the inputs and labels horizontally.
        seed = random.random()
        random.seed(seed)
        
        horizontal_flip = transforms.RandomHorizontalFlip(p=self.h_flip)
        inputs = horizontal_flip(inputs)
        
        random.seed(seed)
        labels = horizontal_flip(labels)
        
        
        
        # Flip both the inputs and labels vertically.
        seed2 = random.random()
        random.seed(seed2)
        
        vertical_flip = transforms.RandomVerticalFlip(p=self.v_flip)
        inputs = vertical_flip(inputs)
        
        random.seed(seed2)
        labels = vertical_flip(labels)
        
        
        
        # Rotate both the inputs and labels by 90, 180 or 270 degrees.
        seed3 = random.random()
        random.seed(seed3)
        
        rotation_transform = MyRotationTransform(angles=[ 0, 90, 180, 270])
        inputs = rotation_transform(inputs)
        
        random.seed(seed3)
        labels = rotation_transform(labels)       
        
        
        # Convert the inputs to tensors and the labels to numpy array.
        inputs = TF.to_tensor(inputs)
        labels = np.asarray(labels)  

        
        if self.normalise:
            # Normalise the inputs
            normalise = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) 
            inputs = normalise(inputs)
        
        
        # Return the transformed inputs/labels.
        return inputs, labels
    
    def transform_eval(self, inputs, labels):
        
        # Randomly crop the image and annotation in the same place.
        i, j, h, w = transforms.RandomCrop.get_params(inputs, (self.cropping_width, self.cropping_height))
        inputs = TF.crop(inputs, i, j, h, w)
        labels = TF.crop(labels, i, j, h, w)
               
        inputs = TF.to_tensor(inputs)
        labels = np.asarray(labels)
        
        if self.normalise:
            # Normalise the inputs
            normalise = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) 
            inputs = normalise(inputs)
        
        return inputs, labels
            

    def __len__(self):
        return len(self.files)
    
    def unpack_params(self):
        cropping_width = self.params["cropping_width"]
        cropping_height = self.params["cropping_height"]
        h_flip = self.params["h_flip"]
        v_flip = self.params["v_flip"]
        normalise = self.params["normalise"]
        
        return cropping_width, cropping_height, h_flip, v_flip, normalise
        
    
    
    
    # The following definition allows us to obtain indexes for the images and annotations.
    def __getitem__(self, idx):
        # Lists of the image and annotation indices.
        img_name = self.files[idx]
        label_name = self.files[idx].replace('.png','-mask.png')          
    
        # Link the images/annotations with their corresponding image.
        inputs = Image.open(os.path.join(self.image_paths, img_name))
        labels = Image.open(os.path.join(self.target_paths, label_name))

        x,y = self.transform_train(inputs, labels) if self.train else self.transform_eval(inputs, labels)
        return x, y
    
    
class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)  