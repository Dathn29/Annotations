import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Dataset(BaseDataset):    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
                
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(64,64))
      
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask,(64,64))
        masks = [(mask == 1)] 
        mask = np.stack(masks, axis=-1).astype('float')
        
    
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)