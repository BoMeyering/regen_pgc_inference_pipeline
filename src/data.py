# Dataset and Dataloader
# BoMeyering 2024

import torch
import cv2
from glob import glob
from pathlib import Path
from typing import Union
from torch.utils.data import Dataset, DataLoader
import albumentations as A

from transforms import get_inf_transforms

class InferenceDataset():
    
    def __init__(self, dir: Union[str, Path], img_size: int=1024):
        self.root_dir = Path(dir)
        self.img_size = img_size
        self.transforms = get_inf_transforms(img_size=self.img_size)
        self.filenames = []
        
        # Append all of the image filenames to the filename list
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            file_names = glob("*" + ext, root_dir=self.root_dir)
            self.filenames.extend(file_names)
            
        
    def __getitem__(self, index):
        
        path = self.root_dir / self.filenames[index]
        img = cv2.imread(str(path))
        t_img = self.transforms(image=img)['image']
        
        return t_img, img, img.shape[:2]
    
    def __len__(self):
        return len(self.filenames)
    
def get_inf_loader(dir: Union[str, Path], img_size: int=1024):
    
    dataset = InferenceDataset(
        dir=dir,
        img_size=img_size
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    
    return dataloader
        
        
        
        
        
        
    