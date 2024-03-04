import os
import sys
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

from osgeo import gdal
from PIL import Image
from torch.utils.data import Dataset

import dataset.datatransforms_scd as tr
import numpy as np


def full_path_loader_for_txt(txt_path):
    dataset = {}

    with open(txt_path, "r") as f1:
        lines = f1.read().splitlines()
    A_path = []
    B_path = []
    label_SCD_path = []
    label_SGA_path = []
    label_SGB_path = []
    
    for index, line in enumerate(lines):
        path = line.split(' ')
        img_A = path[0]
        img_B = path[1]
        label_SCD = path[2]
        label_SGA = path[3]
        label_SGB = path[4]
        assert os.path.isfile(img_A)
        assert os.path.isfile(img_B)
        assert os.path.isfile(label_SCD)
        assert os.path.isfile(label_SGA)
        assert os.path.isfile(label_SGB)
        A_path.append(img_A)
        B_path.append(img_B)
        label_SCD_path.append(label_SCD)
        label_SGA_path.append(label_SGA)
        label_SGB_path.append(label_SGB)
        dataset[index] = {'img_A': A_path[index],
                          'img_B': B_path[index],
                          'label_SCD': label_SCD_path[index],
                          'label_SGA': label_SGA_path[index],
                          'label_SGB': label_SGB_path[index]
                          }
    return dataset


class SECONDDataset_SCD(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.split = split
        if split == 'train':  
            self.full_load = full_path_loader_for_txt(args.train_txt_path)
        elif split =='val':
            self.full_load = full_path_loader_for_txt(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_for_txt(args.test_txt_path)
        ST_CLASSES = ['sports field', 'low vegetation', 'ground', 'water', 'building', 'tree',  'unchanged']
        self.class_rgb_values = [[255, 0, 0], [0, 128, 0], [128, 128, 128], [0, 0, 255], [128, 0, 0], [0, 255, 0], [255, 255, 255]] # [255, 255, 255] is unchanged background equal 6
        
    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        img_B_path = self.full_load[idx]['img_B']
        label_SCD_path = self.full_load[idx]['label_SCD']
        
        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        label_SCD = Image.open(label_SCD_path)
            
        sample = {'img_A': img_A,
                  'img_B': img_B,
                  'label_SCD': label_SCD,
                  }
        
        train_transforms = tr.get_train_transforms(seg_ignore_value=37, with_colorjit=self.with_colorjit)
        if not self.pretrained:
            train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
            train_transforms.transforms[5].std = (0.5, 0.5, 0.5)              
        if self.split == 'train':       
            sample = train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = tr.test_transforms(sample)
            sample['name'] = img_A_path
        return sample
    

class HiUCDDataset_SCD(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.split = split
        if split == 'train':  
            self.full_load = full_path_loader_for_txt(args.train_txt_path)
        elif split =='val':
            self.full_load = full_path_loader_for_txt(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_for_txt(args.test_txt_path)
        
    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        img_B_path = self.full_load[idx]['img_B']
        label_path = self.full_load[idx]['label_SCD']

        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        label_SCD = Image.open(label_path)
            
        sample = {'img_A': img_A,
                  'img_B': img_B,
                  'label_SCD': label_SCD,
                  }
        
        train_transforms = tr.get_train_transforms(seg_ignore_value=82, with_colorjit=self.with_colorjit)
        if not self.pretrained :
            train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
            train_transforms.transforms[5].std = (0.5, 0.5, 0.5)              
        if self.split == 'train':       
            sample = train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = tr.test_transforms(sample)
            sample['name'] = img_A_path
        return sample


class NanjingDataset_SCD(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.split = split
        if split == 'train':  
            self.full_load = full_path_loader_for_txt(args.train_txt_path)
        elif split =='val':
            self.full_load = full_path_loader_for_txt(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_for_txt(args.test_txt_path)
        
    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        img_B_path = self.full_load[idx]['img_B']
        label_path = self.full_load[idx]['label_SCD']

        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        label_SCD = Image.open(label_path)
            
        sample = {'img_A': img_A,
                  'img_B': img_B,
                  'label_SCD': label_SCD,
                  }
        
        train_transforms = tr.get_train_transforms(seg_ignore_value=50, with_colorjit=self.with_colorjit)
        if not self.pretrained :
            train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
            train_transforms.transforms[5].std = (0.5, 0.5, 0.5)              
        if self.split == 'train':       
            sample = train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = tr.test_transforms(sample)
            sample['name'] = img_A_path
        return sample

  
if __name__ == '__main__':
    
    pass