import os
import sys
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

from osgeo import gdal
from PIL import Image
from torch.utils.data import Dataset

DEBUG = False
if DEBUG:
    import datatransforms as tr
    import datatransforms_seg as tr_seg
    import datatransforms_single_cd as tr_single_cd
else:
    import dataset.datatransforms as tr
    import dataset.datatransforms_seg as tr_seg
    import dataset.datatransforms_single_cd as tr_single_cd

import numpy as np


def full_path_loader_for_txt(txt_path):
    dataset = {}

    with open(txt_path, "r") as f1:
        lines = f1.read().splitlines()
    A_path = []
    B_path = []
    label_BCD_path = []
    label_SGA_path = []
    label_SGB_path = []
    
    for index, line in enumerate(lines):
        path = line.split(' ')
        img_A = path[0]
        img_B = path[1]
        label_BCD = path[2]
        label_SGA = path[3]
        label_SGB = path[4]
        # assert os.path.isfile(img_A)
        # assert os.path.isfile(img_B)
        # assert os.path.isfile(label_BCD)
        # assert os.path.isfile(label_SGA)
        # assert os.path.isfile(label_SGB)
        A_path.append(img_A)
        B_path.append(img_B)
        label_BCD_path.append(label_BCD)
        label_SGA_path.append(label_SGA)
        label_SGB_path.append(label_SGB)
        dataset[index] = {'img_A': A_path[index],
                          'img_B': B_path[index],
                          'label_BCD': label_BCD_path[index],
                          'label_SGA': label_SGA_path[index],
                          'label_SGB': label_SGB_path[index]
                          }
    return dataset


class HiUCDDataset(Dataset):
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
        
        self.train_transforms = tr.get_train_transforms(seg_ignore_value=9, with_colorjit=self.with_colorjit)
        self.test_transforms = tr.test_transforms
        if not self.pretrained:
            self.train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
            self.train_transforms.transforms[5].std = (0.5, 0.5, 0.5)           
        
    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        img_B_path = self.full_load[idx]['img_B']
        label_path = self.full_load[idx]['label_BCD']

        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        label_array = np.asarray(Image.open(label_path))
        label_SGA = self.convert_numpy_to_Image(label_array, 0)
        label_SGB = self.convert_numpy_to_Image(label_array, 1)
        label_BCD = self.convert_numpy_to_Image(label_array, 2)
            
        sample = {'img_A': img_A,
                  'img_B': img_B,
                  'label_BCD': label_BCD,
                  'label_SGA': label_SGA,
                  'label_SGB': label_SGB
                  }
                   
        if self.split == 'train':       
            sample = self.train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = self.test_transforms(sample)
            sample['name'] = img_A_path
        return sample
    
    def convert_numpy_to_Image(self, numpy_array, channel):
        return Image.fromarray(numpy_array[:, :, channel])
  

class NanjingDataset(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.seg_ignore_value = args.num_segclass
        self.split = split
        if split == 'train':  
            self.full_load = full_path_loader_for_txt(args.train_txt_path)
        elif split =='val':
            self.full_load = full_path_loader_for_txt(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_for_txt(args.test_txt_path)
        
        self.train_transforms = tr.get_train_transforms(seg_ignore_value=self.seg_ignore_value, with_colorjit=self.with_colorjit)
        self.test_transforms = tr.test_transforms
        if not self.pretrained:
            self.train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
            self.train_transforms.transforms[5].std = (0.5, 0.5, 0.5)           
        
    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        img_B_path = self.full_load[idx]['img_B']
        label_BCD_path = self.full_load[idx]['label_BCD']
        label_SGA_path = self.full_load[idx]['label_SGA']
        label_SGB_path = self.full_load[idx]['label_SGB']

        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        label_SGA = Image.open(label_SGA_path)
        label_SGB = Image.open(label_SGB_path)
        label_BCD = Image.open(label_BCD_path)
            
        sample = {'img_A': img_A,
                  'img_B': img_B,
                  'label_BCD': label_BCD,
                  'label_SGA': label_SGA,
                  'label_SGB': label_SGB
                  }
                   
        if self.split == 'train':       
            sample = self.train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = self.test_transforms(sample)
            sample['name'] = img_A_path
        return sample


class LoveDADataset(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.seg_ignore_value = args.num_segclass
        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375) 
        self.split = split
        if split == 'train':  
            self.full_load = full_path_loader_for_txt(args.train_txt_path)
        elif split =='val':
            self.full_load = full_path_loader_for_txt(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_for_txt(args.test_txt_path)
       
        if self.pretrained:
            div_255 = True
        else:
            div_255 = False
        
        self.train_transforms = tr_seg.get_train_transforms(div_255=div_255, seg_ignore_value=self.seg_ignore_value, with_colorjit=self.with_colorjit)
        self.test_transforms = tr_seg.test_transforms
        
        if (not self.pretrained) and (not self.with_colorjit):
            idx = 6
            self.train_transforms.transforms[idx].mean = self.mean
            self.train_transforms.transforms[idx].std = self.std
            self.test_transforms.transforms[1].mean = self.mean
            self.test_transforms.transforms[1].std = self.std 
        elif (not self.pretrained) and self.with_colorjit:
            idx = 7
            self.train_transforms.transforms[idx].mean = self.mean
            self.train_transforms.transforms[idx].std = self.std   
            self.test_transforms.transforms[1].mean = self.mean
            self.test_transforms.transforms[1].std = self.std        
        
    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        label_SGA_path = self.full_load[idx]['label_SGA']
        
        img_A = Image.open(img_A_path)
        label_SGA = Image.open(label_SGA_path)
            
        sample = {'img_A': img_A,
                  'label_SGA': label_SGA,
                  }
                
        if self.split == 'train':       
            sample = self.train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = self.test_transforms(sample)
            sample['name'] = img_A_path
        return sample


class LoveDADataset_for_SCD(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.seg_ignore_value = args.num_segclass
        self.split = split
        if split == 'train':  
            self.full_load = full_path_loader_for_txt(args.train_txt_path)
        elif split =='val':
            self.full_load = full_path_loader_for_txt(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_for_txt(args.test_txt_path)
            
        self.train_transforms = tr_single_cd.get_train_transforms(seg_ignore_value=self.seg_ignore_value, with_colorjit=self.with_colorjit)      
        self.test_transforms = tr_single_cd.test_transforms  
        self.test_transforms.transforms[-1].seg_ignore_value = self.seg_ignore_value
        self.test_transforms.transforms[1].fillcolor = self.seg_ignore_value
        
    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        img_B_path = self.full_load[idx]['img_B']
        label_SGA_path = self.full_load[idx]['label_SGA']
        label_SGB_path = self.full_load[idx]['label_SGB']

        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        label_SGA = Image.open(label_SGA_path)
        label_SGB = Image.open(label_SGB_path)
            
        sample = {'img_A': img_A,
                  'img_B': img_B,
                  'label_SGA': label_SGA,
                  'label_SGB': label_SGB
                  }

        if self.split == 'train':       
            sample = self.train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = self.test_transforms(sample)
            sample['name'] = img_A_path
        return sample
         
  
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from utils.parser import get_parser_with_args_from_json
    
    args = get_parser_with_args_from_json(r'F:\ContrastiveLearningCD\configs\test.json')
    
    train_dataset = LoveDADataset_for_SCD(args)
      
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=None)
    print(train_dataloader.__len__())
    
    sample = next(iter(train_dataloader))
    
    colors = dict((
                    (1, (255, 255, 255, 255)), # 前三位RGB，255代表256色
                    (0, (0, 0, 0, 255)),  
                    (2, (255, 0, 0, 255)),  
                ))
    for k in colors:
        v = colors[k]
        _v = [_v / 255.0 for _v in v]
        colors[k] = _v
    index_colors = [colors[key] if key in colors else
                    (255, 255, 255, 0) for key in range(0, len(colors))]
    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', len(index_colors))
    
    _, axes = plt.subplots(1, 5)
    axes[0].imshow(sample['img_A'].cpu().numpy()[0, ...].transpose(1, 2, 0))
    axes[1].imshow(sample['img_B'].cpu().numpy()[0, ...].transpose(1, 2, 0))
    axes[2].imshow(sample['label_SGA'].cpu().numpy()[0, ...])
    axes[3].imshow(sample['label_SGB'].cpu().numpy()[0, ...])
    
    label_BCD = sample['label_BCD'].cpu().numpy()[0, ...]
    label_BCD[label_BCD == 255] = 2
    axes[4].imshow(label_BCD, cmap=cmap)
    plt.show()
    print(1)