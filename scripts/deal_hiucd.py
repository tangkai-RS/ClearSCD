import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from osgeo import gdal


image_floder = r'F:\Hi-UCD-S\Hi-UCD-S-upload\test\mask\2018_2019'
save_floder = r'D:\CDDataset\hiucd\test\mask'

img_names = os.listdir(image_floder)

for img_name in tqdm(img_names):
    if img_name.endswith('png'):
        img_path = os.path.join(image_floder, img_name)
        
        img = np.asarray(Image.open(img_path))
        img_new = img.copy()
        img_new[:, :, 0:2] = img[:, :, 0:2] - 1
        img_new[:, :, 0:2][img[:, :, 0:2] == 0] = 9 # unlabeled equal num
               
        img_new[:, :, -1] = img[:, :, -1] - 1
        img_new[:, :, -1][img[:, :, -1] == 0] = 255 # unlabeled equal 255
        
        img_path = os.path.join(save_floder, img_name)
        Image.fromarray(img_new).save(img_path)
        

        