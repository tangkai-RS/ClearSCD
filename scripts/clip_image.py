import os
from PIL import Image
from tqdm import tqdm
from osgeo import gdal


def crop_img(image_path, save_floder, size, stride): 
    # fp = open(image_path,'rb')
    # image = np.array(Image.open(fp))
    # height, width = image.shape[0:2]
    # if len(image.shape) == 3: 
    #     image = image.transpose(2, 0, 1)
        
    image = gdal.Open(image_path).ReadAsArray() 
    if len(image.shape) == 3:
        height, width = image.shape[1:3]
    else:
        height, width = image.shape
    
    if not os.path.exists(save_floder):
        os.makedirs(save_floder)
    imagename_or = os.path.basename(image_path)
    idx = 0

    for i in range(0, height-size+1, stride):
        for j in range(0, width-size+1, stride):
            
            if len(image.shape) == 3:
                image_crop = image[:, i:i+size, j:j+size]
                Image.fromarray(image_crop.transpose(1, 2, 0), 'RGB').save(save_floder + '/%d_%s' % (idx, imagename_or))
            else:
                image_crop = image[i:i+size, j:j+size]
                Image.fromarray(image_crop).save(save_floder + '/%d_%s' % (idx, imagename_or))
            idx += 1
            
    if (i+size - width) < 0:
        m = 1
        for i in range(0, height-size+1, stride):
            
            if len(image.shape) == 3:
                image_crop = image[:, i:i+size, width-size:width]
                Image.fromarray(image_crop.transpose(1, 2, 0), 'RGB').save(save_floder + '/%d_%s' % (idx, imagename_or))
            else:
                image_crop = image[i:i+size, width-size:width]
                Image.fromarray(image_crop).save(save_floder + '/%d_%s' % (idx, imagename_or))
            idx += 1
    else:
        m = 0
    
    if (j+size - height) < 0:
        n = 1
        for j in range(0, width-size+1, stride):
            
            if len(image.shape) == 3:
                image_crop = image[:, height-size:height, j:j+size]
                Image.fromarray(image_crop.transpose(1, 2, 0), 'RGB').save(save_floder + '/%d_%s' % (idx, imagename_or))
            else:
                image_crop = image[height-size:height, j:j+size]
                Image.fromarray(image_crop).save(save_floder + '/%d_%s' % (idx, imagename_or))
            idx += 1
    else:
        n = 0
            
    if (m == 1) & (n == 1):
        if len(image.shape) == 3:
            image_crop = image[:, height-size:height, width-size:width]
            Image.fromarray(image_crop.transpose(1, 2, 0), 'RGB').save(save_floder + '/%d_%s' % (idx, imagename_or))
        else:
            image_crop = image[height-size:height, width-size:width]
            Image.fromarray(image_crop).save(save_floder + '/%d_%s' % (idx, imagename_or))
        idx += 1
    # fp.close()


if __name__ == '__main__':    
    size = 512
    stride = 512
    suffix = 'png'
    image_floders = [
                     r'D:\CDDataset\hiucd_mini\train\2017\9',
                     r'D:\CDDataset\hiucd_mini\train\2018\9',
                     r'D:\CDDataset\hiucd_mini\train\mask_merge\2017_2018\9',
                     r'D:\CDDataset\hiucd_mini\val\2017\9',
                     r'D:\CDDataset\hiucd_mini\val\2018\9',
                     r'D:\CDDataset\hiucd_mini\val\mask_merge\2017_2018\9',
                     r'D:\CDDataset\hiucd_mini\test\2018\9',
                     r'D:\CDDataset\hiucd_mini\test\2019\9',
                     r'D:\CDDataset\hiucd_mini\test\mask_merge\2018_2019\9',
                    ] 
    save_floders = [
                    r'D:\CDDataset\hiucd_mini\train\2017_512',
                    r'D:\CDDataset\hiucd_mini\train\2018_512',
                    r'D:\CDDataset\hiucd_mini\train\mask_512',
                    r'D:\CDDataset\hiucd_mini\val\2017_512',
                    r'D:\CDDataset\hiucd_mini\val\2018_512',
                    r'D:\CDDataset\hiucd_mini\val\mask_512',
                    r'D:\CDDataset\hiucd_mini\test\2018_512',
                    r'D:\CDDataset\hiucd_mini\test\2019_512',
                    r'D:\CDDataset\hiucd_mini\test\mask_512',                    
                   ] 
    
    for image_floder, save_floder in zip(image_floders, save_floders):
        img_names = os.listdir(image_floder)
        if not os.path.exists(save_floder):
            os.makedirs(save_floder)
        
        for img_name in tqdm(img_names):
            if img_name.endswith(suffix):
                img_path = os.path.join(image_floder, img_name)
                crop_img(img_path, save_floder, size, stride) 