import os
import random

from PIL import Image

file_train = open('../txt/from_to/val_HiUCD.txt',"w")
n = 0
path_train = 'D:/CDDataset/hiucd/val/'

path_train_list = sorted(os.listdir(path_train + 'image/2018/'))
for file_A in path_train_list:
    path_file_A = path_train + "image/2018/" + file_A
    path_file_B = path_train + "image/2019/" + file_A
    path_file_BCD = path_train + "mask_scd/" + file_A
    path_file_label_SGA = path_train + "mask_scd/" + file_A
    path_file_label_SGB = path_train + "mask_scd/" + file_A
    n += 1
    file_train.write(path_file_A + ' ' + path_file_B + ' ' + path_file_BCD + ' ' + path_file_label_SGA + ' ' + path_file_label_SGB + '\n')
print(n)