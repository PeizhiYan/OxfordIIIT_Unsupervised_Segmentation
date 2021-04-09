############################################
##
## Preprocess the Oxford-IIIT Pet Dataset
##
## Author:  Peizhi Yan
##   Date:  Feb. 16, 2021
##
############################################

import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

source_path = './data/original/'
output_path = './data/224x224/'

def read_meta_file(file_path):
    meta_info = {} # File_Name, Class (1-37), Species (1 Cat, 2 Dog), Breed (1-25:Cat 1:12:Dog)
    with open(file_path) as meta_file:
        while True:
            line = meta_file.readline()
            if not line:
                break
            tmp = line.split(' ')
            meta_info[tmp[0]] = [int(tmp[1]),int(tmp[2]),int(tmp[3])]
    return meta_info

def print_meta_head(meta):
    print("Total Entries: ", len(meta))
    for key in list(meta.keys())[:5]:
        print(key, meta[key])
    print()


""" read the meta information """
meta_info_all = read_meta_file(source_path + 'annotations/list.txt')
meta_info_test = read_meta_file(source_path + 'annotations/test.txt')
print_meta_head(meta_info_all)
print_meta_head(meta_info_test)


""" get Train/Val/Test lists """
test_list = list(meta_info_test.keys()) # list of testing file names
train_list = []
val_list = []
train_val_list = []
for key in list(meta_info_all.keys()):
    if key not in test_list:
        train_val_list.append(key)
category_id = 1
temp_list = []
for key in train_val_list:
    current_category_id = meta_info_all[key][0]
    if current_category_id != category_id:
        category_id = current_category_id
        # Split train/val data
        pivot = int(len(temp_list)/2)
        for kk in temp_list[:pivot]:
            train_list.append(kk)
        for kk in temp_list[pivot:]:
            val_list.append(kk)
        temp_list = []
    else:
        temp_list.append(key)
print('train \t', len(train_list))
print('val \t', len(val_list))
print('test \t', len(test_list))


""" preprocess images and annotations """
dd = {'train': train_list, 'val': val_list, 'test': test_list}
for ddtype in list(dd.keys()):
    ddlist = dd[ddtype] 
    for key in ddlist:
        # image
        img = cv2.imread(source_path + 'images/' + key + '.jpg')
        img = cv2.resize(img, (224,224))
        cv2.imwrite(output_path + ddtype + '/' + key + '.jpg', img) # should use BGR
        # mask
        tmap = cv2.imread(source_path + 'annotations/trimaps/' + key + '.png', 0)
        tmap = cv2.resize(tmap, (224,224))
        mask = np.zeros([tmap.shape[0],tmap.shape[1],3], dtype=int) # channels: foreground, background, boundary
        mask[:,:,0] = np.where(tmap == 1, 1, 0) # Foreground
        mask[:,:,1] = np.where(tmap == 2, 1, 0) # Background
        mask[:,:,2] = np.where(tmap == 3, 1, 0) # Boundary
        np.save(output_path + ddtype + '/_' + key + '.npy', mask)


print('done')