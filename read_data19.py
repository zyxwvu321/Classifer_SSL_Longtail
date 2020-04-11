# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import shutil
import os
from pathlib import Path
from tqdm import tqdm
#import cv2
#%%
src_im_fd = 'D:/dataset/ISIC/ISIC_2019_Training_Input/'   
tar_im_fd = '../data/train19/'

df = pd.read_csv('D:/dataset/ISIC/ISIC_2019_Training_GroundTruth.csv')

df_v = df.values
img_name = df_v[:,0]

label_np = df_v[:,1:]

#labels = np.zeros_like(img_name)
#
#for idx,v in enumerate(label_np):
#    labels[idx] = np.where(label_np[1]==1)[0][0]
#    
labels = [ np.where(v==1)[0][0] for v in label_np]

dict_label = dict()
for i in range(8):
    dict_label[i] = df.columns[1:][i]
    
    
for val,key in dict_label.items():
    os.makedirs(tar_im_fd +key, exist_ok =True)
    


for idx,fn in enumerate(tqdm(img_name)):
    src_fn = Path(src_im_fd)/(fn + '.jpg')
    tar_fn = Path(tar_im_fd)/ dict_label[labels[idx]]/(fn + '.jpg') 
    if os.path.exists(str(src_fn)):
        shutil.copyfile(src_fn,tar_fn)
    else:
        print(f'filename {str(src_fn)} not exist')
# #%% write test
# df = pd.read_csv('./data/ISIC/ISIC2018_Task3_Testing_Score_imb.csv')
# tar_im_fd = './data/ISIC/test18/'
# src_im_fd = '/home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input/'
# for val,key in dict_label.items():
#     os.makedirs(tar_im_fd +key, exist_ok =True)

# df_v = df.values
# img_name = df_v[:,0]
# label_np = df_v[:,1:]
# labels = [ np.where(v==v.max())[0][0] for v in label_np]
# for idx,fn in enumerate(tqdm(img_name)):
#     src_fn = Path(src_im_fd)/(fn + '.jpg')
#     tar_fn = Path(tar_im_fd)/ dict_label[labels[idx]]/(fn + '.jpg') 
#     if os.path.exists(str(src_fn)):
#         shutil.copyfile(src_fn,tar_fn)
#     else:
#         print(f'filename {str(src_fn)} not exist')

# #%% read ISIC19 data
        
# src_im_fd = '/home/minjie/dataset/ISIC/ISIC_2019_Training_Input/'
# tar_im_fd = './data/ISIC/train19/'

# df = pd.read_csv('./data/ISIC/ISIC_2019_Training_GroundTruth.csv')

# df_v = df.values
# img_name = df_v[:,0]

# label_np = df_v[:,1:]

# #labels = np.zeros_like(img_name)
# #
# #for idx,v in enumerate(label_np):
# #    labels[idx] = np.where(label_np[1]==1)[0][0]
# #    
# labels = [ np.where(v==1)[0][0] for v in label_np]

# dict_label = dict()
# n_label = len(df.columns)-2
# for i in range(n_label):
#     dict_label[i] = df.columns[1:][i]
    
# dict_label[3]= 'AKIEC'
    
# for val,key in dict_label.items():
#     os.makedirs(tar_im_fd +key, exist_ok =True)
    


# for idx,fn in enumerate(tqdm(img_name)):
#     src_fn = Path(src_im_fd)/(fn + '.jpg')
#     tar_fn = Path(tar_im_fd)/ dict_label[labels[idx]]/(fn + '.jpg') 
#     if os.path.exists(str(src_fn)):
#         #img = cv2.imread(str(src_fn))
#         #img_resize = cv2.resize(img,(600,450))
#         #cv2.imwrite(str(tar_fn),img_resize)
#         shutil.copyfile(src_fn,tar_fn)
#     else:
#         print(f'filename {str(src_fn)} not exist')
        