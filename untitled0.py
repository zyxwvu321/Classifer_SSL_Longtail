# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:34:29 2020

@author: cmj
"""

# test part

%run test_gen_classifier.py --config_file ./configs/resnet50.yaml MISC.TTA True MISC.N_TTA 10

# test testset
%run test_gen_classifier.py --config_file ./configs/resnet50.yaml    DATASETS.ROOT_DIR  ../data/ISIC18/task3/ISIC2018_Task3_Test_Input_coloradj  DATASETS.INFO_CSV ./dat/ISIC18_info_test.csv  MISC.TTA True MISC.N_TTA 10 MISC.ONLY_TEST True 




%run  train_ISIC_gllcmeta.py --config_file  ./configs/resnet50_bone_meta.yaml  DATALOADER.NUM_WORKERS 4 DATALOADER.BATCH_SIZE 8
%run  -d train_ISIC_gllcmeta.py --config_file  ./configs/resnet50_att.yaml  DATALOADER.NUM_WORKERS 0 DATALOADER.BATCH_SIZE 4 DATASETS.ROOT_DIR  ../data/ISIC18/task3/ISIC2018_Task3_Training_Input

%run  train_ISIC_gllcmeta.py --config_file  ./configs/resnet50_meta.yaml  DATALOADER.NUM_WORKERS 4 DATALOADER.BATCH_SIZE 16 
%run -d train_ISIC_gllcmeta.py --config_file  ./configs/resnet50_meta.yaml  DATALOADER.NUM_WORKERS 0 DATALOADER.BATCH_SIZE 4 DATASETS.ROOT_DIR  ../data/ISIC18/task3/ISIC2018_Task3_Training_Input

from skimage import io
import matplotlib.pyplot as plt

import numpy as np
img = io.imread('D:/tmp/00000986_fingerdown_isfinger.bmp') 
plt.imshow(img,cmap = 'Greys')



with  open('D:/tmp/00000986_fingerdown_isfinger.bin','rb') as fp:
    datas= np.array(list(fp.read()))
    
    
    img_raw = datas[0::2] +datas[1::2]*256
    
    img_raw = (img_raw - img_raw.min())/(img_raw.max()-img_raw.min())
img_raw  =img_raw.astype('float32')
img_raw = np.reshape(img_raw,img.shape)
plt.imshow(img_raw,cmap = 'Greys')
    #d16 = struct.unpack("h",datas)