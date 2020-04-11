#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:18:40 2019

@author: minjie
"""

from pathlib import Path
import cv2
from utils.utils import del_mkdirs
from tqdm import tqdm

fd = 'data/ISIC/train_18'
fd_in  = list(Path(fd).glob('*'))

#fd_out = ['./data/bone_aug/train','./data/bone_aug/test']

fd_out = './test_aug_crp'
del_mkdirs([fd_out])

import albumentations as A
aug =  A.Compose([A.Resize( 384, 512,  interpolation=1,  p=1),
                                 #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,  p=0.5), 
                                # A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=15, val_shift_limit=20,p = 0.5),
                                 
                                 A.OneOf([A.Blur(blur_limit=5, p=0.3),
                                 A.GaussNoise(var_limit=(5.0, 10.0), p=0.3),
                                 A.IAASharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.4)],p=1.0),
    
                                 A.Flip(p = 0.5),
                                 #A.Transpose(p = 0.5),
                                 #A.RandomRotate90(p = 0.5),
                                 A.OneOf([A.IAACropAndPad(percent=(-0.1,0.1),keep_size=True,  p=0.5),
                                       A.ShiftScaleRotate(p =0.5)],p = 1.0)
                                ])




for fds in fd_in:
    flist = [str(fn)  for fn in Path(fds).glob('*.jpg')]
    im_fn = flist[0]
#    for im_fn  in tqdm(flist):
    img = cv2.cvtColor(cv2.imread(im_fn), cv2.COLOR_BGR2RGB)

    for n_test in range(9):
        #img_aug = (aug(image= img)['image']*255).astype('uint8')
        img_aug = aug(image= img)['image']
        #img_aug = cv2.cvtColor(cv2.imread(im_fn), cv2.COLOR_RGB2BGR)
        #cv2.imwrite(im_fn.replace('.','_'+ str(n_test) +'.').replace('bone','bone_aug'),img_aug)
        cv2.imwrite(str(Path(fd_out)/Path(im_fn).name).replace('.jpg','_aug' +str(n_test+1) +'.jpg'),img_aug[:,:,::-1])
    

    cv2.imwrite(str(Path(fd_out)/Path(im_fn).name),img[:,:,::-1])