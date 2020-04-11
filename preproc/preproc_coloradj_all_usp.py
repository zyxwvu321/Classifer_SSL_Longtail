#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:36:15 2020
preproc using a color adjustment for the dataset
also save gain_rgb for this image


@author: minjie
"""


from pathlib import Path
import os

import cv2
from tqdm import tqdm
from color_enh import sod_minkowski

import pandas as pd
import numpy as np


#fn_gts = ['../data/ISIC18/task3/ISIC2018_Task3_Training_GroundTruth.csv',
#       './dat/jiuyuan_gt.csv',
#       '../data/ISIC19/ISIC_2019_Training_GroundTruth.csv'
#       ]

#fd_ins = ['../data/ISIC18/task3/ISIC2018_Task3_Validation_Input','../data/ISIC18/task3/ISIC2018_Task3_Test_Input']


#fd_out = '../data/all18_usp_coloradj'
         

#out_csv = './dat/all18_usp_colorgain.csv'


fd_ins = ['../data/ISIC19/ISIC_2019_Test_Input']


fd_out = '../data/all19_usp_coloradj'
         

out_csv = './dat/all19_usp_colorgain.csv'


os.makedirs(fd_out,exist_ok = True)
list_gain = list()
for fd_in in fd_ins:
    flist = sorted(list(Path(fd_in).glob('*.jpg')))
    flist = [str(fn) for fn in flist]
    



    
    
    
    
    for img_fn in tqdm(flist):    
        
        img = cv2.imread(img_fn)
        hh,ww,_ = img.shape
        if min(hh,ww)>1024: # limit image size
            rt = max(1024.0/hh,1024.0/ww)
            tar_hw = (int(rt*ww),int(rt*hh))
            img = cv2.resize(img,tar_hw,cv2.INTER_CUBIC)
            
        
        #img_eh = shades_of_gray_method(img)
        img_eh, gain_rgb = sod_minkowski(img)
        
        list_gain.append(np.array([Path(img_fn).stem, *gain_rgb]))
        
        
        #img_eh = np.clip(img_eh,0.0,1.0)
        
        cv2.imwrite(str(Path(fd_out)/Path(img_fn).name),(img_eh).astype('uint8'),[int(cv2.IMWRITE_JPEG_QUALITY),100])
    
    
    
    
    
df = pd.DataFrame(data = list_gain, columns = ['fn','gain_r','gain_g','gain_b'])
df.to_csv(out_csv, index=False)