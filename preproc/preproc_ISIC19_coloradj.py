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


df = pd.read_csv('../data/ISIC19/ISIC_2019_Training_GroundTruth.csv')
datas = df.values

fd_in = '../data/ISIC19/ISIC_2019_Training_Input'
fd_out = '../data/ISIC19/ISIC_2019_Training_Input_coloradj'



os.makedirs(fd_out,exist_ok = True)
flist = Path(fd_in).glob('*.jpg')
flist = [str(fn) for fn in flist]



list_gain = list()
for fn in tqdm(datas[:,0]):    
    img_fn = str(Path(fd_in)/(fn + '.jpg'))
    img = cv2.imread(img_fn)
    
    #img_eh = shades_of_gray_method(img)
    img_eh, gain_rgb = sod_minkowski(img)
    
    list_gain.append(np.array([fn, *gain_rgb]))
    
    
    #img_eh = np.clip(img_eh,0.0,1.0)
    
    cv2.imwrite(str(Path(fd_out)/Path(img_fn).name),(img_eh).astype('uint8'))





df = pd.DataFrame(data = list_gain, columns = ['fn','gain_r','gain_g','gain_b'])
df.to_csv('./dat/ISIC19_colorgain.csv', index=False)