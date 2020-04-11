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


df = pd.read_csv('../data/ISIC18/task3/ISIC2018_Task3_Training_GroundTruth.csv')
datas = df.values

fd_in = '../data/ISIC18/task3/ISIC2018_Task3_Test_Input'
fd_out = '../data/ISIC18/task3/ISIC2018_Task3_Test_Input_coloradj'



os.makedirs(fd_out,exist_ok = True)
flist = sorted(list(Path(fd_in).glob('*.jpg')))
flist = [str(fn) for fn in flist]



list_gain = list()
for fn in tqdm(flist):    
    img_fn = str(fn)
    img = cv2.imread(img_fn)
    
    #img_eh = shades_of_gray_method(img)
    img_eh, gain_rgb = sod_minkowski(img)
    
    list_gain.append(np.array([fn, *gain_rgb]))
    
    
    #img_eh = np.clip(img_eh,0.0,1.0)
    
    cv2.imwrite(str(Path(fd_out)/Path(img_fn).name),(img_eh).astype('uint8'))





df = pd.DataFrame(data = list_gain, columns = ['fn','gain_r','gain_g','gain_b'])
df.to_csv('./dat/ISIC18_colorgain_test.csv', index=False)