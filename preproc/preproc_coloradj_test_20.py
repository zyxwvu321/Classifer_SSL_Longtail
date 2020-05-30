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




fd_in = '../data/ISIC20/test'
fd_out = '../data/all20_usp_coloradj'

out_csv = './dat/all20_usp_colorgain.csv'

os.makedirs(fd_out,exist_ok = True)
flist = sorted(list(Path(fd_in).glob('*.jpg')))
flist = [str(fn) for fn in flist]



list_gain = list()
for fn in tqdm(flist):    
    img_fn = str(fn)
    img = cv2.imread(img_fn)
    hh,ww,_ = img.shape
    
    if min(hh,ww)>1024: # limit image size
        rt = max(1024.0/hh,1024.0/ww)
        tar_hw = (int(rt*ww),int(rt*hh))
        img = cv2.resize(img,tar_hw,cv2.INTER_CUBIC)
    #img_eh = shades_of_gray_method(img)
    img_eh, gain_rgb = sod_minkowski(img)
    
    list_gain.append(np.array([Path(img_fn).stem, *gain_rgb,hh,ww]))
    
    
    #img_eh = np.clip(img_eh,0.0,1.0)
    
    cv2.imwrite(str(Path(fd_out)/Path(img_fn).name),(img_eh).astype('uint8'))





df = pd.DataFrame(data = list_gain, columns = ['fn','gain_r','gain_g','gain_b','height','width'])
df.to_csv(out_csv, index=False)