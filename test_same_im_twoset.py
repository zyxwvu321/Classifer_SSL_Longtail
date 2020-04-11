# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:26:35 2019

@author: cmj
"""
import numpy as np

from pathlib import Path
import cv2

fd_18 = 'D:/dataset/ISIC/ISIC2018_Task3_Training_Input'
fd_19 = 'D:/dataset/ISIC/ISIC_2019_Training_Input'
flist_19 = Path(fd_19).rglob('*.jpg')


flist_18 = Path(fd_18).rglob('*.jpg')



fn_18 = [fn.stem for fn in flist_18]
fn_19 = [fn.stem for fn in flist_19]

for fn in fn_18:
    if fn in fn_19:
        fn_19.remove(fn)


for idx,fn in enumerate(fn_18):
    if fn in fn_19:
        print(f'{idx} {fn}')
        img1 = cv2.imread( str(Path(fd_18)/(fn + '.jpg')))
        img2 = cv2.imread( str(Path(fd_18)/(fn + '.jpg')))
        
        dd = img1.astype('float') - img2.astype('float')
        
        max_d = np.abs(dd).max()
        print(f'max d is {max_d}')
