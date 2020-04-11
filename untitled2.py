# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:35:31 2020

@author: cmj
"""

import cv2
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
from  scipy.ndimage import binary_opening
from pathlib import Path
from tqdm import tqdm
fn = 'D:\dataset\ISIC\ISIC_2019_Training_Input\ISIC_0000004.jpg'

img = cv2.imread(fn)

img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)




label_blk =measure.label(img_grey<10, connectivity=img_grey.ndim)




def get_mask(img):
    #return a mask of black background
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_grey = cv2.medianBlur(img_grey,5)
    label_blk =measure.label(img_grey<10, connectivity=img_grey.ndim)
    props = measure.regionprops(label_blk)
    areas = np.array([prop.area for prop in props])
    
    idx_area = areas > img_grey.size *0.1
    
    mask = np.ones_like(img_grey)
    if idx_area.sum()>0: 
        # there are black regions
        for idx,val in enumerate(idx_area):
            if val!=0:
                mask[label_blk==idx+1] = 0
        mask = 255*binary_opening(mask,structure = np.ones((7,7))).astype(img_grey.dtype)
    return mask
        
    




#%%
flist = sorted(list(Path('D:\dataset\ISIC\ISIC_2019_Training_Input').glob('*.jpg')))
flist = [str(fn)  for fn in flist]
fd_mask = Path('D:\git_code\SkinClassifier\data\ISIC19\mask')

for fn in tqdm(flist):
    img = cv2.imread(fn)
    
    mask= get_mask(img)
    
    if mask.sum()!= img.shape[0]*img.shape[1]:
        print(Path(fn).name)


        cv2.imwrite(str(fd_mask/Path(fn).name), mask)
    
#    
#
#def hf_circle(img):
#    # input is greyscale
#    blur = cv2.medianBlur(img,5)
#    circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,30,
#                            param1=80,param2=40,minRadius=150,maxRadius=800)
#    circles = np.uint16(np.around(circles))
#    return circles
#
##%%
#
#mask = get_mask(img)
##%%
#circles = hf_circle(img_grey)
#for i in circles[0,:]:
#    # draw the outer circle
#    cv2.circle(img,(i[0], i[1]), i[2], (255, 0, 0), 2)
#    # draw the center of the circle
#    cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 5)
# 
#plt.imshow(img[:,:,::-1])
#
#
##%%
#props = measure.regionprops(label_blk)
#
#
#areas = [prop.area for prop in props]
#
#
#plt.imshow(label_blk==1)
#
#for prop in props:
#    me_r = prop.mean_intensity
#    mi_r = prop.min_intensity
#    ma_r = prop.max_intensity
#    n_area = prop.area