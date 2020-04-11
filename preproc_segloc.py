# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:11:41 2019

@author: cmj
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
fd = 'D:/dataset/ISIC/ISIC07_task1_train'
fd_img = 'ISIC-2017_Training_Data'
fd_anno = 'ISIC-2017_Training_Part1_GroundTruth'
fd_box = 'ISIC-2017_Box'
fd_imgt = 'ISIC-2017_Img'

out_wh = (512,512)

fd_img = Path(fd)/fd_img
fd_anno = Path(fd)/fd_anno
fd_box = Path(fd)/fd_box
fd_imgt = Path(fd)/fd_imgt


fd_box.mkdir(parents = True,exist_ok = True)
fd_imgt.mkdir(parents = True,exist_ok = True)


flist   = sorted(list(fd_img.glob('*.jpg')))



pos_infos = list()

for fn in tqdm(flist):
    img = cv2.imread(str(fn))
    
    fn_mask = str(Path(fd_anno)/fn.name).replace('.jpg','_segmentation.png')
    img_mask = cv2.imread(str(fn_mask))[...,0]
    
    maskpos = np.where(img_mask)
    
    ymin,ymax = maskpos[0].min(),maskpos[0].max()
    xmin,xmax = maskpos[1].min(),maskpos[1].max()
    
    im_w,im_h = img.shape[1],img.shape[0]
    
    img_t = cv2.resize(img,out_wh)
    
    
    xx = (xmin+xmax)/2.0
    yy = (ymin+ymax)/2.0
    ww = xmax - xmin + 1
    hh = ymax - ymin + 1
    
    pos_infos.append(np.array([fn.name,img_mask.shape[1],img_mask.shape[0],round(xx),round(yy),ww,hh]))
    
    xx = round(xx * out_wh[0]/float(im_w))
    ww = round(ww * out_wh[0]/float(im_w))
    yy = round(yy * out_wh[1]/float(im_h))
    hh = round(hh * out_wh[1]/float(im_h))
    
    fn_imgt = str(fd_imgt/fn.name)
    fn_txt = str(fd_box/fn.name).replace('.jpg','.txt')
    
    cv2.imwrite(fn_imgt,img_t)
    
    np.savetxt(fn_txt,np.array([xx,yy,ww,hh])[None,...], fmt = '%d')
    
    

pos_infos = np.array(pos_infos)

df = pd.DataFrame(data = pos_infos,columns=['fname','x','y','boxx','boxy','boxw','boxh'])
df.to_csv(str(Path(fd)/'info.csv'))



