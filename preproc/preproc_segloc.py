# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:11:41 2019
preproc ISIC18 task 1 segmentation 
use mask to calc xywh range and train a image localization model
@author: cmj
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from preproc.letterbox_resize import letterbox_image,letterbox_xywh




#fd = '../data/ISIC07_task1_train'
#fd_img = 'ISIC-2017_Training_Data'
#fd_anno = 'ISIC-2017_Training_Part1_GroundTruth'
#
#
#fd_box = 'ISIC-2017_Box'
#fd_imgt = 'ISIC-2017_Img'
#fd_imganno = 'ISIC-2017_Anno'


fd = '../data/ISIC18/task1'
fd_img = 'ISIC2018_Task1-2_Training_Input'
fd_anno = 'ISIC2018_Task1_Training_GroundTruth'


fd_box = 'ISIC-2018_Box'
fd_imgt = 'ISIC-2018_Img'
fd_imganno = 'ISIC-2018_Anno'



b_letterbox = True
b_drawbox  = True

out_wh = (512,512)

fd_img = Path(fd)/fd_img
fd_anno = Path(fd)/fd_anno
fd_box = Path(fd)/fd_box
fd_imgt = Path(fd)/fd_imgt
fd_imganno = Path(fd)/fd_imganno


fd_box.mkdir(parents = True,exist_ok = True)
fd_imgt.mkdir(parents = True,exist_ok = True)
fd_imganno.mkdir(parents = True,exist_ok = True)

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
    
    xx = (xmin+xmax)/2.0
    yy = (ymin+ymax)/2.0
    ww = xmax - xmin 
    hh = ymax - ymin 
    
    xx = xx /float(im_w)
    ww = ww /float(im_w)
    yy = yy /float(im_h)
    hh = hh /float(im_h)
    
    pos_infos.append(np.array([fn.name,img_mask.shape[1],img_mask.shape[0],xx,yy,ww,hh]))
    
    if b_letterbox is False:
        img_t = cv2.resize(img,out_wh)
        

    else:
        box_dim = (out_wh[1], out_wh[0])
        
        img_t = letterbox_image(img, box_dim)
        
        
        xx,yy,ww,hh = letterbox_xywh(img,box_dim, [xx,yy,ww,hh])
        
        
    fn_imgt = str(fd_imgt/fn.name)
    fn_txt = str(fd_box/fn.name).replace('.jpg','.txt')
    
    
    
    cv2.imwrite(fn_imgt,img_t)
    
    if b_drawbox is True:
        x1,y1,x2,y2 = xx-ww/2,yy-hh/2,xx+ww/2,yy+hh/2
        x1 = int(round(x1*out_wh[0]))
        y1 = int(round(y1*out_wh[1]))
        x2 = int(round(x2*out_wh[0]))
        y2 = int(round(y2*out_wh[1]))
        cv2.rectangle(img_t, (x1, y1), (x2, y2), (0, 0, 255), 2)
        fn_img_anno = str(fd_imganno/fn.name)
        cv2.imwrite(fn_img_anno,img_t)
    
    
    np.savetxt(fn_txt,np.array([xx,yy,ww,hh])[None,...], fmt = '%2.8f')
    
    
    
    

pos_infos = np.array(pos_infos)

df = pd.DataFrame(data = pos_infos,columns=['fname','x','y','boxx','boxy','boxw','boxh'])
df.to_csv(str(Path(fd)/'info.csv'))



