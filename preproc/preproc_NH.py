#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:36:15 2020
preproc new images



@author: minjie
"""


from pathlib import Path
import os

import cv2
from tqdm import tqdm
from color_enh import sod_minkowski

import pandas as pd
import numpy as np


def str2id(num, prefix = None):
    nstr = str(num)
    if len(nstr)<=5:
        nstr = '0'*(5-len(nstr)) + nstr
    if prefix is not None:
        nstr = prefix + '_' + nstr
    nstr = nstr + '.jpg'
    return nstr



fd_in = '../data/jiuyuan'
tar_gt = '../data/jiuyuan/GT.csv'

fd_out = '../data/jiuyuan_extra'
dict_labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


os.makedirs(fd_out,exist_ok = True)


flist = sorted(list(Path(fd_in).rglob('*.*')))
flist = [str(fn) for fn in flist]


#%%
nid = 1

fn_gts = list()

for img_fn in tqdm(flist):    
    if not Path(img_fn).is_file or Path(img_fn).suffix.lower() not in ['.jpg','.bmp']:
        continue
    
    
    img = cv2.imread(img_fn)
    if img is None:
        continue
    
    hh,ww,_ = img.shape
    
    if min(hh,ww)>1024: # limit image size
        rt = max(1024.0/hh,1024.0/ww)
        tar_hw = (int(rt*ww),int(rt*hh))
        img = cv2.resize(img,tar_hw,cv2.INTER_CUBIC)
    
    
    if Path(img_fn).parts[-2]  not in dict_labels:
        continue
    
    str_id = str2id(nid,'jiuyuan')
    
    fn_gts.append([Path(str_id).stem ,    dict_labels.index(Path(img_fn).parts[-2])])
    
        
    cv2.imwrite(str(Path(fd_out)/str_id),(img).astype('uint8'),[int(cv2.IMWRITE_JPEG_QUALITY),100])
    nid = nid +1


#%%  create csv

n_img = len(fn_gts)
gt_np = np.zeros((n_img,len(dict_labels)),dtype='int64')
for idx,gt in enumerate(fn_gts):
    gt_np[idx, gt[1]] = 1
    

df = pd.DataFrame(data = gt_np, columns = dict_labels,index =np.array(fn_gts)[:,0] )
df.to_csv('./dat/jiuyuan_gt.csv',index_label = 'image')    
