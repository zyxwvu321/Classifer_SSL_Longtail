# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 12:31:29 2020
test if meta data in ISIC19 include all the images in ISIC18
@author: cmj
"""
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp

fn_ISIC18_task3_gt = '../data/ISIC19/ISIC_2019_Training_GroundTruth.csv'
fn_ISIC19_meta ='../data/ISIC19/ISIC_2019_Training_Metadata.csv'
fn_ISIC18_presegbox = './dat/bbox_isic19.csv'
fn_ISIC18_colorgain = './dat/ISIC19_colorgain.csv'
fd_im = '../data/ISIC19/ISIC_2019_Training_Input_coloradj'

df = pd.read_csv(fn_ISIC18_task3_gt)
datas = df.values

df = pd.read_csv(fn_ISIC19_meta)
datas_meta = df.values

df = pd.read_csv(fn_ISIC18_presegbox)
datas_box =  df.values[:,1:]

df = pd.read_csv(fn_ISIC18_colorgain)
datas_colorgain =  df.values[:,1:]


#seg result




for fn in datas[:,0]:
    if fn not in datas_meta[:,0]:
        #print img filename if metafile not include
        print(fn)
        


#flist = sorted(list(Path('../data/ISIC18/ISIC2018_Task3_Training_Input').glob('*.jpg')))
#flist = [str(fn) for fn in flist]
flist = [osp.join(fd_im, fn+'.jpg') for fn in datas[:,0]]

# FN, hh, ww, class, meta(age,pos,sex, 3col, lesion_id is skipped), cropped bbox(4 col)) color_gain(3col)--------14 col

info_list = []
for idx, fn in enumerate(tqdm(flist)):
    img = cv2.imread(fn)
    hh,ww,_ = img.shape
    
    idx_meta = np.where(datas_meta[:,0]==Path(fn).stem)[0][0]
    meta = datas_meta[idx_meta][[1,2,4]]
    
    
    #idx_box = np.where(datas_box[:,0]==Path(fn).stem)[0][0]
    bbox = datas_box[idx][1:]
    
    gt = np.where(datas[idx][1:]==1)[0][0]
    
    color_gain = datas_colorgain[idx]
    
    
    map_gt = [0,1,2,3,4,5,6,3]
    gt = map_gt[gt]
    
    info_list.append([Path(fn).stem, hh, ww, gt,*meta, *bbox, *color_gain])





df = pd.DataFrame(data = info_list, index = None,columns = ['fn','hh','ww','gt','age','pos','sex','x1','y1','x2','y2','g_r','g_g','g_b'])
df.to_csv('./dat/ISIC19_info.csv', index=False)