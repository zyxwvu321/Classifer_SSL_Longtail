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
import math

fn_gts = ['../data/ISIC19/ISIC_2019_Training_GroundTruth.csv',
       './dat/extra_GT.csv']

fn_metas = ['../data/ISIC19/ISIC_2019_Training_Metadata.csv',None]

map_gts = [[0,1,2,3,4,5,6,3],[0,1,2,3,4,5,6,3]]


fd_in = '../data/all18_coloradj'
colorgain_csv = './dat/all18_colorgain.csv'
out_csv = './dat/all18_info1.csv'



#fn_gts = ['../data/ISIC19/ISIC_2019_Training_GroundTruth.csv',
#       './dat/extra_GT.csv']
#fn_metas = ['../data/ISIC19/ISIC_2019_Training_Metadata.csv',None]
#
#map_gts = [[0,1,2,3,4,5,6,3],[0,1,2,3,4,5,6,3]]




fn_gts = ['./dat/extra_GT.csv']
fn_metas = [None]

map_gts = [[0,1,2,3,4,5,6,3]]



fd_in = '../data/all18_coloradj1'
#fd_in = '../data/extra_all'
colorgain_csv = './dat/all18_colorgain1.csv'
out_csv = './dat/all18_info1.csv'



info_list = []
df = pd.read_csv(colorgain_csv)
datas_colorgain =  df.values

for fn_gt, fn_meta, map_gt in zip(fn_gts, fn_metas, map_gts):
    

    df = pd.read_csv(fn_gt)
    datas = df.values
    
    
    if fn_meta is not None:
        df = pd.read_csv(fn_meta)
        datas_meta = df.values
    

        
    flist = [osp.join(fd_in, fn+'.jpg') for fn in datas[:,0]]
    


    for idx, fn in enumerate(tqdm(flist)):
        img = cv2.imread(fn)
        hh,ww,_ = img.shape
        
        
        if fn_meta is not None:
            idx_meta = np.where(datas_meta[:,0]==Path(fn).stem)[0][0]
            meta = datas_meta[idx_meta][[1,2,4]]
        else:
            meta = [math.nan,math.nan,math.nan]
        
        
        
        gt = np.where(datas[idx][1:]==1)[0][0]
        
        idx_colorgain = np.where(datas_colorgain[:,0]==Path(fn).stem)[0][0]
        color_gain = datas_colorgain[idx_colorgain]
        
        
        if map_gt is not None:
            if gt>= len(map_gt):
                continue # skip unknown
            gt = map_gt[gt]
        
        
        
        info_list.append([Path(fn).stem, hh, ww, gt,*meta,  *color_gain[1:]])





df = pd.DataFrame(data = info_list, index = None,columns = ['fn','hh','ww','gt','age','pos','sex','g_r','g_g','g_b'])
df.to_csv(out_csv, index=False)