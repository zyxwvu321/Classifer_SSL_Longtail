# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 12:31:29 2020
test if meta data in ISIC20 
@author: cmj
"""
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp
import math
from count_meta import count_imfq_samemeta

#image_name	patient_id	sex	age_approx	anatom_site_general_challenge	diagnosis	benign_malignant	target
#ISIC_2637011	IP_7279968	male	45	head/neck	unknown	benign	0



fn_gts = ['../data/ISIC20/train.csv']

fn_metas = ['../data/ISIC20/train.csv']

map_gts = [[0,1]]


fd_in = '../data/all20_coloradj'
colorgain_csv = './dat/all20_colorgain.csv'
out_csv = './dat/all20_info1.csv'





info_list = []
df = pd.read_csv(colorgain_csv)
datas_colorgain =  df.values

for fn_gt, fn_meta, map_gt in zip(fn_gts, fn_metas, map_gts):
    
    
    df = pd.read_csv(fn_gt)
    datas = df.values
    
    
    if fn_meta is not None:
        df = pd.read_csv(fn_meta)
        datas_meta = df.values
        dict_im_fq = count_imfq_samemeta(fn_meta)
    

        
    flist = [osp.join(fd_in, fn+'.jpg') for fn in datas[:,0]]
    


    for idx, fn in enumerate(tqdm(flist)):
        img = cv2.imread(fn)
        hh,ww,_ = img.shape
        
        
        if fn_meta is not None:
            idx_meta = np.where(datas_meta[:,0]==Path(fn).stem)[0][0]
            meta = datas_meta[idx_meta][[3,4,2]]
            n_rep = dict_im_fq[Path(fn).stem]
        else:
            meta = [math.nan,math.nan,math.nan]
            n_rep = 1.0
        
        
        #gt = np.where(datas[idx][1:]==1)[0][0]
        gt = datas[idx][1]
        
        idx_colorgain = np.where(datas_colorgain[:,0]==Path(fn).stem)[0][0]
        color_gain = datas_colorgain[idx_colorgain]
        
        
        if map_gt is not None:
            if gt>= len(map_gt):
                continue # skip unknown
            gt = map_gt[gt]
        
        
        
        info_list.append([Path(fn).stem, hh, ww, gt,*meta,  *color_gain[1:4], n_rep])





df = pd.DataFrame(data = info_list, index = None,columns = ['fn','hh','ww','gt','age','pos','sex','g_r','g_g','g_b','n_rep'])
df.to_csv(out_csv, index=False)