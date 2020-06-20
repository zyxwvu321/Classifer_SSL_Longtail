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

    # divide h*w into five scales, consider it as extra scale meta for n_rep calc.    
    s_hw = []
    scale_th = np.array([500000,2500000,7000000,15000000])

    for fn in datas[:,0]:
        idx_colorgain = np.where(datas_colorgain[:,0]==fn)[0][0]
        cg = datas_colorgain[idx_colorgain]
        hw = cg[4]*cg[5]
        ss = (scale_th<=hw).sum()
        s_hw.append(ss)
    
    
    if fn_meta is not None:
        df = pd.read_csv(fn_meta)
        datas_meta = df.values
        datas_meta = np.hstack((datas_meta[:,:5], np.int32(s_hw)[:,None]))
        
        dict_im_fq = count_imfq_samemeta(datas_meta,meta_format = 20)
    

        
    flist = [osp.join(fd_in, fn+'.jpg') for fn in datas[:,0]]
    


    for idx, fn in enumerate(tqdm(flist)):
        #img = cv2.imread(fn)
        #hh,ww,_ = img.shape
  
        #gt = np.where(datas[idx][1:]==1)[0][0]
        gt = datas[idx][-1]
        if map_gt is not None:
            if gt>= len(map_gt):
                continue # skip unknown
            gt = map_gt[gt]
           
        
        if fn_meta is not None:
            idx_meta = np.where(datas_meta[:,0]==Path(fn).stem)[0][0]
            meta = datas_meta[idx_meta][[3,4,2]]
            
            if gt==1:
                n_rep = dict_im_fq[Path(fn).stem]
            else:
                n_rep = 1.0
        else:
            meta = [math.nan,math.nan,math.nan]
            n_rep = 1.0
        
        

        
        idx_colorgain = np.where(datas_colorgain[:,0]==Path(fn).stem)[0][0]
        color_gain = datas_colorgain[idx_colorgain]
        
        

        
        
        #info_list.append([Path(fn).stem, hh, ww, gt,*meta,  *color_gain[1:4], n_rep])
        info_list.append([Path(fn).stem, color_gain[4], color_gain[5], gt,*meta,  *color_gain[1:4], n_rep])




df = pd.DataFrame(data = info_list, index = None,columns = ['fn','hh','ww','gt','age','pos','sex','g_r','g_g','g_b','n_rep'])
df.to_csv(out_csv, index=False)