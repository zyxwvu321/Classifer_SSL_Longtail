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

#fn_ISIC18_task3_gt = '../data/ISIC18/task3/ISIC2018_Task3_Training_GroundTruth.csv'
fn_ISIC19_meta ='../data/ISIC19/ISIC_2019_Test_Metadata.csv'

fn_ISIC18_colorgain = './dat/all19_usp_colorgain.csv'
fd_im = '../data/all19_usp_coloradj'

out_csv = './dat/all19_info_test.csv'

#df = pd.read_csv(fn_ISIC18_task3_gt)
#datas = df.values

df = pd.read_csv(fn_ISIC19_meta)
datas_meta = df.values



df = pd.read_csv(fn_ISIC18_colorgain)
datas_colorgain =  df.values


#seg result



flist = sorted(list(Path(fd_im).glob('*.jpg')))
fns = [fn.stem for fn in flist]
flist = [str(fn) for fn in flist]

for fn in fns:
    if fn not in datas_meta[:,0]:
        #print img filename if metafile not include
        print(fn)
        

# FN, hh, ww, class, meta(age,pos,sex, 3col, lesion_id is skipped), cropped bbox(4 col)) color_gain(3col)--------14 col

info_list = []
for idx, fn in enumerate(tqdm(flist)):
    img = cv2.imread(fn)
    hh,ww,_ = img.shape
    
    idx_meta = np.where(datas_meta[:,0]==Path(fn).stem)[0][0]
    meta = datas_meta[idx_meta][[1,2,3]]
    
    
 
    
    #gt = np.where(datas[idx][1:]==1)[0][0]
    idx_colorgain = np.where(datas_colorgain[:,0]==Path(fn).stem)[0][0]
    color_gain = datas_colorgain[idx_colorgain]
    
    info_list.append([Path(fn).stem, hh, ww,-1, *meta,  *color_gain[1:]])





df = pd.DataFrame(data = info_list, index = None,columns = ['fn','hh','ww','gt','age','pos','sex','g_r','g_g','g_b'])
df.to_csv(out_csv, index=False)