# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:43:15 2020
add ISIC19 images exclude pos = 'oral/genital'
 
@author: cmj
"""
import os



import pandas as pd

from tqdm import tqdm
import os.path as osp
import shutil



fn_ISIC18_gt = '../data/ISIC18/task3/ISIC2018_Task3_Training_GroundTruth.csv'
fn_ISIC19_meta ='../data/ISIC19/ISIC_2019_Training_Metadata.csv'

path_im = '../data/ISIC19/ISIC_2019_Training_Input'
path_tar = '../data_19a'
os.makedirs(path_tar,exist_ok = True)


df = pd.read_csv(fn_ISIC18_gt)
datas18 = df.values

df = pd.read_csv(fn_ISIC19_meta)
datas_meta = df.values


#idx = np.where(datas_meta[:,2] == 'palms/soles')[0]

fids = datas_meta[:,0]


idx_cols = []
for meta in tqdm(datas_meta):
    fn = meta[0]
    pos = meta[2]
    
    if pos!='oral/genital' and fn not in datas18[:,0]:
    
        shutil.copyfile(osp.join(path_im, fn + '.jpg'), osp.join(path_tar, fn + '.jpg'))
    
