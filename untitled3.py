# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:43:15 2020

@author: cmj
"""

import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp
import shutil



fn_ISIC19_gt = '../data/ISIC19/ISIC_2019_Training_GroundTruth.csv'
fn_ISIC19_meta ='../data/ISIC19/ISIC_2019_Training_Metadata.csv'

path_im = '../data/ISIC19/ISIC_2019_Training_Input'
path_tar = '../data_finger'

df = pd.read_csv(fn_ISIC19_gt)
datas = df.values

df = pd.read_csv(fn_ISIC19_meta)
datas_meta = df.values


idx = np.where(datas_meta[:,2] == 'palms/soles')[0]

fids = datas_meta[:,0][idx]


idx_cols = []
for fid in tqdm(fids):
    fn_id = np.where(datas[:,0]==fid)[0][0]
    idx_cols.append(fn_id)
    shutil.copyfile(osp.join(path_im,datas[fn_id,0] + '.jpg'), osp.join(path_tar,datas[fn_id,0] + '.jpg'))
    
data_finger = datas[np.array(idx_cols)]


df_finger = pd.DataFrame(data = data_finger, index = None,columns = ['fn','MEL','NV','BCC',	'AK','BKL',	'DF',	'VASC',	'SCC',	'UNK'])
df_finger.to_csv('../data_finger/finger.csv', index=False)