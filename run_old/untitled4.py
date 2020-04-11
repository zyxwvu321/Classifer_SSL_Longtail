# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 12:31:29 2020
test if meta data in ISIC19 include all the images in ISIC18
@author: cmj
"""

import pandas as pd
import numpy as np
from tqdm import tqdm



df = pd.read_csv('D:\dataset\ISIC\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv')

datas = df.values
df = pd.read_csv('D:\dataset\ISIC\ISIC_2019_Training_Metadata.csv')
datas_meta = df.values


for fn in datas[:,0]:
    if fn not in datas_meta[:,0]:
        print(fn)
        
#%% meta19test include meta18 test?
from pathlib import Path


df = pd.read_csv('../data/ISIC_2019_Test_Metadata.csv')
datas_meta = df.values


fn_test_18 =  Path('../data/ISIC2018_Task3_Test_Input').glob('*.jpg')
fn_test_18 = [fn.stem for fn in fn_test_18]
for fn in fn_test_18:
    if fn not in datas_meta[:,0]:
        print(fn)



#%%

import cv2
from pathlib import Path
flist = Path('D:\dataset\ISIC\ISIC2018_Task3_Training_Input').glob('*.jpg')
flist = [str(fn) for fn in flist]


hw_list = []
for fn in tqdm(flist):
    img = cv2.imread(fn)
    hh,ww,_ = img.shape
    hw_list.append([Path(fn).stem, hh, ww])


df = pd.DataFrame(data = hw_list, index = None,columns = ['fn','hh','ww'])
df.to_csv('./dat/ISIC18_hw.csv')