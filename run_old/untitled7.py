#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:44:04 2020
test if gt 2019 include valid18/test18
@author: minjie
"""

from pathlib import Path
import pandas as pd
import numpy as np


fn_ISIC19_gt ='../data/ISIC19/ISIC_2019_Training_GroundTruth.csv'


flist_valid = sorted(list(Path('../data/ISIC18/task3/ISIC2018_Task3_Validation_Input').glob('*.jpg')))
flist_valid = [fn.stem for fn in flist_valid]

flist_test = sorted(list(Path('../data/ISIC18/task3/ISIC2018_Task3_Test_Input').glob('*.jpg')))
flist_test = [fn.stem for fn in flist_test]


df = pd.read_csv(fn_ISIC19_gt)
datas = df.values
fns_19 = datas[:,0]



gt_inc_valid = np.zeros(len(flist_valid))

for idx,fn in enumerate(flist_valid):
    if fn in fns_19:
        gt_inc_valid[idx] = 1



gt_inc_test = np.zeros(len(flist_test))

for idx,fn in enumerate(flist_test):
    if fn in fns_19:
        gt_inc_test[idx] = 1




