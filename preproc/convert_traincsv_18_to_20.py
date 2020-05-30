# -*- coding: utf-8 -*-
"""
Created on Sun May 31 00:13:48 2020
convert 18 train csv to 20
@author: cmj
"""



import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp


#image_name	patient_id	sex	age_approx	anatom_site_general_challenge	diagnosis	benign_malignant	target
#ISIC_2637011	IP_7279968	male	45	head/neck	unknown	benign	0


fn_csv_18 = './dat/all18_info1.csv'
fn_csv_20 = './dat/all18_info1_convert20.csv'

map_18  = np.array([1,0,1, 1,0,0,0])








df = pd.read_csv(fn_csv_18)
val_18 =  df.values

val_18[:,3] = map_18[val_18[:,3].astype('int')]




df = pd.DataFrame(data = val_18, index = None,columns = ['fn','hh','ww','gt','age','pos','sex','g_r','g_g','g_b','n_rep'])
df.to_csv(fn_csv_20, index=False)