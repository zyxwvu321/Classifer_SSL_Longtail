# -*- coding: utf-8 -*-
"""
Created on Tue May 26 01:19:14 2020
count the images with same meta, out dict
this is used as extra weight in loss func. (e.g. there is a case with 19 repeated similar images in ISIC)
@author: cmj
"""

import pandas as pd
import numpy as np
#from pathlib import Path
from tqdm import tqdm
def count_imfq_samemeta(fn_meta):
    dict_im_fq = dict()
    df = pd.read_csv(fn_meta)
    df_v = df.values

    n_r = df_v.shape[0]
    prop = []
    n_nan = []
    for idx1 in tqdm(range(n_r)):
        r1 = df_v[idx1,1:]
        prop.append(str(r1[0]) + str(r1[1])+str(r1[2])+str(r1[3]))
        n_nan.append(pd.isnull(r1).sum())

    uni_meta,uni_meta_count =np.unique(prop,return_counts=True)



    im_fn = df_v[:,0]
    for idx1 in tqdm(range(n_r)):
        fn = im_fn[idx1]
        meta = prop[idx1]
        meta_item = n_nan[idx1]
        if meta_item>0:
            dict_im_fq[fn] = 1.0 # not all meta data available
        else:
            meta_idx = np.where(uni_meta==meta)[0][0]
            dict_im_fq[fn] = uni_meta_count[meta_idx]
            

    
    return dict_im_fq



if __name__ == '__main__':
    fn_meta = '../data/ISIC19/ISIC_2019_Training_Metadata.csv'   
    dict_im_fq = count_imfq_samemeta(fn_meta)
    