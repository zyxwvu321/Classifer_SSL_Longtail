#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:43:36 2020
sort csv fn
@author: minjie
"""


import pandas as pd
import numpy as np





#%%
fns = ['../checkpoints/eval_resnet50_singleview-Loss-ce-tta-0.csv',
       '../checkpoints/eval_resnet50_singleview-Loss-ce-tta-1.csv',
       '../checkpoints/eval_resnet50_metasingleview-Loss-ce-tta-0.csv',
       '../checkpoints/eval_resnet50_metasingleview-Loss-ce-tta-1.csv',
       '../checkpoints/eval_effnetb4_singleview-Loss-ce-tta-0.csv',
       '../checkpoints/eval_effnetb4_singleview-Loss-ce-tta-1.csv',
       '../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-0.csv',
       '../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-1.csv',
       
       ]


for fn in fns:
    
    gts = pd.read_csv(fn).values
    
    idx = np.argsort(gts[:,0])
    gts = gts[idx,:]
    
    
    
    df = pd.DataFrame(data = gts[:,1:].astype('float32'),index =gts[:,0], columns = [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC','pred', 'GT'])
    for col in [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']:
        df[col] = df[col].apply(lambda x: format(x,'.4f'))
    for col in ['pred', 'GT']:
        df[col] = df[col].apply(lambda x: format(x,'.0f'))    

    
    df.to_csv(fn)
    
    