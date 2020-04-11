#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:43:36 2020
a test of K-L grade est 
@author: minjie
"""


from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np



#%% a test to read csv of gt infos

gts = pd.read_csv('./dat/ISIC18_info.csv').values



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

#def est_kl(arr):
#    n_kl = arr.shape[1]
#    kl_est = np.sum(arr[:,1:5].astype('float32') * np.arange(1,n_kl+1),axis=1)
#    
#    
def bal_acc(cm):
    cls_acc1 = cm.diagonal()/np.sum(cm,axis = 1)

    cls_acc2 = cm.diagonal()/np.sum(cm,axis = 0)
    
    cls_acc3  = cm.diagonal()/( np.sum(cm,axis = 0) +  np.sum(cm,axis = 1)-cm.diagonal()) 
        
    avg_acc = np.sum(cm.diagonal())/cm.sum()
    
    
    bal_acc1 = np.mean(cls_acc1)
    bal_acc2 = np.mean(cls_acc2)
    bal_acc3 = np.mean(cls_acc3)
    
    
    probs = np.array([avg_acc, bal_acc1,bal_acc2,bal_acc3])
    probs = np.round_(probs,decimals=4)
    return  probs

for fn in fns:
    print('*'*32)
    kl1 = pd.read_csv(fn).values[:,-2:]
    
    y_true = kl1[:,1]
    y_pred = kl1[:,0]
    y_true = np.array(y_true).astype('int64') 
    y_pred = np.array(y_pred).astype('int64') 
    cm = confusion_matrix(y_true, y_pred) 
    print(fn)
    print(cm)
    print(bal_acc(cm))
    
    