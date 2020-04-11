# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:42:58 2020

@author: cmj
"""

import pandas as pd
import numpy as np
vls = pd.read_csv('../checkpoints/ISIC18_test_submi_predt.csv').values

n_samp = 1512
n_class = 7
preds = vls[:,1:8].astype('float32')



n_m = 0
map_v = {'MEL':0,'NV':1,'BCC':2,'AKIEC':3,'BKL':4,'DF':5,'VASC':6}


gcomp = list()
gcomp0 = list()
for idx,vl in enumerate(vls):
    pred = vl[1:8].astype('float32')
    pred_o = vl[8]
    pred_m = vl[9]
    
    if isinstance(pred_m,str) and pred_m !='NONE':
        
        gt_v = map_v[pred_m]
        pr_v = map_v[pred_o]
        
        v1 = preds[idx, pr_v] 
        v2 =  preds[idx, gt_v]
        
        
        gcomp.append([pr_v,gt_v,v1,v2])
        
        proc = 0
        if v1-v2<0.2:
            proc = 1
        if (pr_v==1) and (v1-v2<0.4):
            proc  = 1
        if (pr_v==0) and (v1-v2<0.3):
            proc  = 1
        if (pr_v==4) and (v1-v2<0.25):
            proc  = 1
            
        #if v1-v2<0.2 or ((pr_v==1) and (v1-v2<0.5)  and (gt_v!=0) and (gt_v!=4)):
        if proc ==1:
            preds[idx, pr_v] = max(0.0,preds[idx, pr_v] - (0.6-preds[idx, gt_v]))
            gcomp0.append([pr_v,gt_v,v1,v2])

            preds[idx, gt_v] = 0.6
            n_m = n_m+1


n_a = 0
for idx,pred in enumerate(preds):
    if np.max(pred)<=0.5:
        #change conf to 0.6 if all conf <0.5
        pos = np.argmax(pred)
        preds[idx, pos] = 0.6
        n_a = n_a +1

preds = np.round_(preds,decimals = 4)

        

df = pd.DataFrame(data = preds,index =vls[:,0], columns = [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC'])


    
df.to_csv('./dat/ISIC18_test_submit2_setmaxconf.csv',index_label = 'image')