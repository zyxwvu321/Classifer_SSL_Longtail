# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:31:39 2020
test fl area
@author: cmj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

vls = pd.read_csv('./dat/ISIC18_info.csv').values
gts = vls[:,3].astype('int64')


vls = pd.read_csv('../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-1.csv').values
preds= vls[:,1:].astype('float32')

preds_gt = np.array([pred[gt] for pred,gt in zip(preds,gts)])

print(np.histogram(preds_gt))


n_class = 7
ce_loss = np.zeros((8,n_class))
np.set_printoptions(precision=4,suppress = True)
for nc in range(n_class):
    print(np.histogram(preds_gt[gts==nc])[0])
    
    pp = preds_gt[gts==nc]  + 0.001
    logv = np.log(pp)
    ce_loss[0,nc]= -np.sum(logv)
    ce_loss[1,nc]= -np.mean(logv)
    
    
    # gamma = 1
    gamma = 1.0
    logv = -np.power((1-pp),gamma) * np.log(pp)
    ce_loss[2,nc]= np.sum(logv)
    ce_loss[3,nc]= np.mean(logv)
    #gamma = 2
    gamma = 2.0
    logv = -np.power((1-pp),gamma) * np.log(pp)
    ce_loss[4,nc]= np.sum(logv)
    ce_loss[5,nc]= np.mean(logv)    
    #gamma = 3
    gamma = 3.0
    logv = -np.power((1-pp),gamma) * np.log(pp)
    ce_loss[6,nc]= np.sum(logv)
    ce_loss[7,nc]= np.mean(logv)    


print(ce_loss)





#%%

xx = np.arange(0.01,1.0,0.01)



gamma = 2.0
yy_g2 = -np.power((1-xx),gamma) * np.log(xx)

gamma = 1.0
yy_g1 = -np.power((1-xx),gamma) * np.log(xx)

gamma = 0.0
yy_g0 = -np.power((1-xx),gamma) * np.log(xx)


#plt.plot(xx,yy_g0,xx,yy_g1,xx,yy_g2)
plt.plot(xx,yy_g2/yy_g0,xx,yy_g1/yy_g0)