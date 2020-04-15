# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:54:30 2020

@author: Administrator
"""
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
fns = [#'../checkpoints/eval_resnet50_singleview-Loss-ce-tta-0-test.csv',

#
#       
#       '../checkpoints/19/eval_effnetb4_singleview-Loss-ce-tta-1-test.csv',
#       #'../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/19/eval_effnetb4_metasingleview-Loss-ce-tta-1-test.csv',
#       
#       '../checkpoints/19/eval_effnetb4_singleview-Loss-ce-tta-1-test.csv',
#       #'../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/19/eval_effnetb4_metasingleview-Loss-ce-tta-1-test.csv',   

       


       '../checkpoint/effb4_meta_default_19/eval_effb4_SVMeta-Loss-ce-tta-1-test.csv',

        '../checkpoint/resnet50_meta_default_19/eval_resnet50_SVMeta-Loss-ce-tta-1-test.csv',
       '../checkpoint/effb4_default_19/eval_effb4_SingleView-Loss-ce-tta-1-test.csv',
       
       ]
#fns = ['../checkpoints/eval_resnet50_singleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/eval_resnet50_singleview-Loss-ce-tta-1-test.csv'
#       
#       
#       ]

##mean mode

mean_mode = 'mean'  #'mean



k_cls = np.array([1.18620359, 2.02358709, 1.01884232, 0.51940186, 0.91387166,
       0.28223699, 0.2827877 , 0.44205242])
k_cls =np.minimum(1.0,k_cls)

for idx,fn in enumerate(fns):#
    #fn = fns[0]
    print('*'*32)
    print(fn)
    
    kl1 = pd.read_csv(fn).values
    n_samp = kl1.shape[0]
    n_class = 9
    
    
    
    
    y_pred = 2.0* np.ones((n_samp,n_class),dtype= 'float32')
    
    y_pred[:,:-1] = kl1[:,1:].astype('float32')
    
    
    y_torch = torch.from_numpy(y_pred)
    
    y_pred_cls_c8 = F.softmax(y_torch[:,:-1],dim = 1).numpy().argmax(axis = 1)
    
    k_mult = k_cls[y_pred_cls_c8]
    y_pred[:,-1] = y_pred[:,-1]*k_mult
    y_torch = torch.from_numpy(y_pred)
    
    y_pred = F.softmax(y_torch,dim = 1).numpy()    
        
    #y_pred = y_pred/1.0
    if idx==0:
        if mean_mode =='mean':
            y_pred_total = np.zeros((n_samp,n_class),dtype= 'float32')
        else:
            y_pred_total = np.ones((n_samp,n_class),dtype= 'float32')    
    
    if mean_mode =='mean':
        y_pred_total = y_pred_total + y_pred
    else:
        y_pred_total = y_pred_total * y_pred
        
if mean_mode =='mean':
    y_pred_total = y_pred_total /len(fns)
else:
    y_pred_total = np.power(y_pred_total,1.0/len(fns))
y_pred_total = y_pred_total /np.sum(y_pred_total,axis = 1,keepdims = True)
y_pred_total = np.round_(y_pred_total,decimals = 4)


dict_label = [ 'MEL', 'NV','BCC', 'AK', 'BKL', 'DF','VASC','SCC','UNK']

df = pd.DataFrame(data = y_pred_total,index =kl1[:,0], columns = dict_label)
for col in dict_label:
    df[col] = df[col].apply(lambda x: format(x,'.4f'))

    
df.to_csv('./dat/ISIC19_test_submit.csv',index_label = 'image')




#%%

for idx,pred in enumerate(y_pred_total):
    if np.max(pred)<=0.5:
        #change conf to 0.6 if all conf <0.5
        pos = np.argmax(pred)
        y_pred_total[idx, pos] = 0.6

y_pred_total = np.round_(y_pred_total,decimals = 4)

        

df = pd.DataFrame(data = y_pred_total,index =kl1[:,0], columns =dict_label)
for col in dict_label:
    df[col] = df[col].apply(lambda x: format(x,'.4f'))

    
df.to_csv('./dat/ISIC19_test_submit2_setmaxconf.csv',index_label = 'image')
    
    
