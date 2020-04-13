# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:54:30 2020

@author: Administrator
"""
import pandas as pd

import numpy as np

fns = [#'../checkpoints/eval_resnet50_singleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/eval_resnet50_singleview-Loss-ce-tta-1-test.csv',
#       #'../checkpoints/eval_resnet50_metasingleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/eval_resnet50_metasingleview-Loss-ce-tta-1-test.csv',
#       #'../checkpoints/eval_effnetb4_singleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/eval_effnetb4_singleview-Loss-ce-tta-1-test.csv',
#       #'../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-1-test.csv',
#       
#
#       
#       '../checkpoints/19/eval_effnetb4_singleview-Loss-ce-tta-1-test.csv',
#       #'../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/19/eval_effnetb4_metasingleview-Loss-ce-tta-1-test.csv',
#       
#       '../checkpoints/19/eval_effnetb4_singleview-Loss-ce-tta-1-test.csv',
#       #'../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/19/eval_effnetb4_metasingleview-Loss-ce-tta-1-test.csv',   

       

       '../checkpoint/eval_resnet50_SVMeta-Loss-ce-tta-1-test.csv',
       #'../checkpoint/eval_resnet50_SingleView-Loss-ce-tta-1-test.csv',
       #'../checkpoint/eval_resnet50_SVBNN-Loss-ce-tta-1-test.csv',
       '../checkpoint/eval_effb4_SingleView-Loss-ce-tta-1-test.csv',
       
       '../checkpoint/eval_effb4_SVMeta-Loss-ce-tta-1-test.csv',
       '../checkpoint/eval_sk50_SVMeta-Loss-ce-tta-1-test.csv',
       '../checkpoint/eval_effb3_SVMeta-Loss-ce-tta-1-test.csv'
       
       
       
       ]
#fns = ['../checkpoints/eval_resnet50_singleview-Loss-ce-tta-0-test.csv',
#       '../checkpoints/eval_resnet50_singleview-Loss-ce-tta-1-test.csv'
#       
#       
#       ]

##mean mode

mean_mode = 'gmean'  #'mean

if mean_mode =='mean':
    y_pred_total = np.zeros((1512,7),dtype= 'float32')
else:
    y_pred_total = np.ones((1512,7),dtype= 'float32')
for fn in fns:#
    #fn = fns[0]
    print('*'*32)
    print(fn)
    
    kl1 = pd.read_csv(fn).values
    n_samp = kl1.shape[0]
    n_class = 7
    
    y_pred = np.zeros((n_samp,n_class),dtype= 'float32')
    pos = 0
    for kk in range(1):
        y_pred = y_pred + kl1[pos:pos+n_samp,1:].astype('float32')
        pos = pos +n_samp
    #y_pred = y_pred/1.0
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



df = pd.DataFrame(data = y_pred_total,index =kl1[:n_samp,0], columns = [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC'])
for col in [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']:
    df[col] = df[col].apply(lambda x: format(x,'.4f'))

    
df.to_csv('./dat/ISIC18_test_submit.csv',index_label = 'image')



y_pred = np.argmax(y_pred_total,axis = 1)
map_name = np.array([ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC'])
y_label = map_name[y_pred]


df = pd.DataFrame(data = np.hstack((y_pred_total,y_label[:,None])),index =kl1[:n_samp,0], columns = [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC','pred'])
#for col in [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']:
#    df[col] = df[col].apply(lambda x: format(x,'.4f'))

    
df.to_csv('./dat/ISIC18_test_submi_predt.csv',index_label = 'image')

#%%

for idx,pred in enumerate(y_pred_total):
    if np.max(pred)<=0.5:
        #change conf to 0.6 if all conf <0.5
        pos = np.argmax(pred)
        y_pred_total[idx, pos] = 0.6

y_pred_total = np.round_(y_pred_total,decimals = 4)

        

df = pd.DataFrame(data = y_pred_total,index =kl1[:n_samp,0], columns = [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC'])
for col in [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']:
    df[col] = df[col].apply(lambda x: format(x,'.4f'))

    
df.to_csv('./dat/ISIC18_test_submit2_setmaxconf.csv',index_label = 'image')
    
    
