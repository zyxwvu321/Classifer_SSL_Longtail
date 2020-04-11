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
import math
from pathlib import Path
import shutil
from tqdm import tqdm
#%%




n_class = 7



fn1 = '../checkpoints/eval_effnetb4_singleview-Loss-ce-tta-1.csv'
fn2 = '../checkpoints/eval_effnetb4_metasingleview-Loss-ce-tta-1.csv'
gt  = pd.read_csv(fn1).values[:,-1]
pred1 = pd.read_csv(fn1).values[:,-2]
pred2 = pd.read_csv(fn2).values[:,-2]
#%% a test to read csv of gt infos

gts = pd.read_csv('./dat/ISIC18_info.csv').values
n_samp = gts.shape[0]
for n in range(n_samp):
    if math.isnan(gts[n,4]):
        gts[n,4] = 0.0
    if not isinstance(gts[n,5], str):
        gts[n,5] = 'unk'
    if not isinstance(gts[n,6], str):
        gts[n,6] = 'unk'

meta1 = gts[:,4]
meta2 = gts[:,5]
meta3 = gts[:,6]

np.unique(meta1,return_counts = True)
np.unique(meta2,return_counts = True)
np.unique(meta3,return_counts = True)

np.unique(meta1[gt==0],return_counts = True)
np.unique(meta2[gt==0],return_counts = True)
np.unique(meta3[gt==0],return_counts = True)


b_meta = (meta1!=0.0).astype('int64') +  (meta2!='unk').astype('int64')  + (meta3!='unk' ).astype('int64')
b_meta = b_meta==3.0


#%% evaluation what kind of cases improve for metamodel





fns = ['../checkpoints_pcs/eval_resnet50_SingleView-Loss-pcs-tta-0.csv',
       '../checkpoints_pcs/eval_resnet50_SingleView-Loss-pcs-tta-1.csv',
       '../checkpoints_pcs/eval_resnet50_SVMeta-Loss-pcs-tta-0.csv',
       '../checkpoints_pcs/eval_resnet50_SVMeta-Loss-pcs-tta-1.csv',
       '../checkpoints_pcs/eval_effb4_SingleView-Loss-pcs-tta-0.csv',
       '../checkpoints_pcs/eval_effb4_SingleView-Loss-pcs-tta-1.csv',
       '../checkpoints_pcs/eval_effb4_SVMeta-Loss-pcs-tta-0.csv',
       '../checkpoints_pcs/eval_effb4_SVMeta-Loss-pcs-tta-1.csv',



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
    print(fn)
    kl1 = pd.read_csv(fn).values[:,-2:]
    
    y_true = kl1[:,1]
    y_pred = kl1[:,0]
    y_true = np.array(y_true).astype('int64') 
    y_pred = np.array(y_pred).astype('int64') 
    
    
    
    cm = confusion_matrix(y_true, y_pred) 
    
    print(cm)
    print(bal_acc(cm))
    
    
    
        
#    map_v = np.array([1,0,1,1,0,0,0])
#    y_true = map_v[y_true]
#    y_pred = map_v[y_pred]
#
#    cm = confusion_matrix(y_true, y_pred) 
#    
#    print(cm)
#    print(bal_acc(cm))    
#    
#%% calc ave embedding
comb_list = [[0,4],[1,5], [2,6],[3,7],[1,3,5,7],[0,1,2,3,4,5,6,7]]    
##%% calc ave embedding
#comb_list = [[1,3,5,7]]    
gt = gt.astype('int64')
n_samp = len(gt)
for comb in comb_list:
    
    print('*'*32)
    print(comb)
    
    y_pred = np.zeros((n_samp,n_class),dtype= 'float32')
    for idx in comb:
        fn_sort = np.argsort(pd.read_csv(fns[idx]).values[:,0])
        kl1 = pd.read_csv(fns[idx]).values[fn_sort,1:-2]
        y_pred = y_pred + kl1.astype('float32')
    
    #y_pred = y_pred/len(comb)
        
    y_pred = np.argmax(y_pred,axis = 1).astype('int64')
    
    cm = confusion_matrix(gt, y_pred) 
    
    print(cm)
    print(bal_acc(cm))
    
    
    
        
#    map_v = np.array([1,0,1,1,0,0,0])
#    y_true = map_v[gt]
#    y_pred = map_v[y_pred]
#
#    cm = confusion_matrix(y_true, y_pred) 
#    
#    print(cm)
#    print(bal_acc(cm))    
    
    
#%% use 1,3,5,7 embedding, output a table with wrong prediction
comb = [1,3,5,7]
gt = gt.astype('int64')
n_samp = len(gt)

    
print('*'*32)
print(comb)

y_pred = np.zeros((n_samp,n_class),dtype= 'float32')
for idx in comb:
    fn_sort = np.argsort(pd.read_csv(fns[idx]).values[:,0])
    kl1 = pd.read_csv(fns[idx]).values[fn_sort,1:-2]
    y_pred = y_pred + kl1.astype('float32')

y_pred = y_pred/len(comb)
    
y_pred_idx = np.argmax(y_pred,axis = 1).astype('int64')


idx_wrong  = y_pred_idx!=gt


out_arr = np.hstack((y_pred[idx_wrong], y_pred_idx[idx_wrong,None], gt[idx_wrong,None]))
df = pd.DataFrame(data = out_arr.astype('float32'),index =gts[idx_wrong,0], columns = [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC','pred', 'GT'])
for col in [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']:
    df[col] = df[col].apply(lambda x: format(x,'.4f'))
for col in ['pred', 'GT']:
    df[col] = df[col].apply(lambda x: format(x,'.0f'))    

    
df.to_csv('./dat/wrong_emb.csv')

##%%copy image
#fd_im = '../data/ISIC18/task3/ISIC2018_Task3_Training_Input'
#fd_out = '../data/ISIC18/task3/wrong'
#types =  [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']
#for tp in types:
#    (Path(fd_out)/tp).mkdir(parents =True,exist_ok = True)
#
#for idx,fn in enumerate(tqdm(gts[idx_wrong,0])):
#    t_lb = int(out_arr[idx,-1])
#    p_lb = int(out_arr[idx,-2])
#    fn_in = str(Path(fd_im)/(fn + '.jpg'))
#    fn_out = str(Path(fd_out)/types[t_lb]/(fn + '_' + types[p_lb] +'.jpg'))
#    shutil.copyfile(fn_in,fn_out)
#

    
#    #%%
#
#b_meta1 = np.array([isinstance(age, (float,int)) and age!=0.0 for age in gts[:,4]]) #age
#b_meta2 = np.array([isinstance(pos, (str)) for pos in gts[:,5]]) #pos
#b_meta3 = np.array([isinstance(sex, (str)) for sex in gts[:,6]]) #sex
#
#
#
#idx_c1c2 = np.logical_and((gt==pred2),(gt==pred1))
#idx_c1w2 = np.logical_and((gt!=pred2),(gt==pred1))
#
#idx_w1c2 = np.logical_and((gt==pred2),(gt!=pred1))
#idx_w1w2 = np.logical_and((gt!=pred2),(gt!=pred1))
#
#
##np.bincount(gt[idx_w1c2].astype('int64'),minlength = 7)
#
##%%
#'''
# ['lower extremity', 2396], 下肢
# ['posterior torso', 2192], 臀部躯干
# ['upper extremity', 1208], 上肢
# ['anterior torso', 1429], 下侧躯干
# ['palms/soles', 7], 手掌
# ['head/neck', 1097]] 头颈
# {'male': 5408, 'female', 4560}
# {0:'MEL',恶性黑色素 1:'NV',痣 2:'BCC',基地细胞癌 3:'AKIEC'色素性鲍温病, 4:'BKL'脂溢性角化, 5:'DF'皮肤纤维瘤,6:'VASC' 血管性皮疹}
#0/2/3 恶性
#1/4/5/6 良性
#0以面部，手掌，足底多见
#'''
##%%%
#t1 = gt    == 0
#t2 = pred1 == 0
#t3 = pred2 == 0 
#
#i1 = (t1*t2*t3).sum()
#
#i2 = (t1*(1-t2)*t3).sum() #meta c
#i3 = (t1*(1-t3)*t2).sum() # ori c
#i4 = (t1*(1-t3)*(1-t2)).sum() # all wrong

