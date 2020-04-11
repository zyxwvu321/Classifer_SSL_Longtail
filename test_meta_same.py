import pandas as pd
import numpy as np
#from pathlib import Path
from tqdm import tqdm
#import cv2
#%%
meta_file = 'D:/dataset/ISIC/ISIC_2019_Training_Metadata.csv'   

df = pd.read_csv(meta_file)

df_v = df.values

n_r = df_v.shape[0]


prop = []
n_nan = []
for idx1 in tqdm(range(n_r)):
    r1 = df_v[idx1,1:]
    prop.append(str(r1[0]) + str(r1[1])+str(r1[2])+str(r1[3]))
    n_nan.append(pd.isnull(r1).sum())

uni_meta,uni_meta_count =np.unique(prop,return_counts=True)

aa=1
fname_all = []
for meta,count in zip(tqdm(uni_meta),uni_meta_count):
    if count>1 and count<10:#when <10 same img has same meta
        idx = [id for id,p in enumerate(prop) if p==meta]
        
        if n_nan[idx[0]]<=0:

            fname = []
            for id in idx:
                fname.append(df_v[id,0])
            fname_all.append(fname)
            #print(fname)
f=open('data_same.txt','w')  
for fname in fname_all:
    f.write(' '.join(fname))
    f.write('\n')
    #f.writelines(fname)

f.close() 



#%% evalate if same meta data img has same GT
gt_file = 'D:/dataset/ISIC/ISIC_2019_Training_GroundTruth.csv'   

df_gt = pd.read_csv(gt_file)
df_gt_v = df_gt.values

for fname in tqdm(fname_all):
    gts = []
    for fn in fname:
        idx = np.where(df_gt_v[:,0] == fn)[0][0]
        gt = np.where(df_gt_v[idx][1:]==1)[0][0]
        gts.append(gt)
    
    
    if len(set(gts))>1:
        print('error gt for same meta')
    



#%% save a ISICid map list

map_fns = dict()
map_fns_remap = list()
idx_map = 0
for idx, fn in enumerate(tqdm(df_v[:,0])):
    map_fns[fn] = idx
    
    idx_g = -1
    for idxm ,fl_in_meta in enumerate(fname_all):
        if fn in fl_in_meta:
            idx_g = fl_in_meta.index(fn)
            if idx_g!=-1:
                n_g_meta = idxm
                break
    
    if idx_g==-1:
        
        map_fns_remap.append(idx_map)
        idx_map = idx_map +1
    elif idx_g==0:
        map_fns_remap.append(idx_map)
        idx_map = idx_map +1
    else:
        
        map_fns_remap.append(map_fns_remap[map_fns[fl_in_meta[0]]])

import torch
dict_sav = dict()
dict_sav['fns'] = map_fns
dict_sav['fn_map'] = np.array(map_fns_remap)
torch.save(dict_sav,'./dat/fn_maps.pth')



#%% change if map is correct
map_fns_remap = np.array(map_fns_remap)
ids = np.where(map_fns_remap==12655)[0]

df_v[ids]
df_gt_v[ids]
ids = np.where(map_fns_remap==12990)[0]

df_v[ids]
df_gt_v[ids]

#         fname = []
#         for id in idx:
#             fname += df_v[id,0]
#         print(fname)
# aa=1

# for idx1 in tqdm(range(n_r)):
#     for idx2 in range(idx1+1,n_r):
#         r1 = df_v[idx1,1:]
#         r2 = df_v[idx2,1:]

#         n_nan1 = pd.isnull(r1).sum()
#         #n_nan2 = np.isnan(r2).sum()

#         d12 = (r1==r2).sum()
#         if d12==4 and n_nan1>0:
#             print(f'{df_v[idx1,0]} {df_v[idx2,0]}')
