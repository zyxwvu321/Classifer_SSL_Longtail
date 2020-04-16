#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:36:15 2020
preproc new images



@author: minjie
"""


from pathlib import Path
import os

import cv2
from tqdm import tqdm
from color_enh import sod_minkowski

import pandas as pd
import numpy as np


def str2id(num, prefix = None):
    nstr = str(num)
    if len(nstr)<=5:
        nstr = '0'*(5-len(nstr)) + nstr
    if prefix is not None:
        nstr = prefix + '_' + nstr
    nstr = nstr + '.jpg'
    return nstr


#%%

tar_gt = './dat/extra_GT.csv'

fd_out = '../data/extra_all'
dict_labels = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC','SCC','UNK']
os.makedirs(fd_out,exist_ok = True)


#%% jiuyuan

fd_in = '../data/jiuyuan'
flist = sorted(list(Path(fd_in).rglob('*.*')))
flist = [str(fn) for fn in flist]
nid = 1
fn_gts = list()

for img_fn in tqdm(flist):    
    if not Path(img_fn).is_file or Path(img_fn).suffix.lower() not in ['.jpg','.bmp']:
        continue
    
    
    img = cv2.imread(img_fn)
    if img is None:
        continue
    
    hh,ww,_ = img.shape
    
    if min(hh,ww)>1024: # limit image size
        rt = max(1024.0/hh,1024.0/ww)
        tar_hw = (int(rt*ww),int(rt*hh))
        img = cv2.resize(img,tar_hw,cv2.INTER_CUBIC)
    
    
    
        
    
    str_id = str2id(nid,'jiuyuan')
    if Path(img_fn).parts[-2]  in dict_labels:
        fn_gts.append([Path(str_id).stem ,    dict_labels.index(Path(img_fn).parts[-2])])
    else:
        fn_gts.append([Path(str_id).stem ,   len(dict_labels)-1 ])
        
    cv2.imwrite(str(Path(fd_out)/str_id),(img).astype('uint8'),[int(cv2.IMWRITE_JPEG_QUALITY),100])
    nid = nid +1


#%% mednode
fd_in = '../data/complete_mednode_dataset'
flist = sorted(list(Path(fd_in).rglob('*.*')))
flist = [str(fn) for fn in flist]
nid = 1


for img_fn in tqdm(flist):    
    if not Path(img_fn).is_file or Path(img_fn).suffix.lower() not in ['.jpg','.bmp']:
        continue
    
    
    img = cv2.imread(img_fn)
    if img is None:
        continue
    
    hh,ww,_ = img.shape
    
    if min(hh,ww)>1024: # limit image size
        rt = max(1024.0/hh,1024.0/ww)
        tar_hw = (int(rt*ww),int(rt*hh))
        img = cv2.resize(img,tar_hw,cv2.INTER_CUBIC)
    
    
    
        
    
    str_id = str2id(nid,'mednode')
    if Path(img_fn).parts[-2] == 'melanoma':
        fn_gts.append([Path(str_id).stem ,    0])
        
    elif Path(img_fn).parts[-2] == 'naevus':
        fn_gts.append([Path(str_id).stem ,   1 ])
    else:
        raise ValueError('unk type')
        
    cv2.imwrite(str(Path(fd_out)/str_id),(img).astype('uint8'),[int(cv2.IMWRITE_JPEG_QUALITY),100])
    nid = nid +1

#%% sevenpoint
fd_in = '../data/release_v0/images'
fd_d = '../data/release_v0/images_d'
os.makedirs(fd_d)
csv_fn = '../data/release_v0/meta/meta.csv'
vals = pd.read_csv(csv_fn).values
diagnosis = vals[:,1]
fns   = vals[:,-3]




nid = 1


for fn,dg in tqdm(zip(fns,diagnosis)):    
    img_fn = Path(fd_in)/fn
    if not Path(img_fn).is_file or Path(img_fn).suffix.lower() not in ['.jpg','.bmp']:
        continue
    
    
    img = cv2.imread(str(img_fn))
    if img is None:
        continue
    
    hh,ww,_ = img.shape
    
    if min(hh,ww)>1024: # limit image size
        rt = max(1024.0/hh,1024.0/ww)
        tar_hw = (int(rt*ww),int(rt*hh))
        img = cv2.resize(img,tar_hw,cv2.INTER_CUBIC)
    
    
    
        
    
    str_id = str2id(nid,'sevenpoint')
    #['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC','SCC','UNK']
    if 'nevus' in  dg:
        fn_gts.append([Path(str_id).stem ,    1])
        tp = 'NV'
    elif 'melanoma' in  dg:
        fn_gts.append([Path(str_id).stem ,    0])
        tp = 'MEL'
    elif 'seborrheic keratosis' in  dg:
        fn_gts.append([Path(str_id).stem ,    4])
        tp = 'BKL'
    elif 'basal cell carcinoma' in  dg:
        fn_gts.append([Path(str_id).stem ,    2])
        tp = 'BCC'
    elif 'dermatofibroma' in  dg:
        fn_gts.append([Path(str_id).stem ,    5])
        tp = 'DF'
    elif 'vascular lesion' in  dg:
        fn_gts.append([Path(str_id).stem ,    6])
        tp = 'VASC'
    else:
        fn_gts.append([Path(str_id).stem ,    8])
        tp = 'UNK'
        
    cv2.imwrite(str(Path(fd_out)/str_id),(img).astype('uint8'),[int(cv2.IMWRITE_JPEG_QUALITY),100])
    cv2.imwrite(str(Path(fd_d)/ (tp+ str_id)),(img).astype('uint8'),[int(cv2.IMWRITE_JPEG_QUALITY),100])
    
    
    nid = nid +1

#%% PH2
fd_in = '../data/PH2Dataset/PH2 Dataset images'
fd_d = '../data/PH2Dataset/image_d'
os.makedirs(fd_d)
csv_fn = '../data/PH2Dataset/PH2_dataset_gt.txt'
vals = pd.read_csv(str(csv_fn)).values
diagnosis = vals[:,2].astype('int64')
fns   = vals[:,0]

nid = 1

for fn0,dg in tqdm(zip(fns,diagnosis)):    
    fn = fn0.strip()
    img_fn = Path(fd_in)/fn/(fn + '_Dermoscopic_Image')/(fn +'.bmp')
    if not Path(img_fn).is_file or Path(img_fn).suffix.lower() not in ['.jpg','.bmp']:
        continue
    
    
    img = cv2.imread(str(img_fn))
    if img is None:
        continue
    
    hh,ww,_ = img.shape
    
    if min(hh,ww)>1024: # limit image size
        rt = max(1024.0/hh,1024.0/ww)
        tar_hw = (int(rt*ww),int(rt*hh))
        img = cv2.resize(img,tar_hw,cv2.INTER_CUBIC)
    
    
    
        
    
    str_id = str2id(nid,'ph2dataset')
    #['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC','SCC','UNK']
    
    if dg in [0,1]:
        fn_gts.append([Path(str_id).stem ,   1])
        tp = 'NV'
    elif dg==2:
        fn_gts.append([Path(str_id).stem ,   0])
        tp = 'MEL'
    
    
    cv2.imwrite(str(Path(fd_out)/str_id),(img).astype('uint8'),[int(cv2.IMWRITE_JPEG_QUALITY),100])
    cv2.imwrite(str(Path(fd_d)/ (tp+ str_id)),(img).astype('uint8'),[int(cv2.IMWRITE_JPEG_QUALITY),100])
    nid = nid +1


#%%  create csv
#fn_gts = np.array(fn_gts)
n_img = len(fn_gts)
gt_np = np.zeros((n_img,len(dict_labels)),dtype='int64')
fns = list()
for idx,gt in enumerate(fn_gts):
    gt_np[idx, gt[1]] = 1
    fns.append(gt[0])
    





df = pd.DataFrame(data = gt_np, columns = dict_labels,index =fns )
df.to_csv(tar_gt,index_label = 'image')    
