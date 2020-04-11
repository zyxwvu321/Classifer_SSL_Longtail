#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:09:02 2020

@author: minjie
"""
import globalvar as gl
import os.path  as osp
from config import cfg
from tqdm import tqdm
import torchvision
import torch
import numpy as np

from collections import Counter


from data.transforms.build import get_transform

from pathlib import Path
from modeling import build_model


#cfg.merge_from_file('./configs/resnet50_reid_bird_ave_centerloss.yaml')
#cfg.MISC.OUT_DIR  = '../checkpoint/resnet50_reid_flower_ave_centern'

#cfg.merge_from_file('./configs/resnet50_reid_flower_ave_centerloss.yaml')
#cfg.MISC.OUT_DIR  = '../checkpoint/resnet50_reid_flower_ave_centern1_lbs'


cfg.merge_from_file('./configs/resnet50_reid_flower_ave_centerloss.yaml')
cfg.MISC.OUT_DIR  = '../checkpoint/resnet50_reid_flower_ave_centern1'

cfg.DATASETS.K_FOLD = 1



gl._init()
gl.set_value('cfg',cfg)



#%% hawaii dataset and transform
fd_hawaiiflower = '../data/Hawaii_flowers'

flower_names = Path(fd_hawaiiflower).glob('*')




#
tfms =  get_transform((448,448), phase='valid')
ds = torchvision.datasets.ImageFolder(root=fd_hawaiiflower, transform=tfms)
labels = ds.classes
n_class = len(labels)



#%% build model
model = build_model(cfg) 
model.eval()
best_model_fn = osp.join(cfg.MISC.OUT_DIR, f"{cfg.MODEL.NAME}-best.pth")
model.load_state_dict(torch.load(best_model_fn))



#%% inference get feature
cls_feats = list()
lbs       = list()
for data in tqdm(ds):
    img,target = data
    img = img.cuda()
    with torch.no_grad():
        feat_out =model(img[None,...],target)
    cls_feats.append(feat_out[1][0].cpu().numpy())
    lbs.append(target)
    




#%% calc 1-center for one flower during querying (feat_m) this is not used, only for comparsion
countDict = Counter(lbs)
cls_feats = np.array(cls_feats)
lbs = np.array(lbs)
feat_m = list()

for cid in range(n_class):
    feat = cls_feats[lbs ==cid]
    feat = feat/np.linalg.norm(feat,axis = 1,keepdims = True)
    feat_m.append(feat.mean(axis = 0))



cos_vals = np.zeros((cls_feats.shape[0],n_class))
for cid in range(n_class):
    for imgid in range(cls_feats.shape[0]):
        
        feat = cls_feats[imgid]/np.linalg.norm(cls_feats[imgid],axis = 0,keepdims = True)
        cos_vals[imgid,cid] = np.dot(feat_m[cid],feat)




#%%  1-center  acc and cm
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=4,suppress = False)

pred_label = np.argmax(cos_vals,axis = 1)

cm = confusion_matrix(lbs.astype('int64'), pred_label.astype('int64')) 

(np.argmax(cos_vals,axis = 1) == lbs).sum()/cls_feats.shape[0]
print(cm)

print('acc top1 w/o th and center=1 for one class= {:03.03%}'.format(cm.diagonal().sum()/cm.sum()))

#%% test in-class dist

for cid in range(n_class):
    feat = cls_feats[lbs ==cid]

    
    feat = feat/np.linalg.norm(feat,axis = 1,keepdims = True)
    featm = feat.mean(axis = 0,keepdims = True)
    dists = np.dot(feat,featm.T)
    print(cid)
    print(dists.mean(),dists.min())


#%% now, calc k-center for one flower during querying (feat_m) this is for debug

from sklearn.cluster import KMeans

def calc_corr_diff_k(feat,max_k = 5):
    feat = feat/np.linalg.norm(feat,axis = 1,keepdims = True)
    
    min_corr,mean_corr = list(),list()
    
    
    for nk in range(max_k):
        if nk >0:
            y_pred = KMeans(n_clusters=nk+1).fit_predict(feat)
        else:
            y_pred = np.zeros(feat.shape[0],dtype = 'int32')
        
        
        featm_k = list()
        for k in range(nk+1):
            featm = feat[y_pred ==k].mean(axis = 0,keepdims = True)
            
            featm = featm/np.linalg.norm(featm,axis = 1,keepdims = True)
            featm_k.append(featm)
        
        if nk==max_k-1:
            featm_k_out = featm_k
        
        
        for k in range(nk+1):
            if k==0:
                cos_v = np.dot(feat,featm_k[k].T)
            else:
                cos_v = np.maximum(cos_v,np.dot(feat,featm_k[k].T))
        min_corr.append(cos_v.min())
        mean_corr.append(cos_v.mean())
        
    min_corr = np.array(min_corr)    
    mean_corr = np.array(mean_corr)    
    
    return min_corr,mean_corr,featm_k_out



min_corr_all,mean_corr_all = list(),list()
for cid in range(n_class):
    feat = cls_feats[lbs ==cid]
    min_corr,mean_corr,_ = calc_corr_diff_k(feat,max_k = 5)
    min_corr_all.append(min_corr)
    mean_corr_all.append(mean_corr)
    
    
#%% calc k-center (k=3) features for query, this is used
featm_k_all = list()
nk = 3
th_eval = 0.5
for cid in range(n_class):
    feat = cls_feats[lbs ==cid]

    
    #feat = feat/np.linalg.norm(feat,axis = 1,keepdims = True)
    #featm = feat.mean(axis = 0,keepdims = True)
    _,_,featm_k = calc_corr_diff_k(feat,max_k = nk)
    featm_k_all.append(featm_k)
    

cos_vals = -1.0*np.ones((cls_feats.shape[0],n_class))
for cid in range(n_class):
    for imgid in range(cls_feats.shape[0]):
        
        feat = cls_feats[imgid]/np.linalg.norm(cls_feats[imgid],axis = 0,keepdims = True)

        for k in range(nk):
            corr_val = np.dot(featm_k_all[cid][k],feat)
            if corr_val>cos_vals[imgid,cid]:
                cos_vals[imgid,cid] = corr_val


pred_label = np.argmax(cos_vals,axis = 1)

cm = confusion_matrix(lbs.astype('int64'), pred_label.astype('int64')) 




# this is the case when non-flower not considered
(np.argmax(cos_vals,axis = 1) == lbs).sum()/cls_feats.shape[0]

print(cm)
#print(cm.diagonal().sum()/cm.sum())
print('acc top1 w/o th = {:03.03%}'.format(cm.diagonal().sum()/cm.sum()))




# this is the case when non-flower  considered
cos_vals_p = np.hstack((np.ones((cos_vals.shape[0],1))*th_eval,cos_vals))
import torch.nn.functional as F
probs = F.softmax(torch.from_numpy(cos_vals_p)*25,dim=1) # THIS IS THE PROBS of all the images, the first is non-flower
pred_label_withnobj = probs.argmax(dim=1).numpy()


acc_th = (pred_label_withnobj==lbs+1).sum()/len(lbs)
print('acc top1 with th = {:03.03%}'.format(acc_th))



#prob
#
#
#torch.topk(torch.from_numpy(cos_vals),2,dim=1)





#%% a test of coco set val17 if it is returned as fp
#flist = list(Path('../data/coco/val2017').glob('*.jpg'))
cls_feats_noobj = list()
ds_noobj = torchvision.datasets.ImageFolder(root='../data/coco', transform=tfms)
for data in tqdm(ds_noobj):
    img,target = data
    img = img.cuda()
    with torch.no_grad():
        feat_out =model(img[None,...],target)
    cls_feats_noobj.append(feat_out[1][0].cpu().numpy())


#%% Here, we tested how many coco val images return top-1 with given flower
cls_feats_noobj = np.array(cls_feats_noobj)
cos_vals_noobj = -1.0*np.ones((cls_feats_noobj.shape[0],n_class))
for cid in range(n_class):
    for imgid in range(cos_vals_noobj.shape[0]):
        
        feat = cls_feats_noobj[imgid]/np.linalg.norm(cls_feats_noobj[imgid],axis = 0,keepdims = True)

        for k in range(nk):
            corr_val = np.dot(featm_k_all[cid][k],feat)
            if corr_val>cos_vals_noobj[imgid,cid]:
                cos_vals_noobj[imgid,cid] = corr_val
                
cos_vals_p_nonobj = np.hstack((np.ones((cos_vals_noobj.shape[0],1))*th_eval,cos_vals_noobj))
probs = F.softmax(torch.from_numpy(cos_vals_p_nonobj)*25,dim=1) # THIS IS THE PROBS of all the images, the first is non-flower
pred_label_withnobj_fp = probs.argmax(dim=1).numpy()


fp_prob = (pred_label_withnobj_fp!=0).sum()/cls_feats_noobj.shape[0]
print('fp for coco with th = {:03.03%}'.format(fp_prob))

fns_fp =np.array(ds_noobj.imgs)[cos_vals_noobj.max(axis = 1)>th_eval]
fns_fp[:,1] = cos_vals_noobj.argmax(axis = 1)[cos_vals_noobj.max(axis = 1)>th_eval]


