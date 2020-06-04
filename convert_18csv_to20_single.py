# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:01:58 2020
use 18 pred result to generate 20 submit table
@author: cmj
"""
import pandas as pd
import numpy as np


label18 = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

map_18  = [1,0,1, 1,0,0,0]



fn_csv_pred = '../checkpoint/18pred_fl.csv'
fn_csv_out= '../checkpoint/20pred_fl_convert_mc.csv'

meta_fn = '../data/ISIC20/test.csv'


pred_18 = pd.read_csv(fn_csv_pred).values

meta_20 = pd.read_csv(meta_fn).values


dict_p = dict()
dict_pn = dict()


for pp in pred_18:
    img_fn = pp[0]
    pred = pp[1:].astype('float32')
    
    idx = np.where(meta_20[:,0]==img_fn)[0][0]
    pid = meta_20[idx,1]
    
    if pid in dict_p.keys():
        
        dict_p[pid] += pred
        dict_pn[pid] +=1
    else:
        dict_p[pid] = pred
        dict_pn[pid] =1




dict_label = [ 'target']
fn_list = meta_20[:,0]
targets = list()
for meta,preds in zip(meta_20,pred_18):
    fn = meta[0]
    pid = meta[1]
    
    pred = preds[1:]
    p1 = pred[0]+pred[2]+pred[3]
    p0 = 1-p1
    
#    p1 = pred[0]
#    p0 = pred[1]
#    p1 = p1/(p1+p0)
    
    #pos = np.argmax(pred)
    #val = map_18[pos]
    targets.append(p1)




df = pd.DataFrame(data = targets,index =fn_list, columns = dict_label)


    
df.to_csv(fn_csv_out,index_label = 'image_name')