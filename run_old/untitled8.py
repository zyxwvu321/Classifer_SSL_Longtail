# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:52:34 2020

@author: cmj
"""
import numpy as np
def calc_metric(spe,ppv,npv,n):
    
   #k_fntp = (1-sen)/sen
    k_fptn = (1-spe)/spe
    
    k_fptp = (1-ppv)/ppv
    k_fntn = (1-npv)/npv
    
    
    k_tptn = 1.0/k_fptp *k_fptn
    
    tn = n/(k_fptn + 1.0 + k_fntn + k_tptn)
    tp = k_tptn * tn
    fp = k_fptn * tn
    fn = k_fntn * tn
    
    return (tp,tn,fp,fn)



n = 1512


spe = np.array([0.953,	0.904,	0.988,	0.988,	0.962,	0.996,	0.997])
ppv = np.array([0.659,	0.934,	0.823,	0.660,	0.776,	0.846,	0.897])
npv = np.array([0.963,	0.864,	0.990,	0.995,	0.964,	0.993,	1.00])

metrics = calc_metric(spe,ppv,npv,n)
metrics = np.array(metrics)
metrics = np.round(np.transpose(metrics,[1,0]))
print(metrics)
# {0:'MEL',恶性黑色素 1:'NV',痣 2:'BCC',基地细胞癌 3:'AKIEC'色素性鲍温病, 
#4:'BKL'脂溢性角化, 5:'DF'皮肤纤维瘤,6:'VASC' 血管性皮疹}

spe = np.array([0.948,	0.920,	0.988,	0.986,	0.964,	0.996,	0.997])
ppv = np.array([0.643,	0.945,	0.825,	0.625,	0.788,	0.850,	0.897])
npv = np.array([0.966,	0.860,	0.991,	0.995,	0.964,	0.993,	1.00])

metrics = calc_metric(spe,ppv,npv,n)
metrics = np.array(metrics)
metrics = np.round(np.transpose(metrics,[1,0]))
print(metrics)



