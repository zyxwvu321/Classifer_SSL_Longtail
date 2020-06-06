#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:17:20 2020

@author: minjie
"""
import numpy as np
import math
def parse_age(age_list, th_list=np.arange(5,90,5)): 
    #parse age to +1/-1/0 0(NaN)
    
    age_list = np.array(age_list)
    n_sample = len(age_list)
    n_dim = len(th_list)
    th_list = np.array(th_list).astype('float32')
    
    meta_age = np.zeros((n_sample,n_dim),dtype = 'float32')
    
    for idx,age in enumerate(age_list):
        age =float(age)
        if age ==0.0  or math.isnan(age) is True:
            pass
        else:
            idx_th = np.where(th_list<age)[0]
            if len(idx_th)>0:
                idx_th = idx_th[-1]+1
                meta_age[idx][:idx_th] = 1
                meta_age[idx][idx_th:] = -1
            else:
                meta_age[idx][:] = -1
             
    return meta_age
            
        
    
def parse_sex(sex_list): 
    #parse age to +1(male)/-1(female)/0 0(NaN)
    
    sex_list = np.array(sex_list)
    n_sample = len(sex_list)
    

    meta_sex = np.zeros((n_sample,1),dtype = 'float32')
    
    for idx,sex in enumerate(sex_list):
        if isinstance(sex,str): 
            if sex.lower() =='male':
                meta_sex[idx] = 1
            elif sex.lower() =='female':
                meta_sex[idx] = -1
            else: 
                raise ValueError('wrong sex input')
        elif math.isnan(sex) is True:
            pass
        else:
            raise ValueError('wrong pos input')
            
             
    return meta_sex




def parse_pos(pos_list, all_pos): 
    #parse pos to +1(is the pos)/ 0(not the position) 
    
    pos_list = np.array(pos_list)
    n_sample = len(pos_list)
    
    n_dim = len(all_pos)
    meta_pos = np.zeros((n_sample,n_dim),dtype = 'float32')
    
    for idx,pos in enumerate(pos_list):
        
        if isinstance(pos,str):
        
            pos = pos.lower()
            if pos in all_pos:
                # in ISIC19, there are more position ,need fix
                pos_idx = all_pos.index(pos)    
                meta_pos[idx][pos_idx] = 1
        elif math.isnan(pos) is True:
            pass
        else:
            raise ValueError('wrong pos input')
             
    return meta_pos


def parse_boxsz(hw, box_info):
    assert hw.shape[0]==box_info.shape[0], 'parse_boxsz, sample not match'
    
    hw = hw.astype('float32')
    box_info = box_info.astype('float32')
    boxsz = np.zeros((hw.shape[0], 2), dtype = 'float32')
    
    
    boxsz[:,0] = (box_info[:,3] - box_info[:,1])/hw[:,0]
    boxsz[:,1] = (box_info[:,2] - box_info[:,0])/hw[:,1]
    return boxsz
     
            
def parse_kpds(kp_aug,hw_in,hw_out):
    hin,win = hw_in
    hout,wout = hw_out
    #points = [[ww/2.0,hh/2.0,1.0],[0.0,0.0,1.0]]
    pt_center, pt_corner = kp_aug
    
    
    
    d_in = math.sqrt((hin/2.0)**2+(win/2.0)**2)
    d_out = math.sqrt((hout/2.0)**2+(wout/2.0)**2)
    
    d_in_t = math.sqrt((pt_center[0]-pt_corner[0])**2+(pt_center[1]-pt_corner[0])**2)
    
    ss = d_out/d_in_t - 0.7
    
    
    d_c = math.sqrt((pt_center[0]-hout/2.0)**2+(pt_center[1]-wout/2.0)**2)
    d_c_ori = d_c / pt_center[2] # scale adjustment
    
    dc_sft = d_c_ori/d_in
    

    

    return (dc_sft,ss)