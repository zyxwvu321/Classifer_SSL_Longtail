# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:40:16 2020

@author: cmj
"""
from dataset.custom_dataset import CustomDataset_withmeta

xx = CustomDataset_withmeta(root ='../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj')


aa = xx[7635]
#meta_age = parse_age(np.array(info_list)[:,3], th_list=np.arange(5,90,5))
#box_inv  = inv_letterbox_bbox(box_pred_abs[None,...].astype('float32'), box_dim, img.shape[:2])
#from dataset.sampler import resample_idx_with_meta
#from pathlib import Path
#flist = list(Path('D:\dataset\ISIC\ISIC2018_Task3_Training_Input').glob('*.jpg'))
#tr_idx,vl_idx = resample_idx_with_meta(flist = flist)