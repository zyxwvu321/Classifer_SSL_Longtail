#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:29:38 2019

@author: minjie

script to run dicom ori input and predict pos/neg
AP/LAT with same fname in one folder, end with 0.dcm & 1.dcm

"""
import cv2
import torch
import modeling.models as models
from pathlib import Path

import pydicom
#import matplotlib.pyplot as plt
from utils.utils import del_mkdirs,extend_box
from tqdm import tqdm
from transforms.data_preprocessing import  TestAugmentation_albu,TestAugmentation_bbox_albu,TrainAugmentation_albu

from datasets.custom_dataset import  CustomDataset_bbox
from modeling.location_box_loss import model_box_to_xy_ori,norm_box_to_abs
from configs import resnet34_c3_locate




#%% convert dicom to png,  
path_dicom = './data/ori/ori_dicom'
path_png   = './data/ori/png_ori_test'
path_png_crp_anno   = './data/ori/png_crp_anno'
path_png_crp = './data/ori/png_crp_test'




fd_list = [str(Path(path_dicom)/'pos'),str(Path(path_dicom)/'neg')]#,str(Path(path_dicom)/'unknown')]
#fd_list = [str(Path(path_dicom)/'unknown')]

#%% net to DEVICE
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')



#%%
#fd_list_png = str(Path(fd_list[0].replace(Path(path_dicom).stem,'png_ori_test')).parents[0])
del_mkdirs([path_png])

#del_mkdirs([str(Path(fd_list_png[0]).parents[0])])

for fds in fd_list:
    flist = [str(fn) for fn in sorted(Path(fds).glob('*.dcm'))]
    
    for fn in flist:
        
        with pydicom.dcmread(fn) as dc:
            img_dicom = dc.pixel_array
            print(img_dicom.shape, img_dicom.shape[0]/img_dicom.shape[1])
            
            
            fname_png = fn.replace('.dcm','.png').replace(Path(path_dicom).stem,Path(path_png).stem).replace('0.png','_AP.png') \
            .replace('1.png','_LAT.png').replace('pos/','Pos').replace('neg/','Neg').replace('unknown/', 'Un')
            if 'par' in fn:
                #patient with right knee
                cv2.imwrite(fname_png, (img_dicom[:,::-1]/16384.0*255).astype('uint8'))
            else:                
                cv2.imwrite(fname_png, (img_dicom/16384.0*255).astype('uint8'))
            

#%% crop image(image localiazation)   
    
model = models.LocateBoneModel_CustomSquare()
    
configs = resnet34_c3_locate

model.load_state_dict(torch.load('./checkpoints/resnet18_c3_locate-best-cpu.pth'))
model = model.to(device)
model.eval()



test_transform = TestAugmentation_bbox_albu(configs.image_size, configs.image_mean, configs.image_std)
valid_dataset =  CustomDataset_bbox(root = path_png, transform = test_transform)

del_mkdirs([path_png_crp,path_png_crp_anno])


print('\n')
print('run model for cropping image \n')
print('\n')



ex_p = 0.1875#configs.ext_p[1] #0.25 # 512+ 512/8 = 512+64 =  
crp_sz = int(512*(1+ex_p))

for idx,ds in enumerate(tqdm(valid_dataset)):
    image_id = valid_dataset.ids[idx]
    img_fn_ori =  Path(path_png) / (image_id +'.png')
    image_ori = cv2.imread(str(img_fn_ori))
    
    ori_im_shape = image_ori.shape
    
    images, box_gt = ds
    
    images = images.unsqueeze(dim=0).to(device)
   
    with torch.no_grad():        
        outputs = model(images)
    
    box_pred = model_box_to_xy_ori(outputs.clone(),ori_im_shape)    
    box_pred_abs0 = norm_box_to_abs(box_pred[0],ori_im_shape).astype('int')
    
   
    box_pred_abs = extend_box(box_pred_abs0,ori_im_shape,ex_p) 
        
    cv2.imwrite(str(img_fn_ori).replace(Path(path_png).stem,Path(path_png_crp).stem),
                cv2.resize(image_ori[box_pred_abs[1]:box_pred_abs[3],box_pred_abs[0]:box_pred_abs[2],:], (crp_sz,crp_sz)))
    
    
    cv2.rectangle(image_ori, (box_pred_abs[0], box_pred_abs[1]), (box_pred_abs[2], box_pred_abs[3]), (0, 0, 255), 4)
    cv2.imwrite(str(Path(path_png_crp_anno)/img_fn_ori.name),image_ori)
    

