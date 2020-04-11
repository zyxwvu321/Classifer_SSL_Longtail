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
from configs import resnet18_c3_locate




#%% convert dicom to png,  
path_dicom = './data/ori/ori_dicom'
path_png   = './data/ori/png_ori_test'
path_png_crp_anno   = './data/ori/png_crp_anno'
path_png_crp = './data/ori/png_crp_test'
path_png_hm = './data/ori/png_crp_test_hm'

use_tta = False


fd_list = [str(Path(path_dicom)/'pos'),str(Path(path_dicom)/'neg'),str(Path(path_dicom)/'unknown')]
fd_list = [str(Path(path_dicom)/'unknown')]

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
            .replace('1.png','_LAT.png').replace('pos/','Pos').replace('neg/','Neg').replace('unknown/', '')
            if 'par' in fn:
                cv2.imwrite(fname_png, (img_dicom[:,::-1]/16384.0*255).astype('uint8'))
            else:                
                cv2.imwrite(fname_png, (img_dicom/16384.0*255).astype('uint8'))
            
            
            
            #x = pil2tensor(img_dicom,np.float32).div_(16384)
            #Image_AP = Image(x)
            

#%% crop image(image localiazation)   
    
model = models.LocateBoneModel_CustomSquare()
    
configs = resnet18_c3_locate

model.load_state_dict(torch.load('./checkpoints/resnet18_c3_locate-best-cpu.pth'))
model = model.to(device)
model.eval()



test_transform = TestAugmentation_bbox_albu(configs.image_size, configs.image_mean, configs.image_std)
valid_dataset =  CustomDataset_bbox(root = path_png, transform = test_transform)

del_mkdirs([path_png_crp,path_png_crp_anno])


print('\n')
print('run model for cropping image \n')
print('\n')

ex_p = 0.0
if use_tta:
    ex_p = 0.125

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
        
    cv2.imwrite(str(img_fn_ori).replace(Path(path_png).stem,Path(path_png_crp).stem),image_ori[box_pred_abs[1]:box_pred_abs[3],box_pred_abs[0]:box_pred_abs[2],:]) 
    cv2.rectangle(image_ori, (box_pred_abs[0], box_pred_abs[1]), (box_pred_abs[2], box_pred_abs[3]), (0, 0, 255), 4)
    cv2.imwrite(str(Path(path_png_crp_anno)/img_fn_ori.name),image_ori)
    
#%% test crop data
from configs import resnet34_c3_v2
from datasets.custom_dataset import  CustomDataset
import torch.nn.functional as F
import numpy as np
import modeling.models as models
configs = resnet34_c3_v2


     
test_transform = TestAugmentation_albu(configs.image_size, configs.image_mean, configs.image_std)            


if use_tta:
    test_transform = TrainAugmentation_albu(configs.image_size, configs.image_mean, configs.image_std)
    n_tta = 10
   


valid_dataset =  CustomDataset(root = path_png_crp, transform = test_transform,input_channels = configs.input_channels)

model = models.SplitBoneModel(configs.input_channels)
model.load_state_dict(torch.load('./checkpoints/resnet34_twoview-cpu.pth'))
model.to(device)
model.eval()


PREDS_ALL = []

print('\n')
print('run model for classification \n')
print('\n')

for idx,ds in enumerate(tqdm(valid_dataset)):

    if use_tta:
        pred_sum = 0.0
        for n_t in range(n_tta):
            img,label = ds
            fname  =  valid_dataset.ids[idx]
            with torch.no_grad():
                preds = model(img.unsqueeze(dim=0).to(device))
                preds = F.softmax(preds,dim=1)
                pred_sum += preds[0,1].item()
        pred_ave = pred_sum /n_tta
        PREDS_ALL.append([fname , np.round_(pred_ave,decimals=4),pred_ave>0.5,label])
                
    else:
        img,label = ds
        fname  =  valid_dataset.ids[idx]
        with torch.no_grad():
            preds = model(img.unsqueeze(dim=0).to(device))
            preds = F.softmax(preds,dim=1)
            PREDS_ALL.append([fname , np.round_(preds[0,1].item(),decimals=4),preds[0,1].item()>0.5,label])
            
            
PREDS_all = np.array(PREDS_ALL)

import pandas as pd
df = pd.DataFrame(PREDS_all, columns=['name','prob','predict_label','truth'])
df.to_csv('res.csv', index=False, float_format='%s %.4f %s, %d')

    
    


#%% Draw heat-map


model_hm = models.SplitBoneModel(configs.input_channels,use_heatmap = True)
model_hm.load_state_dict(torch.load('./checkpoints/resnet34_twoview-cpu.pth'))

model_hm = model_hm.to(device)
model_hm.eval()
del_mkdirs([path_png_hm])

print('\n')
print('run model for generate heat map \n')
print('\n')

for indx in tqdm(range(len(valid_dataset))):
    img,label = valid_dataset[indx]
    fname  =  valid_dataset.ids[indx]
    preds = model_hm(img.unsqueeze(dim=0).to(device))
    preds = F.softmax(preds,dim=1)
    
    img = img.unsqueeze(dim=0).to(device)
    model_hm(img).argmax(dim=1)
    
    preds[:, preds.argmax(dim=1)].backward()
    
    gradients_AP = model_hm.get_activations_gradient_AP()
    gradients_LAT = model_hm.get_activations_gradient_LAT()
    
    # pool the gradients across the channels
    pooled_gradients_AP = torch.mean(gradients_AP, dim=[0, 2, 3])
    pooled_gradients_LAT = torch.mean(gradients_LAT, dim=[0, 2, 3])
    
    #pooled_gradients_AP = torch.mean(torch.abs(gradients_AP), dim=[0, 2, 3])
    #pooled_gradients_LAT = torch.mean(torch.abs(gradients_LAT), dim=[0, 2, 3])
    
    # get the activations of the last  layer
    activations_AP  = model_hm.get_activations_AP(img).detach()
    activations_LAT  = model_hm.get_activations_LAT(img).detach()
    
    # weight the channels by corresponding gradients
    for i in range(pooled_gradients_AP.shape[0]):
        activations_AP[:, i, :, :] *= pooled_gradients_AP[i]
        activations_LAT[:, i, :, :] *= pooled_gradients_LAT[i]
    
    
    # average the channels of the activations
    heatmap_AP = torch.mean(activations_AP, dim=1).squeeze().cpu()
    heatmap_LAT = torch.mean(activations_LAT, dim=1).squeeze().cpu()
    
    # relu on top of the heatmap
    heatmap_AP = np.maximum(heatmap_AP, 0)
    heatmap_LAT = np.maximum(heatmap_LAT, 0)
    
    # normalize the heatmap
    heatmap_AP /= torch.max(heatmap_AP)
    heatmap_LAT /= torch.max(heatmap_LAT)
    
    heatmap_AP *= (heatmap_AP>0.4).float()
    heatmap_LAT *= (heatmap_LAT>0.4).float()
    # draw the heatmap
    #plt.matshow(heatmap_AP.squeeze())
    #plt.matshow(heatmap_LAT.squeeze())
    
    
    #%%
    

    fn_AP = str(Path(path_png_crp)/str(fname + '_AP.png'))
    fn_LAT = str(Path(path_png_crp)/str(fname + '_LAT.png'))
    
    img_AP = cv2.imread(fn_AP)
    img_LAT = cv2.imread(fn_LAT)
    
    
    heatmap_AP = cv2.resize(heatmap_AP.numpy(), (img_AP.shape[1], img_AP.shape[0]))
    heatmap_AP = np.uint8(255 * heatmap_AP)
    heatmap_AP = cv2.applyColorMap(heatmap_AP, cv2.COLORMAP_JET)
    superimposed_img_AP = heatmap_AP * 0.4 + img_AP
    cv2.imwrite(str(Path(path_png_hm)/str(fname + 'map_AP.jpg')), superimposed_img_AP)
    
    heatmap_LAT = cv2.resize(heatmap_LAT.numpy(), (img_LAT.shape[1], img_LAT.shape[0]))
    heatmap_LAT = np.uint8(255 * heatmap_LAT)
    heatmap_LAT = cv2.applyColorMap(heatmap_LAT, cv2.COLORMAP_JET)
    superimposed_img_LAT = heatmap_LAT * 0.4 + img_LAT
    
    
    cv2.imwrite(str(Path(path_png_hm)/str(fname + 'map_LAT.jpg')), superimposed_img_LAT)

