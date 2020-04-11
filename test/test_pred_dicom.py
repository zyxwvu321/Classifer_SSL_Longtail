#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:29:38 2019

@author: minjie
"""
import cv2
import torch
import modeling.models as models
from pathlib import Path

from fastai import *
from fastai.vision import *
import pydicom
import matplotlib.pyplot as plt
from utils.utils import del_mkdirs
from tqdm import tqdm
from transforms.data_preprocessing import  TestAugmentation_albu
#model = models.LocateBoneModel()
from transforms.data_preprocessing import  TestAugmentation_albu



#%% convert dicom to png
path_dicom = './data/ori/ori_dicom'


fd_list = [str(Path(path_dicom)/'pos'),str(Path(path_dicom)/'neg')]


fd_list_png = [fd.replace('dicom','png') for fd in fd_list]
del_mkdirs(fd_list_png)

#del_mkdirs([str(Path(fd_list_png[0]).parents[0])])

for fds in fd_list:
    flist = [str(fn) for fn in sorted(Path(fds).glob('*.dcm'))]
    
    for fn in flist:
        
        with pydicom.dcmread(fn) as dc:
            img_dicom = dc.pixel_array
            print(img_dicom.shape, img_dicom.shape[0]/img_dicom.shape[1])
            
            
            fname_png = fn.replace('.dcm','.png').replace('dicom','png')
  
            cv2.imwrite(fname_png, (img_dicom/16384.0*255).astype('uint8'))
            #x = pil2tensor(img_dicom,np.float32).div_(16384)
            #Image_AP = Image(x)
            

#%% crop image(image localiazation)   
    
#fastai_model='./jupyter_fastai/resnet18_location.pth'

#defaults.device = torch.device('cpu')

learn = load_learner('/home/minjie/fastai_test/test_bone/checkpoints/','locate_fastai.pkl')
    
fd_list_png = [fd.replace('ori_','AI_') for fd in fd_list_png]
fd_list = [str(Path(path_dicom)/'pos'),str(Path(path_dicom)/'neg')]
fd_list_pnganno = [fd.replace('_png','_pnganno') for fd in fd_list_png]
fd_list_pngcrp = [fd.replace('_png','_pngcrp') for fd in fd_list_png]
del_mkdirs(fd_list_pnganno)
del_mkdirs([str(Path(fd_list_pngcrp[0]).parents[0])])






#model = models.LocateBoneModel()
#fastai_model='./jupyter_fastai/resnet18_location.pth'
#model.init_fastaimodel(fastai_model)
#dicts_fastai = torch.load(fastai_model)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
#model = model.to(device)
#model.eval()

for fds in fd_list_png:
    flist = [str(fn) for fn in sorted(Path(fds).glob('*.png'))]
    
    for fn in tqdm(flist):
        img = open_image(fn)
        pred_class,pred_idx,outputs = learn.predict(img)
        print(outputs)
        
        
        
#        img = cv2.imread(fn)
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#        test_transform = TestAugmentation_albu([256,320], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#
#
#        img_torch =test_transform(img)
#
#        with torch.no_grad():
#            outputs = model(img_torch.unsqueeze(dim=0).to(device)).cpu()[0]

        

        fn_pnganno = fn.replace('_png','_pnganno')
        
        box = (outputs.numpy() + 1.0)/2.0
        
        image = cv2.imread(fn)    
        hh,ww,_ = image.shape
        box[0] *= hh
        box[1] *= ww
        box[2] *= hh
        box[3] *= ww
        box = box.astype('int')
        
        img_crp = image[box[0]:box[2]-1,box[1]:box[3]-1,:]
        img_crp = cv2.resize(img_crp,(512,512))
        cv2.imwrite(fn.replace('_png','_pngcrp').replace('0.png','_AP.png').replace('1.png','_LAT.png').replace('pos/','Pos').replace('neg/','Neg'),img_crp)
        
        cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (255, 255, 0), 4)
        cv2.imwrite(fn_pnganno,image)
        






#%%
from configs import resnet34_c3_v2
from datasets.custom_dataset import  CustomDataset
import torch.nn.functional as F
import numpy as np
import modeling.models as models
configs = resnet34_c3_v2


     
test_transform = TestAugmentation_albu(configs.image_size, configs.image_mean, configs.image_std)        
        
test_dataset =  CustomDataset(root = './data/ori/AI_pngcrp', transform = test_transform,input_channels = configs.input_channels)

model = models.SplitBoneModel(configs.input_channels)
model.load_state_dict(torch.load('./checkpoints/tmp1.pth'))
model.to(device)
model.eval()


PREDS_ALL = []


for idx,ds in enumerate(tqdm(test_dataset)):
    img,label = ds
    fname  =  test_dataset.ids[idx]
    with torch.no_grad():
        preds = model(img.unsqueeze(dim=0).to(device))
        preds = F.softmax(preds,dim=1)
        

        
        PREDS_ALL.append([fname , np.round_(preds[0,1].item(),decimals=4),preds[0,1].item()>0.5,label])
PREDS_all = np.array(PREDS_ALL)

import pandas as pd
df = pd.DataFrame(PREDS_all, columns=['name','prob','predict_label','truth'])
df.to_csv('res.csv', index=False, float_format='%s %.4f %s, %d')

