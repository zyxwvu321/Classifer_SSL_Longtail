#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:18:58 2019

@author: minjie

Train bbox of bone x-ray images
"""

# %run test_locate_box_ds.py --datasets ../data/ISIC18/task1/ISIC-2018_Img --fd_box ../data/ISIC18/task1/ISIC-2018_Box --net resnet50_c3_locate --sav_model ../data/ISIC18/task1/models/resnet50_c3_locate-Epoch-119-loss-0.1520.pth --net resnet50_c3_locate  --outdir ../data/ISIC18/task1/ISIC-2018_out_box

import argparse
import os



import os.path as osp
from tqdm import tqdm
from utils.utils import set_seed

import modeling.models as models





import torch
import torch.nn as nn
import cv2
import copy
import numpy as np
#import albumentations as A
#from albumentations.pytorch import ToTensor as ToTensor_albu

from matplotlib import pyplot as plt
from transforms.data_preprocessing import  TrainAugmentation_bbox_albu,TestAugmentation_bbox_albu



from dataset.custom_dataset import  CustomDataset_bbox
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.optim import SGD
from optim import build_optimizer
from torch.optim.lr_scheduler import MultiStepLR,CyclicLR
from torch.nn.utils import clip_grad_norm_

from tools.loggers import call_logger
from utils.utils import AvgerageMeter


from pathlib import Path
from modeling.location_box_loss import model_box_to_xy_ori_l1,norm_box_to_abs,iou_of

from preproc.letterbox_resize import letterbox_image,letterbox_xywh
    
import pandas as pd
from modeling.location_box_loss import Location_Box_Loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test skin box localization')

    parser.add_argument('--datasets', default = '../data/ISIC07_task1_train/ISIC-2017_Img', type=str, help='Dataset directory path')#nargs='+'
    parser.add_argument('--fd_box', default = '../data/ISIC07_task1_train/ISIC-2017_Box', type=str, help='Box directory path')#nargs='+'
    
    parser.add_argument('--outdir', default = '../data/ISIC07_task1_train/ISIC-2017_Training_Data_box', type=str, help='output directory path')#nargs='+'
    
    
    parser.add_argument('--sav_model', default = '../checkpoints/resnet34_c3_locate-Epoch-119-loss-0.1531.pth', type=str, help='Dataset directory path')#nargs='+'
    
    
    
   
    parser.add_argument('--net', default="resnet34_c3_locate", help="The network architecture")


    
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')

    
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    

  
    
    args = parser.parse_args()
    
    if not osp.exists(args.datasets) :
        raise ValueError(f"Dataset Folder not exist")  

    os.makedirs(args.outdir,exist_ok = True)

    
    #%%model  
    if args.net == 'resnet34_c3_locate':
        from configs import resnet34_c3_locate
        configs = resnet34_c3_locate
        model = models.LocateSkinImgModel_CustomSquare(backbone=configs.model_type)
        
    elif args.net == 'resnet50_c3_locate':
        from configs import resnet50_c3_locate
        configs = resnet50_c3_locate
        model = models.LocateSkinImgModel_CustomSquare(backbone=configs.model_type)
    else:
        raise ValueError(f"Model not exist")
    
    
    model.load_state_dict(torch.load(args.sav_model))

    #%% augmentation dataset and dataloader
    
   
    test_transform = TestAugmentation_bbox_albu(configs.image_size, configs.image_mean, configs.image_std)

    
    
    
    
 
    


    #%% net to DEVICE
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    
    
    
    #%% loss 

    criterion = Location_Box_Loss(prior_w = configs.prior_w, prior_cx=configs.prior_cx, prior_cy=configs.prior_cy,size_variance = configs.size_variance,center_variance = configs.center_variance)
    criterion = criterion.to(device)

    model.eval()

    test_dataset =  CustomDataset_bbox(root = args.datasets, box_root = args.fd_box,transform = test_transform,cfg = configs)
    test_loader = DataLoader(test_dataset, batch_size = 1, num_workers=0, shuffle=False)

    flist = sorted(list(Path( args.datasets).glob('*.jpg')))
    out_wh = (256,256)
    
    data_list = list()
    
    for  idx, data in enumerate(tqdm(test_loader)):
        fn = flist[idx]
        img = cv2.imread(str(fn))
        
        
        box_dim = (out_wh[1], out_wh[0])
        
        img_t = letterbox_image(img, box_dim)
        boxes0 = np.array([[0.0,0.0,1.0,1.0]]).astype('float32')
        
        images, boxes = data
        #img_t, boxes = test_transform(img_t,boxes0)
        
        
        images = images.to(device)
        
        boxes = boxes.to(device)
        

        with torch.no_grad():
            outputs = model(images)
            loss,loss_xx,loss_yy,loss_ww,loss_hh = criterion(boxes,outputs)
            
            box_pred = model_box_to_xy_ori_l1(outputs.clone())    
            

            iou = iou_of(torch.from_numpy(box_pred), boxes.cpu()).item()
            
            
            dat_line =  np.hstack((fn.stem,boxes[0].cpu().numpy(),box_pred[0],iou))
            data_list.append(dat_line)
            
            box_pred_abs = norm_box_to_abs(box_pred[0],img_t.shape).astype('int')


    

            cv2.rectangle(img_t, (box_pred_abs[0], box_pred_abs[1]), (box_pred_abs[2], box_pred_abs[3]), (0, 0, 255), 4)
            cv2.imwrite(str(Path(args.outdir)/fn.name),img_t)
#

data_list = np.array(data_list)

data_pos = data_list[:,1:-1].astype('float32')
dx = np.abs((data_pos[:,0]+ data_pos[:,2])/2.0 - (data_pos[:,4]+ data_pos[:,6])/2.0)[:,None]
dy = np.abs((data_pos[:,1]+ data_pos[:,3])/2.0 - (data_pos[:,5]+ data_pos[:,7])/2.0)[:,None]
dw = np.abs((data_pos[:,2]- data_pos[:,0]) - (data_pos[:,6]- data_pos[:,4]))[:,None]
dh = np.abs((data_pos[:,3]- data_pos[:,1]) - (data_pos[:,7]- data_pos[:,5]))[:,None]

data_list0 = np.hstack((data_list,dx,dy,dw,dh))

df = pd.DataFrame(data = data_list0,columns=['fname','x1t','y1t','x2t','x2t','x1','y1','x2','x2','iou','dx','dy','dw','dh'])



df.to_csv('./dat/bbox18test.csv',index = False)