#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:18:58 2019

@author: minjie

Train bbox of bone x-ray images
"""

# %run test_locate_box.py --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input --outdir ../data/ISIC18/task3/ISIC_2018_Task3_Training_locate --sav_model ../data/ISIC18/task1/models/resnet50_c3_locate-Epoch-119-loss-0.1520.pth --net resnet50_c3_locate


import argparse
import os



import os.path as osp
from tqdm import tqdm
from utils.utils import set_seed

import modeling.models as models
from modeling.location_box_loss import Location_Box_Loss




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
from utils.utils import AvgerageMeter,extend_box_square


from pathlib import Path
from modeling.location_box_loss import model_box_to_xy_ori_l1,norm_box_to_abs

from preproc.letterbox_resize import letterbox_image,letterbox_xywh,inv_letterbox_bbox
    
import pandas as pd    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test skin box localization')

    #parser.add_argument('--datasets', default = '/home/minjie/dataset/ISIC/ISIC_2019_Training_Input', type=str, help='Dataset directory path')#nargs='+'
    #parser.add_argument('--outdir', default = '/home/minjie/dataset/ISIC/ISIC_2019_Training_locate', type=str, help='output directory path')#nargs='+'
    parser.add_argument('--datasets', default = '../data/ISIC18/ISIC2018_Task3_Training_Input', type=str, help='Dataset directory path')#nargs='+'
    parser.add_argument('--outdir', default = '../data/ISIC18/ISIC_2018_Task3_Training_locate', type=str, help='output directory path')#nargs='+'
    
    parser.add_argument('--sav_model', default = '../checkpoints/resnet34_c3_locate-Epoch-119-loss-0.1531.pth', type=str, help='Dataset directory path')#nargs='+'
    
    #parser.add_argument('--fd_box', default = '/home/minjie/dataset/ISIC/ISIC_2019_Training_Input', type=str, help='Box directory path')#nargs='+'
    
   
    parser.add_argument('--net', default="resnet34_c3_locate", help="The network architecture")


    
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')

    
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    
    parser.add_argument('--out_csv', default='./dat/bbox_isic18.csv', type=str, help='outcsv')
  
    
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



    model.eval()



    flist = sorted(list(Path( args.datasets).glob('*.jpg')))
    out_wh = (256,256)
    
    box_list = list()
    
    for  fn in tqdm(flist):
        img = cv2.imread(str(fn))
        box_dim = (out_wh[1], out_wh[0])
        
        img_t = letterbox_image(img, box_dim)
        boxes = np.array([[0.0,0.0,1.0,1.0]]).astype('float32')
        images, boxes = test_transform(img_t,boxes)
        
        
        images = images.to(device)[None,...]
        

        

        with torch.no_grad():
            outputs = model(images)
            box_pred = model_box_to_xy_ori_l1(outputs.clone())    

            box_pred_abs = norm_box_to_abs(box_pred[0],img_t.shape).astype('int')
            
            box_pred_ext = extend_box_square(box_pred_abs,sz = out_wh,ex_p = 0.25)

            box_inv  = inv_letterbox_bbox(box_pred_abs[None,...].astype('float32'), box_dim, img.shape[:2])
            box_inv = np.round(box_inv).astype('int')[0]
            box_list.append(np.hstack([fn.stem,box_inv]))



            cv2.rectangle(img_t, (box_pred_abs[0], box_pred_abs[1]), (box_pred_abs[2], box_pred_abs[3]), (0, 0, 255), 2)
            cv2.rectangle(img_t, (box_pred_ext[0], box_pred_ext[1]), (box_pred_ext[2], box_pred_ext[3]), (255, 0, 0), 2)
            cv2.imwrite(str(Path(args.outdir)/fn.name),img_t)



            #cv2.rectangle(img, (box_inv[0], box_inv[1]), (box_inv[2], box_inv[3]), (0, 0, 255), 4)
            #cv2.imwrite(str(Path(args.outdir)/(fn.stem + 'o.png')),img)

df = pd.DataFrame(data = np.array(box_list),columns=['fname','x1','y1','x2','y2'])
df.to_csv(args.out_csv,index= False)
