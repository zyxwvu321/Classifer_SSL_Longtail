#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:18:58 2019
Train with global, local, and meta infos
@author: minjie
"""
#%run train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_meta --out_dir ../checkpoints/resnet50_meta

import argparse
import os
from config import cfg
import numpy as np
import os.path as osp
from tqdm import tqdm
from utils.utils import set_seed




from modeling.models import ISICModel_meta,ISICModel_twoview,ISICModel_singleview,ISICModel_singleview_meta


from dataset.custom_dataset import CustomDataset_withmeta
from dataset.sampler import ImbalancedDatasetSampler,resample_idx_with_meta

import torch
import torch.nn as nn
#import cv2
import copy
import pandas as pd
#import albumentations as A
#from albumentations.pytorch import ToTensor as ToTensor_albu

#from matplotlib import pyplot as plt
from dataset.transform.data_preprocessing import  TrainAugmentation_albu,TestAugmentation_albu

#from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.utils.data import DataLoader
#import torch.nn.functional as F


from utils.utils import AvgerageMeter

from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from modeling.class_loss import FocalLoss,BCElLoss
from modeling.diceloss import DiceLoss
from modeling.cross_entropy_loss import CrossEntropyLoss_labelsmooth
from tools.loggers import call_logger

from sklearn.metrics import confusion_matrix




    
#%%



    
    
def test_tta(ds, net,  device,epoch = -1,n_tta = 10,n_class = 7):
    net.eval()

    
    n_case = 0.0
    y_true = list()
    y_pred = list()  


    PREDS_ALL = []
    for idx in tqdm(range(len(ds))):
        
        #print(images.shape)
        
        
        with torch.no_grad():
            pred_sum = torch.zeros((n_class),dtype = torch.float32)
            
            for n_t in range(n_tta):
                images, images_roi, labels,meta_infos = ds[idx]
                
                if n_t==0:
                    y_true.append(labels.item())
        
                images = images.to(device)
                images_roi = images_roi.to(device) 
                meta_infos = meta_infos.to(device)                
                labels = labels.to(device)
                
 
                if net.mode =='metatwoview':
                    outputs = net(images[None,...],images_roi[None,...],meta_infos[None,...])
                elif net.mode =='twoview':
                    outputs = net(images[None,...],images_roi[None,...])
                elif net.mode =='singleview':
                    outputs = net(images[None,...])
                elif net.mode =='metasingleview':
                    outputs = net(images[None,...],meta_infos[None,...])
                
    
                
                probs_0 = F.softmax(outputs,dim=1)[0].cpu()
                pred_sum = pred_sum + probs_0
                    
                
            pred_sum = pred_sum/n_tta
            n_case += 1
            probs = np.round_(pred_sum.numpy(),decimals=4)
            
            preds = torch.argmax(pred_sum).item()
            
            y_pred.append(preds)
            
            
            
            PREDS_ALL.append([*probs,preds])
            
    PREDS_ALL = np.array(PREDS_ALL)



    
#    logger.info(f"Valid  Epoch: {epoch}, " +
#                f"Average Loss: {total_loss.avg:.4f}, " +
#                f"Average Acc:  {avg_acc}, " )

    
    return PREDS_ALL

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train ISIC classification')

    parser.add_argument('--datasets', default = '../data/ISIC18/task3/ISIC2018_Task3_Test_Input_coloradj', type=str, help='Dataset directory path')
    parser.add_argument('--info_csv', default = './dat/ISIC18_info_test.csv', type=str, help='Dataset info file, include meta')
    
    #this is skip nut load for consistency
    parser.add_argument('--fn_map', default = './dat/fn_maps_ISIC18.pth', type=str, help='Dataset mapping index, map same id for same meta')
    
    
    parser.add_argument('--net', default="resnet50", help="The network architecture")

    parser.add_argument('--n_class', default = 7, type=int, help='n_class')

    
    parser.add_argument('--seed', default = 0, type=int, help='random seed')
    
    #parser.add_argument('--input_channels', default = 1, type=int, help='channel = 1(resnet22 from mammo pretrained), 3 resnet34 from imagenet')
    
    parser.add_argument('--batch_size', default=48, type=int, help='Batch size for training')
  
    parser.add_argument('--K_fold', default=5, type=int,help='K-folder validation')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    
    parser.add_argument('--loss_type', default='ce',  type=str,help='loss_func')# ce/ ce_smooth/bce/bce_smmoth/fl_bce/focal_loss
    
    parser.add_argument('--log_file', default='log.txt',  type=str,help='logfile')   
    parser.add_argument('--out_dir', default='../checkpoints_cls',  type=str,help='model out folder')   


    parser.add_argument('--tta', default=0, type=int,help='if tta')




    
    
    args = parser.parse_args()
    
    if not osp.exists(args.datasets):
        raise ValueError(f"Dataset Folder not exist")  
    
    #logger file
    os.makedirs(args.out_dir,exist_ok = True)
    logger = call_logger(osp.join(args.out_dir, args.log_file))
    
    

    

    set_seed(args.seed)
    
    #%%model  
    
    if 'metatwoview' in args.net:
        model = ISICModel_meta(n_class = args.n_class,arch = args.net )
    
    elif 'metasingleview' in args.net:
        model = ISICModel_singleview_meta(n_class = args.n_class,arch = args.net )  
        
    elif 'twoview' in args.net:
        model = ISICModel_twoview(n_class = args.n_class,arch = args.net )
    elif 'singleview' in args.net:
        model = ISICModel_singleview(n_class = args.n_class,arch = args.net )

    else:
        raise ValueError('unknown net')
    configs = model.configs
    model.init()
    
    args.batch_size = configs.batch_size
    
    #%% augmentation dataset and dataloader
    
#    train_transform = TrainAugmentation_albu(sz_in_hw = configs.sz_in_hw, sz_out_hw = configs.sz_out_hw, mean = configs.image_mean, std = configs.image_std, 
#                                             crp_scale= configs.crp_scale,crp_ratio = configs.crp_ratio)    
#    
#    
#    train_transform_lc = TrainAugmentation_albu(sz_in_hw = configs.sz_in_hw_lc, sz_out_hw = configs.sz_out_hw_lc, mean = configs.image_mean, std = configs.image_std, 
#                                             crp_scale= configs.crp_scale_lc,crp_ratio = configs.crp_ratio_lc)    
    
    
    train_transform = TrainAugmentation_albu(sz_in_hw = configs.sz_in_hw, sz_out_hw = configs.sz_out_hw, mean = configs.image_mean, std = configs.image_std, 
                                             minmax_h= configs.minmax_h,w2h_ratio = configs.w2h_ratio)    
    
    
    train_transform_lc = TrainAugmentation_albu(sz_in_hw = configs.sz_in_hw_lc, sz_out_hw = configs.sz_out_hw_lc, mean = configs.image_mean, std = configs.image_std, 
                                             minmax_h= configs.minmax_h_lc,w2h_ratio = configs.w2h_ratio_lc)    

    test_transform  = TestAugmentation_albu(size = configs.sz_out_hw, mean = configs.image_mean, std = configs.image_std)        
    test_transform_lc   = TestAugmentation_albu(size = configs.sz_out_hw_lc, mean = configs.image_mean, std = configs.image_std)


    dataseto = CustomDataset_withmeta(root = args.datasets, info_csv = args.info_csv,fn_map =args.fn_map, transform = train_transform,transform_localbbox = train_transform_lc ,is_test = True)
    
    
    
    
    #%% net to DEVICE
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)


    
    
    pred_out_all = []
    for nf in range(args.K_fold):
        logger.info(f'start fold {nf}')

        
        train_ds = dataseto


        
        
        if args.K_fold ==1:
            best_model_fn = osp.join(args.out_dir, f"{args.net}-best.pth")
        else:
            best_model_fn = osp.join(args.out_dir, f"{args.net}-Fold-{nf}-best.pth")
            
            

        model.load_state_dict(torch.load(best_model_fn))
        if args.tta==1:
            
            pred_out = test_tta(dataseto, model,  device,epoch=-1,n_tta = 10,n_class = args.n_class)
            
        else:
            dataseto.transform= test_transform
            pred_out = test_tta(dataseto, model,  device,epoch=-1,n_tta = 1,n_class= args.n_class)
        
        
        fns_kfd = np.array(dataseto.fname)
        pred_out = np.hstack((fns_kfd[:,None],np.array(pred_out)))
        pred_out_all.append(pred_out)        
        
       

    pred_out_all = np.vstack(pred_out_all)    
    df = pd.DataFrame(data = pred_out_all[:,1:-1].astype('float32'),index =pred_out_all[:,0], columns = [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC'])
    for col in ['MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']:
        df[col] = df[col].apply(lambda x: format(x,'.4f'))

        
    eval_path = osp.join(args.out_dir, f"eval_{args.net}-Loss-{args.loss_type}-tta-{args.tta}-test.csv")
    df.to_csv(eval_path, index_label = 'fn')
    