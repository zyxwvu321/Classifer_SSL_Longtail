#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:54:08 2019

@author: minjie
"""

#%% test crop data
import argparse
import os

import numpy as np
import os.path as osp
from tqdm import tqdm
from utils.utils import set_seed

import modeling.models as models

from dataset.custom_dataset import CustomDataset
from dataset.sampler import ImbalancedDatasetSampler

import torch
import torch.nn as nn
#import cv2
import copy
#import albumentations as A
#from albumentations.pytorch import ToTensor as ToTensor_albu

#from matplotlib import pyplot as plt
from transforms.data_preprocessing import  TrainAugmentation_albu,TestAugmentation_albu

#from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
import torch.nn.functional as F




from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from modeling.class_loss import FocalLoss
from modeling.cross_entropy_loss import CrossEntropyLoss_labelsmooth
from tools.loggers import call_logger

import pandas as pd

from sklearn.metrics import confusion_matrix




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preidit ISIC classification')

    parser.add_argument('--datasets', default = './data/ISIC/valid', type=str, help='Dataset directory path')#nargs='+'
    
    parser.add_argument('--is_valid', default = 1, type=int, help='valid(dataset) test(a folder)')#nargs='+'

    parser.add_argument('--net', default="resnet50", help="The network architecture")

    parser.add_argument('--n_class', default = 7, type=int, help='n_class')
    
    
    parser.add_argument('--model_file', default = './checkpoints/resnet50-best.pth', type=str, help='model file')
    
    parser.add_argument('--batch_size', default=48, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    
    
    parser.add_argument('--is_tta', default=False, type=bool, help='use tta')
    parser.add_argument('--n_tta', default=10, type=int, help='Num of tta')
    
    parser.add_argument('--log_file', default='test.txt',  type=str,help='logfile')   
    parser.add_argument('--loss_type', default='ce',  type=str,help='loss_func')# ce/ ce_smooth/bce/bce_smmoth/focal_loss_bce/focal_loss
    
    parser.add_argument('--fn_csv', default='out.csv',  type=str,help='out csv file')

    
    args = parser.parse_args()
    
    if args.is_valid and not osp.exists(args.datasets) :
        raise ValueError(f"Dataset Folder not exist")  
    
    
    #logger file
    logger = call_logger(args.log_file)
    
    




    
    #%%model  
        
    model = models.ISICModel(n_class = args.n_class,arch = args.net )
    model.init()
    model.load_state_dict(torch.load(args.model_file))
    configs = model.configs

    
    #%% augmentation dataset and dataloader
    
    if args.is_tta:
        test_transform = TrainAugmentation_albu(size = configs.image_size, mean = configs.image_mean, std = configs.image_std)    
    else:
        test_transform  = TestAugmentation_albu(size = configs.image_size, mean = configs.image_mean, std = configs.image_std)
    
    if args.is_valid:
        dataset = CustomDataset(root = args.datasets, transform = test_transform,label_name =configs.dict_label)
    else:
        dataset = CustomDataset(root = args.datasets, transform = test_transform,label_name =configs.dict_label, is_test = True)
    

        
    test_loader = DataLoader(dataset, batch_size = 2*args.batch_size, num_workers=args.num_workers,shuffle = False)    


    
    #%% net to DEVICE
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    

    #%% loss 
    
    
    
    model.eval()
    PREDS_ALL = []
    fname_all = []
    y_true = []
    
    logger.info('run model for classification \n')
    

    for idx,ds in enumerate(tqdm(dataset)):
        img,label = ds
        y_true.append(label)
        fname  =  dataset.ids[idx]
        
        if args.is_tta:
            pred_sum = np.zeros(args.n_class, dtype = 'float')
            for n_t in range(args.n_tta):
                with torch.no_grad():
                    preds = model(img.unsqueeze(dim=0).to(device))
                    preds = F.softmax(preds,dim=1)
                    pred_sum += preds.cpu().numpy()[0]
            pred_ave = pred_sum /args.n_tta
            PREDS_ALL.append(pred_ave)
            fname_all.append(fname)        
        else:
            with torch.no_grad():
                preds = model(img.unsqueeze(dim=0).to(device))
                preds = F.softmax(preds,dim=1).cpu().numpy()[0]
                PREDS_ALL.append(preds)
                fname_all.append(fname)        
                
                
    PREDS_all = np.array(PREDS_ALL)
    
    
    df = pd.DataFrame(PREDS_all,columns=list(configs.dict_label.values()))
    df.insert(0,'image',  fname_all)
    
    df.to_csv(args.fn_csv, index=False)
    
    if  args.is_valid ==1:
        # generate balance accuracy and confusion matrix
        np.set_printoptions(precision= 4,suppress=False)
        cm = confusion_matrix(np.array(y_true),  np.argmax(PREDS_all,axis = 1))    
        norm_cm = cm.diagonal()/np.sum(cm,axis = 1)
        
        
        norm_cm1 = cm.diagonal()/np.sum(cm,axis = 0)
        
        
        acc = np.sum(cm.diagonal())/cm.sum()
        acc_bal = np.mean(norm_cm)
        acc_bal1 = np.mean(norm_cm1)
        
        
        norm_cm2 =  cm.diagonal()/( np.sum(cm,axis = 0) +  np.sum(cm,axis = 1)-cm.diagonal())
        acc_bal2 = np.mean(norm_cm2)
        
        logger.info('confusion matix\n')
        logger.info('{}\n'.format(cm))
        logger.info('norm class acc  {}\n'.format(norm_cm))
        logger.info('norm class acc1  {}\n'.format(norm_cm1))
        logger.info('norm class acc2  {}\n'.format(norm_cm2))
        
        logger.info(f'acc acc_bal acc_bal1 acc_bal2 {acc :.4f} {acc_bal :.4f} {acc_bal1 :.4f} {acc_bal2 :.4f}\n')
        
    