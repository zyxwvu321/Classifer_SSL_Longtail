#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:18:58 2019

@author: minjie
"""
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
#import torch.nn.functional as F




from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from modeling.class_loss import FocalLoss,BCElLoss
from modeling.diceloss import DiceLoss
from modeling.cross_entropy_loss import CrossEntropyLoss_labelsmooth
from tools.loggers import call_logger

from sklearn.metrics import confusion_matrix




def train(loader, net, criterion, optimizer, device, epoch,up_freq = 1, scheduler = None):    
    net.train(True)
    running_loss = 0.0
    running_corrects = 0.0
    n_label = 0.0
    optimizer.zero_grad()
    

    
    for i, data in enumerate(tqdm(loader)):
        
        images, labels = data

        #print(images.shape)
        images = images.to(device)
        labels = labels.to(device)
            
        outputs = net(images)
        
        _, preds = torch.max(outputs, 1)
        
        loss = criterion(outputs, labels)
        
        if up_freq > 1:
            loss = loss / up_freq               
            
        loss.backward()
        #clip_grad_norm_(net.parameters(), 0.5)
        if scheduler is not None:
            scheduler.step()        
            
        if (i+1)%up_freq==0:
            optimizer.step()
            optimizer.zero_grad()
            
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum((preds == labels.data).float())
        n_label += images.size(0)
        
        #print(f'{running_corrects/((i+1)*images.size(0))}')
    avg_loss = running_loss / n_label
    avg_acc  = running_corrects/ n_label     

      
    logger.info(f"Train  Epoch: {epoch}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Acc:  {avg_acc: .4f}, " )
              


def test(loader, net, criterion, device,epoch):
    net.eval()
    running_loss = 0.0
    #running_corrects = 0.0
    
   # n_class = len(loader.dataset.dict_label_inv)
    #running_corrects_label = np.zeros(n_class).astype('float')
    #running_label = np.zeros(n_class).astype('float')
    
    n_label = 0.0
    y_true = list()
    y_pred = list()    
        
    for _, data in enumerate(tqdm(loader)):
        images, labels = data
        y_true.append(np.array(labels))
        images = images.to(device)
        
        labels = labels.to(device)
        

        with torch.no_grad():
            outputs = net(images)
            _, preds = torch.max(outputs, 1)
            
            y_pred.append(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            
            #pp = F.softmax(outputs,dim = 1)
            #logger.info("Conf: {}".format(pp[:,1].cpu().numpy()))
            #logger.info(f"labels: {labels}")
            
        running_loss += loss.item()* images.size(0)
        
        #running_corrects += torch.sum(preds == labels.data).float()
        n_label += images.size(0)
        
#        for idx,label in enumerate(labels.data):
#            running_label[label.item()] +=1.0
#            if preds[idx]==label:
#                running_corrects_label[label.item()] +=1.0
#            
        
        
#        for idx, label in enumerate(labels):
#           
#            class_correct[label] += c[i].item()
#            class_total[label] += 1
    avg_loss = running_loss / n_label
        
    y_true = np.reshape(np.array(y_true),(-1)) 
    y_pred = np.reshape(np.array(y_pred),(-1)) 
    cm = confusion_matrix(y_true, y_pred)    
    
    cls_acc1 = cm.diagonal()/np.sum(cm,axis = 1)

    cls_acc2 = cm.diagonal()/np.sum(cm,axis = 0)
    
    cls_acc3  = cm.diagonal()/( np.sum(cm,axis = 0) +  np.sum(cm,axis = 1)-cm.diagonal()) 
    
    
    
    
    avg_acc = np.sum(cm.diagonal())/cm.sum()
    
    
    bal_acc1 = np.mean(cls_acc1)
    bal_acc2 = np.mean(cls_acc2)
    bal_acc3 = np.mean(cls_acc3)

   
        

    logger.info(f"Valid  Epoch: {epoch}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Acc:  {avg_acc: .4f}, " )
    
    #ave_acc_all_cls = running_corrects_label/running_label
    
    np.set_printoptions(precision=4)
    logger.info('confusion matix\n')
    logger.info('{}\n'.format(cm))

    logger.info("Num All Class: {}".format(np.sum(cm,axis = 1)))
    logger.info("Acc All Class1: {}".format(cls_acc1))
    logger.info("Acc All Class2: {}".format(cls_acc2))
    logger.info("Acc All Class3: {}".format(cls_acc3))
    
    logger.info(f"Balance Acc 1 2 3 : {bal_acc1:.4f} {bal_acc2:.4f} {bal_acc3:.4f}")
    
    
    return avg_loss,bal_acc3

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train ISIC classification')

    parser.add_argument('--datasets', default = './data/ISIC/train', type=str, help='Dataset directory path')#nargs='+'

    parser.add_argument('--validation_dataset',   default =  './data/ISIC/valid',type=str, help='Valid Dataset directory path')

    parser.add_argument('--net', default="resnet50", help="The network architecture")

    parser.add_argument('--n_class', default = 7, type=int, help='n_class')
    parser.add_argument('--seed', default = 0, type=int, help='random seed')
    
    #parser.add_argument('--input_channels', default = 1, type=int, help='channel = 1(resnet22 from mammo pretrained), 3 resnet34 from imagenet')
    

    parser.add_argument('--batch_size', default=48, type=int, help='Batch size for training')
    parser.add_argument('--num_epochs_freeze', default=50, type=int,help='the number epochs')
    parser.add_argument('--num_epochs', default=100, type=int,help='the number epochs')
    parser.add_argument('--validation_epochs', default=5, type=int,help='the number epochs')
    parser.add_argument('--sav_epochs', default=10, type=int,help='the number epochs for saving the model')

    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,help='initial learning rate')   
    parser.add_argument('--lr_freeze', default=1e-2, type=float,help='initial learning rate')     
    parser.add_argument('--backbone_mult', default=0.1, type=float,help='rt learning rate')   
    parser.add_argument('--imbalance_batchsampler', default=1,  type=int,help='imbalance sampler')   
    
    parser.add_argument('--log_file', default='log.txt',  type=str,help='logfile')   
    parser.add_argument('--loss_type', default='ce',  type=str,help='loss_func')# ce/ ce_smooth/bce/bce_smmoth/fl_bce/focal_loss

    
    
    args = parser.parse_args()
    
    if not osp.exists(args.datasets) or not osp.exists(args.validation_dataset):
        raise ValueError(f"Dataset Folder not exist")  
    
    #logger file
    logger = call_logger(args.log_file)
    
    
    log_dir = osp.join('../checkpoints_cls')
    os.makedirs(log_dir,exist_ok = True)

    set_seed(args.seed)
    
    #%%model  
        
    model = models.ISICModel(n_class = args.n_class,arch = args.net )
    model.init()
    configs = model.configs
    args.batch_size = configs.batch_size
    
    #%% augmentation dataset and dataloader
    
    train_transform = TrainAugmentation_albu(size = configs.image_size, mean = configs.image_mean, std = configs.image_std)    
    test_transform  = TestAugmentation_albu(size = configs.image_size, mean = configs.image_mean, std = configs.image_std)


    train_dataset = CustomDataset(root = args.datasets, transform = train_transform,label_name =configs.dict_label)
    valid_dataset = CustomDataset(root = args.validation_dataset, transform = test_transform,label_name =configs.dict_label)



    if args.imbalance_batchsampler==1:
        train_loader = DataLoader(train_dataset, batch_size = 2*args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset),num_workers=args.num_workers)    
    else:
        train_loader = DataLoader(train_dataset, batch_size = 2*args.batch_size, num_workers=args.num_workers,shuffle = True)    

    
    valid_loader = DataLoader(valid_dataset, batch_size = 2*args.batch_size, num_workers=args.num_workers, shuffle=False)


    
    
    #%% net to DEVICE
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    

    #%% loss 
    if args.loss_type =='ce':
        if args.imbalance_batchsampler:
        
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            
            lb_freq = np.array(train_dataset.lb_freq).astype('float')
            lb_freq = lb_freq/np.sum(lb_freq)
            lb_freq = (1.0/len(lb_freq))/lb_freq
            criterion = nn.CrossEntropyLoss(weight = torch.tensor(lb_freq).float()).to(device)
    elif args.loss_type =='ce_smooth':
        criterion = CrossEntropyLoss_labelsmooth(num_classes = args.n_class).to(device)
        
    elif args.loss_type =='bce':
        if args.imbalance_batchsampler:
            criterion = BCElLoss(num_classes = args.n_class,mult_val = 10.0).to(device)    
        else:
            lb_freq = np.array(train_dataset.lb_freq).astype('float')
            lb_freq = lb_freq/np.sum(lb_freq)
            lb_freq = (1.0/len(lb_freq))/lb_freq
            criterion = BCElLoss(num_classes = args.n_class,weight = torch.tensor(lb_freq).float().to(device),mult_val = 10.0).to(device)
        
    elif args.loss_type =='fl_bce':
        criterion = FocalLoss(num_classes = args.n_class).to(device)
    elif args.loss_type == 'dice_bce':
        if args.imbalance_batchsampler:
            criterion = DiceLoss().to(device)
        else:
            criterion = DiceLoss(weight= torch.tensor(lb_freq).float().to(device)).to(device)
        
    else:
        raise ValueError(f"Unknown crit")  


        
    #%% train start
    best_acc = 0.0
    min_loss = 100000
        
    if args.num_epochs_freeze>0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        optim_freeze = SGD(model.head.parameters(),  lr=args.lr_freeze, momentum=0.9,weight_decay=5e-04)

        
        scheduler_freeze = CyclicLR(optim_freeze, base_lr=0.1*args.lr_freeze, max_lr=args.lr_freeze, 
                                                     step_size_up=len(train_loader)*args.num_epochs_freeze//2, step_size_down=None, 
                                                     mode='triangular',  scale_mode='cycle', 
                                                     cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        

        
        for epoch in range(args.num_epochs_freeze):

            train(train_loader, model, criterion, optim_freeze, device,epoch=epoch, up_freq = 1, scheduler = scheduler_freeze)
            
            if epoch % args.validation_epochs == 0 or  epoch ==args.num_epochs_freeze - 1:
                epoch_loss,epoch_acc = test(valid_loader, model, criterion, device,epoch=epoch)
            
            if epoch % args.sav_epochs == 0 or epoch == args.num_epochs_freeze - 1:
                
                model_path = osp.join(log_dir, f"{args.net}-Epoch-{epoch}-freeze-loss-{epoch_loss:.4f}.pth")
                torch.save(model.state_dict(),model_path)

            # deep copy the model
            if epoch_loss < min_loss:
                best_acc = epoch_acc
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                logger.info('Best val Acc and loss: {:.4f}, {:.4f}'.format(best_acc,min_loss))

    # Unfreeze
    if args.imbalance_batchsampler==1:
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset),num_workers=args.num_workers)    
    else:
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=args.num_workers,shuffle = True)    
    
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    
    
    for param in model.backbone.parameters():
        param.requires_grad = True

    optim_unfreeze = SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr*args.backbone_mult},
        {'params': model.head.parameters(), 'lr': args.lr},
    ],momentum=0.9,weight_decay=5e-04)
    scheduler_unfreeze = CyclicLR(optim_unfreeze,
                                  base_lr=[args.backbone_mult*0.1*args.lr, 0.1*args.lr], max_lr=[args.backbone_mult*args.lr, args.lr],
                                  step_size_up=int(args.num_epochs*len(train_loader)/2),
                                  step_size_down=None, 
                                  mode='triangular',  scale_mode='cycle', 
                                  cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)    
    

    for epoch in range(args.num_epochs):

        train(train_loader, model, criterion, optim_unfreeze, device,epoch=epoch, up_freq = 1, scheduler = scheduler_unfreeze)

        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            epoch_loss,epoch_acc = test(valid_loader, model, criterion, device,epoch=epoch)
        
        if epoch % args.sav_epochs == 0 or epoch == args.num_epochs - 1:
            
            model_path = osp.join(log_dir, f"{args.net}-Epoch-{epoch}-loss-{epoch_loss:.4f}.pth")

            torch.save(model.state_dict(),model_path)

    
    
        # deep copy the model
        if epoch_loss < min_loss:
            best_acc = epoch_acc
            min_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            logger.info('Best val Acc and loss: {:.4f}, {:.4f}'.format(best_acc,min_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(),osp.join(log_dir, f"{args.net}-best.pth"))
