#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:18:58 2019

@author: minjie

Train bbox of bone x-ray images
"""


#%run train_locate_box.py  --datasets ../data/ISIC18/task1/ISIC-2018_Img --fd_box ../data/ISIC18/task1/ISIC-2018_Box --net resnet50_c3_locate




import argparse
import os



import os.path as osp
from tqdm import tqdm
from utils.utils import set_seed

import modeling.models as models
from modeling.location_box_loss import Location_Box_Loss,Location_Box_CenterNet


import torch
import torch.nn as nn
import cv2
import copy
import numpy as np
#import albumentations as A
#from albumentations.pytorch import ToTensor as ToTensor_albu

from matplotlib import pyplot as plt
from transforms.data_preprocessing import  TrainAugmentation_bbox_albu,TestAugmentation_bbox_albu



from dataset.custom_dataset import  CustomDataset_bbox,CustomDataset_bbox_centernet
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.optim import SGD
from optim import build_optimizer
from torch.optim.lr_scheduler import MultiStepLR,CyclicLR
from torch.nn.utils import clip_grad_norm_

from tools.loggers import call_logger
from utils.utils import AvgerageMeter

log_file = 'log_locate1.txt'
logger = call_logger(log_file)


def train(loader, net, criterion, optimizer, device, epoch,scheduler = None,net_type = 'resnet50_c3_locate'):    
    net.train(True)
    optimizer.zero_grad()
    
    
    
    if net_type == 'resnet34_c3_locate' or net_type == 'resnet50_c3_locate':

        total_loss, x_loss,y_loss,w_loss,h_loss = AvgerageMeter(),AvgerageMeter(),AvgerageMeter(),AvgerageMeter(),AvgerageMeter()    
    
    elif net_type == 'resnet34_c3_locate_centernet':
        total_loss, score_loss,xy_loss,wh_loss = AvgerageMeter(),AvgerageMeter(),AvgerageMeter(),AvgerageMeter()
    
    
    
    
    
    for i, data in enumerate(loader):
        
        images, boxes = data
        #print(images.shape)
        images = images.to(device)
        
        if type(boxes)==dict:
            for key,value in boxes.items():
                boxes[key] = value.to(device)
        else:
            boxes = boxes.to(device)
            
        outputs = net(images)
        
        #_, preds = torch.max(outputs, 1)
        if net_type == 'resnet34_c3_locate' or net_type == 'resnet50_c3_locate':
            loss,loss_xx,loss_yy,loss_ww,loss_hh = criterion(boxes,outputs)
        
            total_loss.update(loss.item(),images.size(0))
            x_loss.update(loss_xx.item(),images.size(0))
            y_loss.update(loss_yy.item(),images.size(0))
            w_loss.update(loss_ww.item(),images.size(0))
            h_loss.update(loss_hh.item(),images.size(0))
        
        elif net_type == 'resnet34_c3_locate_centernet':
            loss,loss_score,loss_xy,loss_wh = criterion(boxes,outputs)
        
            total_loss.update(loss.item(),images.size(0))
            score_loss.update(loss_score.item(),images.size(0))
            xy_loss.update(loss_xy.item(),images.size(0))
            wh_loss.update(loss_wh.item(),images.size(0))

            
        loss.backward()
        #clip_grad_norm_(net.parameters(), 0.5)
        if scheduler is not None:
            scheduler.step()        
            

        optimizer.step()
        optimizer.zero_grad()

    if net_type == 'resnet34_c3_locate' or net_type == 'resnet50_c3_locate':
        logger.info(f"Train  Epoch: {epoch}, " +
                f"Lr    : {scheduler.get_lr()[1]:.5f}, " + 
                f"Average Loss: {total_loss.avg:.4f}, " + 
                f"Loss X: {x_loss.avg:.4f}, " + 
                f"Loss Y: {y_loss.avg:.4f}, " + 
                f"Loss W: {w_loss.avg:.4f}, " +
                f"Loss H: {h_loss.avg:.4f} " 
                )
    elif net_type == 'resnet34_c3_locate_centernet': 
        logger.info(f"Train  Epoch: {epoch}, " +
                f"Lr    : {scheduler.get_lr()[1]:.5f}, " + 
                f"Average Loss: {total_loss.avg:.4f}, " + 
                f"Loss Score: {score_loss.avg:.4f}, " + 
                f"Loss XY: {xy_loss.avg:.4f}, " + 
                f"Loss WH: {wh_loss.avg:.4f}, "

                )
        
    return total_loss.avg
              


def test(loader, net, criterion, device,epoch,net_type = 'resnet50_c3_locate'):
    net.eval()

    if net_type == 'resnet34_c3_locate' or net_type == 'resnet50_c3_locate':

        total_loss, x_loss,y_loss,w_loss,h_loss = AvgerageMeter(),AvgerageMeter(),AvgerageMeter(),AvgerageMeter(),AvgerageMeter()    
    
    elif net_type == 'resnet34_c3_locate_centernet':
        total_loss, score_loss,xy_loss,wh_loss = AvgerageMeter(),AvgerageMeter(),AvgerageMeter(),AvgerageMeter()
    
    
    
    for _, data in enumerate(loader):
        images, boxes = data
        images = images.to(device)
        if type(boxes)==dict:
            for key,value in boxes.items():
                boxes[key] = value.to(device)
        else:
            boxes = boxes.to(device)     
       
        

        with torch.no_grad():
            outputs = net(images)
            if net_type == 'resnet34_c3_locate' or net_type == 'resnet50_c3_locate':
                loss,loss_xx,loss_yy,loss_ww,loss_hh = criterion(boxes,outputs)
            
                total_loss.update(loss.item(),images.size(0))
                x_loss.update(loss_xx.item(),images.size(0))
                y_loss.update(loss_yy.item(),images.size(0))
                w_loss.update(loss_ww.item(),images.size(0))
                h_loss.update(loss_hh.item(),images.size(0))
            
            elif net_type == 'resnet34_c3_locate_centernet':   
                loss,loss_score,loss_xy,loss_wh = criterion(boxes,outputs)
            
                total_loss.update(loss.item(),images.size(0))
                score_loss.update(loss_score.item(),images.size(0))
                xy_loss.update(loss_xy.item(),images.size(0))
                wh_loss.update(loss_wh.item(),images.size(0))
                
            #pp = F.softmax(outputs,dim = 1)
            #logger.info("Conf: {}".format(pp[:,1].cpu().numpy()))
            #logger.info(f"labels: {labels}")

      
    if net_type == 'resnet34_c3_locate' or net_type == 'resnet50_c3_locate':
        logger.info(f"Test  Epoch: {epoch}, " +
                f"Average Loss: {total_loss.avg:.4f}, " + 
                f"Loss X: {x_loss.avg:.4f}, " + 
                f"Loss Y: {y_loss.avg:.4f}, " + 
                f"Loss W: {w_loss.avg:.4f}, " +
                f"Loss H: {h_loss.avg:.4f} " 
                )
    elif net_type == 'resnet34_c3_locate_centernet': 
        logger.info(f"Test  Epoch: {epoch}, " +
                f"Average Loss: {total_loss.avg:.4f}, " + 
                f"Loss Score: {score_loss.avg:.4f}, " + 
                f"Loss XY: {xy_loss.avg:.4f}, " + 
                f"Loss WH: {wh_loss.avg:.4f}, "

                )
        
    return total_loss.avg

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train skin box localization')

    parser.add_argument('--datasets', default = '../data/ISIC07_task1_train/ISIC-2017_Img', type=str, help='Dataset directory path')#nargs='+'

    parser.add_argument('--validation_dataset',   default = None, type=str, help='Valid Dataset directory path')
    
    parser.add_argument('--fd_box', default = '../data/ISIC07_task1_train/ISIC-2017_Box', type=str, help='Box directory path')#nargs='+'
    
    parser.add_argument('--fd_box_valid', default = None, type=str, help='Valid Box directory path')#nargs='+'
    
    parser.add_argument('--pct', default = 0.8, type=float, help='pct of train valid split')#nargs='+'

    parser.add_argument('--net', default="resnet50_c3_locate", help="The network architecture")

    parser.add_argument('--seed', default = 0, type=int, help='random seed')
    
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')

    parser.add_argument('--num_epochs', default=120, type=int,help='the number epochs')
    
    parser.add_argument('--validation_epochs', default=1, type=int,help='the number epochs')
    
    parser.add_argument('--sav_epochs', default=10, type=int,help='the number epochs for saving the model')

    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    
    parser.add_argument('--milestones', default="30,60", type=str,help="milestones for MultiStepLR")    
    
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,help='initial learning rate')   
    
    parser.add_argument('--lr_freeze', default=1e-2, type=float,help='initial learning rate')       
    
    parser.add_argument('--num_epochs_freeze', default=0, type=int,help='the number epochs')
    
    parser.add_argument('--K_fold', default=1, type=int,help='K-folder validation')
    
    parser.add_argument('--log_file', default='log.txt', type=str,help='log_file')
    
    args = parser.parse_args()
    
    if not osp.exists(args.datasets) :
        raise ValueError(f"Dataset Folder not exist")  

    log_dir = osp.join('../checkpoints')
    os.makedirs(log_dir,exist_ok = True)

    set_seed(args.seed)
    
    #%%model  
    if args.net == 'resnet34_c3_locate':

        from configs import resnet34_c3_locate
        cfg = resnet34_c3_locate
        model = models.LocateSkinImgModel_CustomSquare(backbone=cfg.model_type)
    elif args.net == 'resnet50_c3_locate':
        
        
        from configs import resnet50_c3_locate
        cfg = resnet50_c3_locate
        model = models.LocateSkinImgModel_CustomSquare(backbone=cfg.model_type)
    elif args.net == 'resnet34_c3_locate_centernet':
        from configs import resnet34_c3_locate_centernet
        cfg = resnet34_c3_locate_centernet
        model = models.LocateSkinImgModel_CustomSquare_Centernet(backbone=cfg.model_type)

    else:
        raise ValueError(f"Model not exist")
    



    #%% augmentation dataset and dataloader
    
    train_transform = TrainAugmentation_bbox_albu(cfg.image_size, cfg.image_mean, cfg.image_std)
    
    test_transform = TestAugmentation_bbox_albu(cfg.image_size, cfg.image_mean, cfg.image_std)

    if args.net == 'resnet34_c3_locate' or args.net == 'resnet50_c3_locate':
        CustomDataset = CustomDataset_bbox
    elif args.net == 'resnet34_c3_locate_centernet':
        CustomDataset = CustomDataset_bbox_centernet
    
    
    if args.validation_dataset is not None:
        train_dataset =  CustomDataset(root = args.datasets, box_root = args.fd_box,transform = train_transform,cfg = cfg)
        valid_dataset =  CustomDataset(root = args.validation_dataset, box_root = args.fd_box_valid, transform = test_transform,cfg = cfg)
    else:
        train_dataseto =  CustomDataset(root = args.datasets, box_root = args.fd_box,transform = train_transform,cfg = cfg)
        n_samp = len(train_dataseto)
        n_train = round(n_samp*args.pct)
        idx_train = np.random.choice(np.arange(n_samp), n_train,replace = False).astype('int')
        idx_valid = np.setdiff1d(np.arange(n_samp),idx_train).astype('int')
        
        train_dataset = torch.utils.data.Subset(train_dataseto,idx_train)   
        valid_dataset = torch.utils.data.Subset(train_dataseto,idx_valid)  
        valid_dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=True)

    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=False)

#
#    for i, data in enumerate(tqdm(train_loader)):
#        
#        images, labels = data
#        
#    for i, data in enumerate(tqdm(valid_loader)):
#        
#        images, labels = data
#    

    #%% net to DEVICE
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    
    
    
    #%% loss 

    if args.net == 'resnet34_c3_locate' or args.net == 'resnet50_c3_locate':
        criterion = Location_Box_Loss(prior_w = cfg.prior_w, prior_cx=cfg.prior_cx, prior_cy=cfg.prior_cy,size_variance = cfg.size_variance,
                                      center_variance = cfg.center_variance)
    
    elif args.net == 'resnet34_c3_locate_centernet':
        criterion =  Location_Box_CenterNet()
    criterion = criterion.to(device)
    
    #50 epoch freeze base
    #num_epoch_fix = 100
    
#    for param in model.backbone.parameters():
#        param.requires_grad = False
#    
#    optim_freeze = SGD(model.head.parameters(),  lr=args.lr_freeze, momentum=0.9,weight_decay=5e-04)
#    
#    scheduler_freeze = CyclicLR(optim_freeze, base_lr=0.1*args.lr_freeze, max_lr=args.lr_freeze, 
#                                                 step_size_up=len(train_loader)*args.num_epochs_freeze//2, step_size_down=None, 
#                                                 mode='triangular',  scale_mode='cycle', 
#                                                 cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
#
#
#    min_loss = 100000
#    
#    for epoch in range(args.num_epochs_freeze):
#        #scheduler.step()
#        train(train_loader, model, criterion, optim_freeze, device,epoch=epoch, up_freq = 1, scheduler = scheduler_freeze)
#
#        
#        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs_freeze - 1:
#            epoch_loss = test(valid_loader, model, criterion, device,epoch=epoch)
#        
#        if epoch % args.sav_epochs == 0 or epoch == args.num_epochs_freeze - 1:
#            
#            model_path = osp.join(log_dir, f"{args.net}-Epoch-{epoch}-loss-{epoch_loss:.4f}-freeze.pth")
#
#            torch.save(model.state_dict(),model_path)
#            logger.info(f"Saved model {model_path}")
#    
#    
#        # deep copy the model
#        if epoch_loss < min_loss:
#
#            min_loss = epoch_loss
#            best_model_wts = copy.deepcopy(model.state_dict())
#            logger.info('Best val loss: {:.4f}'.format(min_loss))
#    
#    
#
#    # Unfreeze
#    for param in model.backbone.parameters():
#        param.requires_grad = True

    min_loss = 100000
    optim_unfreeze = SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr/10},
        {'params': model.head.parameters(), 'lr': args.lr},
    ],momentum=0.9,weight_decay=5e-04)



    scheduler_unfreeze = CyclicLR(optim_unfreeze,
                                  base_lr=[0.01*args.lr, 0.1*args.lr], max_lr=[0.1*args.lr, args.lr],
                                  step_size_up=int(args.num_epochs*len(train_loader)/2),
                                  step_size_down=None, 
                                  mode='triangular',  scale_mode='cycle', 
                                  cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)



    for epoch in range(args.num_epochs):
        #scheduler.step()
        train(train_loader, model, criterion, optim_unfreeze, device,epoch=epoch, scheduler = scheduler_unfreeze,net_type = args.net)

        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            epoch_loss = test(valid_loader, model, criterion, device,epoch=epoch,net_type = args.net)
        
        if epoch % args.sav_epochs == 0 or epoch == args.num_epochs - 1:
            
            model_path = osp.join(log_dir, f"{args.net}-Epoch-{epoch}-loss-{epoch_loss:.4f}.pth")

            torch.save(model.state_dict(),model_path)
            logger.info(f"Saved model {model_path}")
    
    
        # deep copy the model
        if epoch_loss < min_loss:
            
            min_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            logger.info('Best val loss: {:.4f}'.format(min_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(),osp.join(log_dir, f"{args.net}-best.pth"))
