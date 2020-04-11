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
from tqdm import tqdm
import os.path as osp
import globalvar as gl
from config import cfg
from utils.utils import set_seed
from tools.loggers import call_logger

import torch
#from torch.utils.tensorboard import SummaryWriter

from modeling import build_model
from data import make_data_loader
from loss_layers import make_loss
from optim import build_optimizer
from torch.utils.data import DataLoader


from engine.BaseTrain import BaseTrainer
#from engine.BaseTest import test_tta

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train ISIC classification')

    parser.add_argument("--config_file", default="", help="path to config file", type=str)

    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()  #skip this, cfg can be modified 

    gl._init()
    gl.set_value('cfg',cfg)

    set_seed(cfg.MISC.SEED)
    #writer = SummaryWriter()
    #gl.set_value('writer', writer)

    output_dir = cfg.MISC.OUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = call_logger(osp.join(output_dir, cfg.MISC.LOGFILE))
    gl.set_value('logger',logger)
    logger.info("Running with config:\n{}".format(cfg))

    
    
    # prepare model
    model = build_model(cfg) 
    torch.save(model.state_dict(),osp.join(output_dir, f"{cfg.MODEL.NAME}-init.pth"))    


    # make dataloader
    train_loader_all, valid_loader_all = make_data_loader(cfg)

    # make loss
    criterion = make_loss(cfg)  


    #%% start train
    start_epoch = cfg.SOLVER.START_EPOCH
    for nf in range(cfg.MISC.START_FOLD, cfg.DATASETS.K_FOLD):
        logger.info(f'start fold {nf}')
        model.load_state_dict(torch.load(osp.join(output_dir, f"{cfg.MODEL.NAME}-init.pth")))
         
        
        # DataLoader
        num_workers = cfg.DATALOADER.NUM_WORKERS
        batch_size  = cfg.DATALOADER.BATCH_SIZE


            
        train_loader = train_loader_all[nf]
        valid_loader = valid_loader_all[nf]
        #make optimizer and scheduler
        optimizer, scheduler = build_optimizer(cfg,model,len(train_loader))


 
        if cfg.MISC.ONLY_TEST is False:
            # train
            trainer = BaseTrainer(cfg, model, train_loader, valid_loader, criterion, optimizer,  scheduler, start_epoch, nf)


            for epoch in range(1,cfg.SOLVER.EPOCHS+1):
                for batch in trainer.train_dl:
                    trainer.step(batch)
                    trainer.handle_new_batch()
                trainer.handle_new_epoch()
        else:
            #test
            #tester = test_tta(cfg, model, train_loader, valid_loader,nf)
            
            
            pass
        
            



           

    #train(cfg)

    #writer.close()



'''
 
    
    
    
    pred_out_all = []
    for nf in range(args.K_fold):

        
        train_ds = train_ds_all[nf]
        valid_ds = valid_ds_all[nf]

        if args.imbalance_batchsampler==1:
            train_loader = DataLoader(train_ds, batch_size = args.batch_size, sampler=ImbalancedDatasetSampler(train_ds),num_workers=args.num_workers,drop_last=True)    
        else:
            train_loader = DataLoader(train_ds, batch_size = args.batch_size, num_workers=args.num_workers,shuffle = True,drop_last=True)    
    
        
        valid_loader = DataLoader(valid_ds, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    
        model.init()
        

    

            
        if args.only_test==1:
            model.load_state_dict(torch.load(best_model_fn))
            if args.tta==1:
                valid_ds.transform= train_transform
                epoch_loss,epoch_acc,pred_out = test_tta(valid_ds, model, criterion, device,epoch=-1,n_tta = 10,n_class = args.n_class)
                
            else:
                epoch_loss,epoch_acc,pred_out = test_tta(valid_ds, model, criterion, device,epoch=0,n_tta = 1,n_class= args.n_class)
            
            
            fns_kfd = np.array(dataseto.fname)[valid_ds.indices]
            pred_out = np.hstack((fns_kfd[:,None],np.array(pred_out)))
            pred_out_all.append(pred_out)        
            
        else:    
            
            for epoch in range(args.num_epochs):
        
                epoch_loss_train,epoch_acc_train = train(train_loader, model, criterion, optim_unfreeze, device,epoch=epoch, up_freq = 1, scheduler = scheduler_unfreeze)
        
                
                if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
                    epoch_loss,epoch_acc = test(valid_loader, model, criterion, device,epoch=epoch)
                
                if epoch % args.sav_epochs == 0 or epoch == args.num_epochs - 1:
                    
                    model_path = osp.join(args.out_dir, f"{args.net}-Fold-{nf}-Epoch-{epoch}-trainloss-{epoch_loss_train:.4f}-loss-{epoch_loss:.4f}-trainacc-{epoch_acc_train:.4f}-acc-{epoch_acc:.4f}.pth")
        
                    torch.save(model.state_dict(),model_path)
        
            
            
                # deep copy the model
                if epoch_loss < min_loss:
                    best_acc = epoch_acc
                    min_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
        
                    logger.info('Best val Acc and loss: {:.4f}, {:.4f}'.format(best_acc,min_loss))
        
            # load best model weights
            model.load_state_dict(best_model_wts)
        
            torch.save(model.state_dict(),best_model_fn)
    if args.only_test==1:   
        pred_out_all = np.vstack(pred_out_all)    
        df = pd.DataFrame(data = pred_out_all[:,1:].astype('float32'),index =pred_out_all[:,0], columns = [ 'MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC','pred', 'GT'])
        for col in ['MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']:
            df[col] = df[col].apply(lambda x: format(x,'.4f'))
        for col in ['pred', 'GT']:
            df[col] = df[col].apply(lambda x: format(x,'.0f'))    
            
        eval_path = osp.join(args.out_dir, f"eval_{args.net}-Loss-{args.loss_type}-tta-{args.tta}.csv")
        df.to_csv(eval_path, index_label = 'fn')
'''




'''
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
from torch.optim.lr_scheduler import CyclicLR,OneCycleLR



from sklearn.metrics import confusion_matrix




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
        
    for _, data in enumerate(loader):
        images, images_roi, labels,meta_infos = data
        y_true.extend(labels.cpu().numpy())
        
        images = images.to(device)
        
        images_roi = images_roi.to(device) 
        meta_infos = meta_infos.to(device)
        
        
        labels = labels.to(device)

        with torch.no_grad():
            if net.mode =='metatwoview':
                outputs = net(images,images_roi,meta_infos)
            elif net.mode =='twoview':
                outputs = net(images,images_roi)
            elif net.mode =='singleview':
                outputs = net(images)
            elif net.mode =='metasingleview':
                outputs = net(images,meta_infos)
                
            _, preds = torch.max(outputs, 1)
            
            y_pred.extend(preds.cpu().numpy())
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
        
    y_true = np.array(y_true).astype('int64') 
    y_pred = np.array(y_pred).astype('int64') 
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
    
    
    return avg_loss,bal_acc1

    
    
def test_tta(ds, net, criterion, device,epoch = -1,n_tta = 10,n_class = 4):
    net.eval()
    
    # in tta, default batch size =1 and no 
    

    
    n_case = 0.0
    y_true = list()
    y_pred = list()  
    total_loss = AvgerageMeter()

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
                
                loss = criterion(outputs, labels[None,...])
                total_loss.update(loss.item())    
                
                probs_0 = F.softmax(outputs,dim=1)[0].cpu()
                pred_sum = pred_sum + probs_0
                    
                
            pred_sum = pred_sum/n_tta
            n_case += 1
            probs = np.round_(pred_sum.numpy(),decimals=4)
            
            preds = torch.argmax(pred_sum).item()
            
            y_pred.append(preds)
            
            
            
            PREDS_ALL.append([*probs,preds, int(labels.item())])
            
    PREDS_ALL = np.array(PREDS_ALL)
    avg_acc =   (PREDS_ALL[:,-2] == PREDS_ALL[:,-1]).sum()/n_case

    cm = confusion_matrix(y_true, y_pred)    
    
    logger.info(f"Valid  Epoch: {epoch}, " +
                f"Average Loss: {total_loss.avg:.4f}, " +
                f"Average Acc:  {avg_acc}, " )
    logger.info(cm)
    
    return total_loss.avg,avg_acc,PREDS_ALL
'''