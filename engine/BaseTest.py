# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:19:44 2020

@author: cmj
"""
import os
import os.path as osp
import globalvar as gl

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.utils import AvgerageMeter
import torch.nn.functional as F
from utils.utils import get_device
from loss_layers.class_loss import pcsoftmax

def parse_batch(batch):
    if len(batch)==2:
         images, targets = batch
         meta_infos = None
    elif len(batch)==3:
        images, targets,meta_infos = batch
    else:
        raise ValueError('parse batch error')
    
    targets = torch.tensor(targets)
#    if meta_infos is not None:
#        meta_infos = torch.from_numpy(meta_infos)
    
    return images,targets,meta_infos
    
def test_tta(cfg, model, ds, criterion,nf):
    #epoch_loss,epoch_acc,pred_out  = test_tta(cfg, model, valid_loader,criterion,nf)

    #ds, net, criterion, device,epoch = -1,n_tta = 10,n_class = 4
    model.eval()
    
    device = get_device(cfg)
    logger = gl.get_value('logger')
    
    if cfg.DATASETS.K_FOLD ==1:
        best_model_fn = osp.join(cfg.MISC.OUT_DIR, f"{cfg.MODEL.NAME}-best.pth")
    else:
        best_model_fn = osp.join(cfg.MISC.OUT_DIR, f"{cfg.MODEL.NAME}-Fold-{nf}-best.pth")
    model.load_state_dict(torch.load(best_model_fn))
            
    n_tta = cfg.MISC.N_TTA
    n_class = cfg.DATASETS.NUM_CLASS 
   
    # in tta, default batch size =1 
    n_case = 0.0
    y_true = list()
    y_pred = list()  
    total_loss = AvgerageMeter()

    PREDS_ALL = []
    for idx in tqdm(range(len(ds))):
        
        #print(images.shape)

        with torch.no_grad():
            if cfg.MISC.TTA_MODE in ['mean','mean_softmax']:
                pred_sum = torch.zeros((n_class),dtype = torch.float32)
            else:
                pred_sum = torch.ones((n_class),dtype = torch.float32)
            
            for n_t in range(n_tta):
                
                images,  labels,meta_infos = parse_batch(ds[idx])
                
                if n_t==0:
                    y_true.append(labels.item())
        
                images = images.to(device)
                if meta_infos is not None:
                    meta_infos = meta_infos.to(device)                
                labels = labels.to(device)
                
 

                if 'SingleView' in cfg.MODEL.NAME  or  'SVBNN' in cfg.MODEL.NAME :
                    outputs = model(images[None,...])
                elif model.mode =='metasingleview':
                    outputs = model(images[None,...],meta_infos[None,...])
                
                
                elif model.mode in ['sv_att','sv_db']:
            
                    outputs = model(images[None,...],labels[None,...])
                    
                    
                if cfg.MISC.ONLY_TEST is False and cfg.DATASETS.NAMES == 'ISIC':
                    loss = criterion(outputs, labels[None,...])
                    total_loss.update(loss.item())    
                
                
                #if cfg.MODEL.LOSS_TYPE == 'pcs':
                    #probs_0 = pcsoftmax(outputs,weight = torch.tensor(cfg.DATASETS.LABEL_W),dim=1)[0].cpu()
                #else:
                if isinstance(outputs,(list,tuple)):
                    probs_0 = 0.5*(F.softmax(outputs[0],dim=1)[0] + F.softmax(outputs[1],dim=1)[0]).cpu()
                else:
                    if 'softmax' in cfg.MISC.TTA_MODE:
                        probs_0 = outputs[0].cpu()    
                    else:
                        probs_0 = F.softmax(outputs,dim=1)[0].cpu()
                if 'mean' in cfg.MISC.TTA_MODE:
                    pred_sum = pred_sum + probs_0
                else:
                    pred_sum = pred_sum * probs_0
                    
            if 'mean' in cfg.MISC.TTA_MODE:
                pred_sum = pred_sum/n_tta
            else:
                pred_sum = np.power(pred_sum,1.0/n_tta)
                
            if 'softmax' in cfg.MISC.TTA_MODE:
                pred_sum = F.softmax(pred_sum[None,...],dim=1)[0]
            
            
            n_case += 1
            probs = np.round_(pred_sum.numpy(),decimals=4)
            
            preds = torch.argmax(pred_sum).item()
            
            y_pred.append(preds)
            
            
            if cfg.MISC.ONLY_TEST is False:
                PREDS_ALL.append([*probs,preds, int(labels.item())])
            else:
                PREDS_ALL.append([*probs,preds])
            
            
            
            
    PREDS_ALL = np.array(PREDS_ALL)
    #avg_acc =   (PREDS_ALL[:,-2] == PREDS_ALL[:,-1]).sum()/n_case
    np.set_printoptions(precision=4)
    
    
    
    if cfg.MISC.ONLY_TEST is False:
        pred_stat = calc_stat(y_pred,y_true)
        logger.info(f"Valid  K-fold: {nf}")
        if n_class<=10:
            logger.info('confusion matix\n') 
            cm = pred_stat['cm']
            logger.info('{}\n'.format(cm))
            logger.info("Num All Class: {}".format(np.sum(cm,axis = 1)))
            logger.info("Acc All Class1: {}".format(pred_stat['cls_acc1']))
            logger.info("Acc All Class2: {}".format(pred_stat['cls_acc2']))
            logger.info("Acc All Class3: {}".format(pred_stat['cls_acc3']))
    
        logger.info(f"Balance Acc 1 2 3 : {pred_stat['bal_acc1']:.4f} {pred_stat['bal_acc2']:.4f} {pred_stat['bal_acc3']:.4f}")
        
        logger.info(f"Average Loss: {total_loss.avg:.4f}, " +
                    f"Average Acc:  {pred_stat['avg_acc']}" )
    
        return total_loss.avg,pred_stat['bal_acc1'],PREDS_ALL
    else:
        return PREDS_ALL



def calc_stat(y_pred,y_true):
    y_true = np.array(y_true).astype('int64') 
    y_pred = np.array(y_pred).astype('int64') 
    cm = confusion_matrix(y_true, y_pred)    
    cls_acc1 = cm.diagonal()/(0.0001+np.sum(cm,axis = 1))
    cls_acc2 = cm.diagonal()/(0.0001+np.sum(cm,axis = 0))
    cls_acc3  = cm.diagonal()/(0.0001+ np.sum(cm,axis = 0) +  np.sum(cm,axis = 1)-cm.diagonal()) 

    avg_acc = np.sum(cm.diagonal())/cm.sum()
    bal_acc1 = np.mean(cls_acc1)
    bal_acc2 = np.mean(cls_acc2)
    bal_acc3 = np.mean(cls_acc3)

    pred_stat = {'cm':cm, 'avg_acc':avg_acc, 'bal_acc1':bal_acc1, 'bal_acc2':bal_acc2,'bal_acc3':bal_acc3, 'cls_acc1':cls_acc1,'cls_acc2':cls_acc2,'cls_acc3':cls_acc3}
    return pred_stat