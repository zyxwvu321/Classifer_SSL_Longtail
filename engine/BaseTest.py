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
    PREDS_ALL_TTA = []
    for idx in tqdm(range(len(ds))):
        
        #print(images.shape)

        with torch.no_grad():
#            if cfg.MISC.TTA_MODE in ['mean','mean_softmax']:
#                pred_sum = torch.zeros((n_class),dtype = torch.float32)
#            else:
#                pred_sum = torch.ones((n_class),dtype = torch.float32)
#            
            #for n_t in range(n_tta):
                
            images,  labels,meta_infos = parse_batch(ds[idx])
            
            
            y_true.append(labels.item())
    
            images = images.to(device)
            if meta_infos is not None:
                meta_infos = meta_infos.to(device)  
                
                if meta_infos.dim()==1:
                    meta_infos = meta_infos[None,...]
                
                if images.dim()>3 and meta_infos.size(0)==1:
                    
                    meta_infos = meta_infos.repeat(images.size(0),1)
                
            labels = labels.to(device)
            labels = labels[None,...]
            if images.dim()==3:
                images = images[None,...]
            

            if 'SingleView' in cfg.MODEL.NAME  or  'SVBNN' in cfg.MODEL.NAME :
                outputs = model(images)
            elif model.mode =='metasingleview':
                outputs = model(images,meta_infos)
            
            elif model.mode in ['sv_att','sv_db']:
        
                outputs = model(images,labels)
                
                
            if cfg.MISC.ONLY_TEST is False and cfg.DATASETS.NAMES == 'ISIC':
                loss = criterion(outputs, labels)
                total_loss.update(loss.item())    
            
            
            #if cfg.MODEL.LOSS_TYPE == 'pcs':
                #probs_0 = pcsoftmax(outputs,weight = torch.tensor(cfg.DATASETS.LABEL_W),dim=1)[0].cpu()
            #else:
            if isinstance(outputs,(list,tuple)):
                probs_0 = 0.5*(F.softmax(outputs[0],dim=1)[0] + F.softmax(outputs[1],dim=1)[0]).cpu()
            else:
                
                if 'softmax' in cfg.MISC.TTA_MODE:
                    probs_0 = outputs.cpu().numpy()    
                else:
                    probs_0 = F.softmax(outputs,dim=-1).cpu().numpy()
                    
            #save outputs result
            #if cfg.MISC.ONLY_TEST is True:
            PREDS_ALL_TTA.append(outputs.cpu().numpy())
            
                    
            if cfg.MISC.TTA_MODE in ['mean','mean_softmax']:
                pred_sum = np.mean(probs_0,axis = 0)
            else:
                pred_sum = np.prod(probs_0,axis = 0)
                pred_sum = np.power(pred_sum,1.0/n_tta)
                

                

            
            
            n_case += 1
            probs = np.round_(pred_sum,decimals=4)
            
            preds = np.argmax(pred_sum)
            
            y_pred.append(preds)
            
            
            if cfg.MISC.ONLY_TEST is False:
                PREDS_ALL.append([*probs,preds, int(labels.item())])
            else:
                PREDS_ALL.append([*probs,preds])
            
            
            
            
    PREDS_ALL = np.array(PREDS_ALL)
    PREDS_ALL_TTA = np.array(PREDS_ALL_TTA)
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
    
        return total_loss.avg,pred_stat['bal_acc1'],PREDS_ALL,PREDS_ALL_TTA
    else:
        return PREDS_ALL,PREDS_ALL_TTA



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