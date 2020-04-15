#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:18:58 2019
Test with global, local, and meta infos
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
import numpy as np
import pandas as pd
#from torch.utils.tensorboard import SummaryWriter

from modeling import build_model
from data import make_data_loader_test
from loss_layers import make_loss



#from engine.BaseTrain import BaseTrainer
from engine.BaseTest import test_tta

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='General Classifcation')

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
    logger = call_logger(osp.join(output_dir, cfg.MISC.LOGFILE_EVAL))
    gl.set_value('logger',logger)
    logger.info("Running with config:\n{}".format(cfg))

    
    
    # prepare model
    model = build_model(cfg) 
    

    
    # make dataloader
    val_ds_all,fns_kfds = make_data_loader_test(cfg)

    # make loss
    if cfg.MISC.ONLY_TEST is True and cfg.DATASETS.NAMES =='ISIC':
        cfg.DATASETS.LABEL_W  = [1.2854575792581184, 0.21338020666879728, 2.7834908282379103, 4.375273044997816, 1.301832835044846, 12.440993788819876, 10.075452716297788]
    criterion = make_loss(cfg)  


    #%% start Test
    pred_out_all = []
    
    for nf in range(cfg.DATASETS.K_FOLD):
        logger.info(f'start Test fold {nf}')
       

        valid_ds   = val_ds_all[nf]
        fns_kfd    = fns_kfds[nf]
        
        
        if cfg.MISC.ONLY_TEST is False:
            epoch_loss,epoch_acc,pred_out  = test_tta(cfg, model, valid_ds,criterion,nf)
        else:
            pred_out  = test_tta(cfg, model, valid_ds,criterion,nf)
        

        pred_out = np.hstack((fns_kfd[:,None],np.array(pred_out)))
        pred_out_all.append(pred_out)       
        

    if cfg.DATASETS.NAMES =='ISIC':
        pred_out_all = np.vstack(pred_out_all)    
        is_tta = 1 if cfg.MISC.TTA is True else 0
                     

        if cfg.MISC.ONLY_TEST is False:
            df = pd.DataFrame(data = pred_out_all[:,1:].astype('float32'),index =pred_out_all[:,0], columns = [ *cfg.DATASETS.DICT_LABEL,'pred', 'GT'])
            for col in cfg.DATASETS.DICT_LABEL:
                df[col] = df[col].apply(lambda x: format(x,'.4f'))
            for col in ['pred', 'GT']:
                df[col] = df[col].apply(lambda x: format(x,'.0f'))    
            
            eval_path = osp.join(output_dir, f"eval_{cfg.MODEL.NAME}-Loss-{cfg.MODEL.LOSS_TYPE}-tta-{is_tta}.csv")
        else:
            
            pred_out_all = np.vstack(pred_out_all)    
            df = pd.DataFrame(data = pred_out_all[:,1:-1].astype('float32'),index =pred_out_all[:,0], columns = cfg.DATASETS.DICT_LABEL)
            for col in cfg.DATASETS.DICT_LABEL:
                df[col] = df[col].apply(lambda x: format(x,'.4f'))
            
            eval_path = osp.join(output_dir, f"eval_{cfg.MODEL.NAME}-Loss-{cfg.MODEL.LOSS_TYPE}-tta-{is_tta}-test.csv")
            
        df.to_csv(eval_path, index_label = 'image')
    
    
    
    elif cfg.DATASETS.NAMES =='flower':
        pred_out_all = np.vstack(pred_out_all)    
        is_tta = 1 if cfg.MISC.TTA is True else 0
        if cfg.MISC.ONLY_TEST is False:
            df = pd.DataFrame(data = pred_out_all[:,1:].astype('float32'),index =pred_out_all[:,0])
#            for col in ['MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']:
#                df[col] = df[col].apply(lambda x: format(x,'.4f'))
#            for col in ['pred', 'GT']:
#                df[col] = df[col].apply(lambda x: format(x,'.0f'))    
#            
            eval_path = osp.join(output_dir, f"eval_{cfg.MODEL.NAME}-Loss-{cfg.MODEL.LOSS_TYPE}-tta-{is_tta}.csv")
        else:
            
            pred_out_all = np.vstack(pred_out_all)    
            df = pd.DataFrame(data = pred_out_all[:,1:-1].astype('float32'),index =pred_out_all[:,0])
#            for col in ['MEL', 'NV','BCC', 'AKIEC', 'BKL', 'DF','VASC']:
#                df[col] = df[col].apply(lambda x: format(x,'.4f'))
#            
            eval_path = osp.join(output_dir, f"eval_{cfg.MODEL.NAME}-Loss-{cfg.MODEL.LOSS_TYPE}-tta-{is_tta}-test.csv")
        df.to_csv(eval_path, index_label = 'fn')
        
    else:
        
        
        raise ValueError('dataset not implemented for test {cfg.DATASETS.NAMES}')
    #train(cfg)

    #writer.close()


