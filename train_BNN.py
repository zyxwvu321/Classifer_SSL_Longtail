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
from engine.BaseTrain_SSL import BaseTrainer_SSL
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

    
    
   


    # make dataloader
    train_loader_all, valid_loader_all = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg) 
    
    torch.save(model.state_dict(),osp.join(output_dir, f"{cfg.MODEL.NAME}-init.pth")) 
    
    

    # make loss
    criterion = make_loss(cfg)  


    #%% start train
    start_epoch = cfg.SOLVER.START_EPOCH
    for nf in range(cfg.MISC.START_FOLD, cfg.DATASETS.K_FOLD):
        logger.info(f'start fold {nf}')
        model.load_state_dict(torch.load(osp.join(output_dir, f"{cfg.MODEL.NAME}-init.pth")))
        #model.load_state_dict(torch.load(osp.join(output_dir, 'effb4_SVMeta_SSL-best.pth'))) 
        #model.load_state_dict(torch.load('../checkpoint/effb4_meta_default/effb4_SVMeta-best.pth')) 
        
        
        # DataLoader
        num_workers = cfg.DATALOADER.NUM_WORKERS
        batch_size  = cfg.DATALOADER.BATCH_SIZE


            
        train_loader = train_loader_all[nf]
        valid_loader = valid_loader_all[nf]
        
        
        #make optimizer and scheduler
        if 'SVBNN' in cfg.MODEL.NAME:
            n_loader = len(train_loader[0])
        else:
            n_loader = len(train_loader)
            
        optimizer, scheduler = build_optimizer(cfg,model,n_loader)


 
        if cfg.MISC.ONLY_TEST is False:
            # train
            if 'SVBNN' in cfg.MODEL.NAME:
                trainer = BaseTrainer(cfg, model, train_loader[0], valid_loader, criterion, optimizer,  scheduler, start_epoch, nf)
            elif cfg.MODEL.SSL_FIXMATCH is True:
                trainer = BaseTrainer_SSL(cfg, model, train_loader, valid_loader, criterion, optimizer,  scheduler, start_epoch, nf)
            else:
                trainer = BaseTrainer(cfg, model, train_loader, valid_loader, criterion, optimizer,  scheduler, start_epoch, nf)

            for epoch in range(1,cfg.SOLVER.EPOCHS+1):
                if 'SVBNN' in cfg.MODEL.NAME:
                    for batch_uni, batch_bal in zip(train_loader[0],train_loader[1]):
                        trainer.step([batch_uni,batch_bal])
                        trainer.handle_new_batch()
                    trainer.handle_new_epoch()
                else:
                    for batch in trainer.train_dl:
                        trainer.step(batch)
                        trainer.handle_new_batch()
                    trainer.handle_new_epoch()
        else:
            #test
            #tester = test_tta(cfg, model, train_loader, valid_loader,nf)
            
            
            pass
        
            



      