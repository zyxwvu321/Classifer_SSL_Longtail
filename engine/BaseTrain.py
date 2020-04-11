#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-11-7
# @Author  : Minjie

import os
import os.path as osp
import time
import torch
import globalvar as gl
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.utils import AvgerageMeter
import torch.nn.functional as F
from modeling.mixup import mixup_data, mixup_criterion,cutmix_data

class BaseTrainer(object):
    def __init__(self, cfg, model, train_dl,  val_dl, criterion, optimizer,  scheduler,  start_epoch, nf):

        self.cfg = cfg
        self.logger = gl.get_value('logger')
        self.train_dl =train_dl
        self.val_dl = val_dl
        self.model = model

        self.train_epoch = start_epoch 
        self.batch_cnt = 0

        use_cuda = torch.cuda.is_available()
        if cfg.MODEL.DEVICE != 'cuda':
            use_cuda = False

        self.device = torch.device('cuda' if use_cuda else 'cpu')
      
        self.mgpu = True if self.device =='cuda' and torch.cuda.device_count() > 1 else False

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.start_time = time.time()
        self.epoch_start_time = time.time()

        self.total_loss = AvgerageMeter()
        
        self.global_loss = AvgerageMeter()
        self.drop_loss = AvgerageMeter()
        self.crop_loss = AvgerageMeter()
        self.center_loss = AvgerageMeter()
        
        self.best_metric = 0.0

        self.nf  = nf
        self.logger.info('Trainer Start')

        self.y_true = list()
        self.y_pred = list()
        self.n_count = 0.0
        self.n_correct = 0.0


    #@(Events.ITERATION_COMPLETED)
    def handle_new_batch(self):
        self.batch_cnt += 1
        if self.batch_cnt % self.cfg.MISC.LOG_PERIOD == 0 or self.batch_cnt == len(self.train_dl):
            elapsed = time.time() - self.start_time

            pstring = (
                "epoch {:2d} | {:4d}/{:4d} batches | ms {:4.02f} | lr {:1.6f}| "
                "acc {:03.03%} | loss {:3.4f}".format(
                    self.train_epoch,
                    self.batch_cnt,
                    len(self.train_dl),
                    elapsed / self.cfg.MISC.LOG_PERIOD,
                    self.scheduler.get_lr()[0],
                    self.n_correct/self.n_count,
                    self.total_loss.avg)
            )
            self.logger.info(f"{pstring}")
            
            if self.cfg.MODEL.LOSS_ATT is True:
               pstring_loss =  ("global loss {:3.4f} | drop loss {:3.4f} | crop loss {:3.4f} | center loss {:3.4f}"  \
                                .format(self.global_loss.avg,self.drop_loss.avg,self.crop_loss.avg,self.center_loss.avg))
               self.logger.info(f"{pstring_loss}")
            if 'SVreid' in self.cfg.MODEL.NAME:
               pstring_loss =  ("global loss {:3.4f} |center loss {:3.4f}"  \
                                .format(self.global_loss.avg,self.center_loss.avg))
               self.logger.info(f"{pstring_loss}")
            self.start_time = time.time()

    


    def step(self, batch):
        self.model.train()
        
        if 'SVBNN' in self.cfg.MODEL.NAME :
            images, targets_a,meta_infos_a = self.parse_batch(batch[0])
            images_bal, targets_b,meta_infos_b = self.parse_batch(batch[1])
            targets = targets_b
            lam =   1.0-((self.train_epoch - 1) / (self.cfg.SOLVER.EPOCHS - 1)) ** 2  # parabolic decay
            
            #lam =   1.0-((self.train_epoch - 1) / (self.cfg.SOLVER.EPOCHS - 1))   # parabolic decay
            
        else:
            images, targets,meta_infos = self.parse_batch(batch)
        
            if self.cfg.MODEL.MIXUP is True:
                
                if self.cfg.MODEL.MIXUP_MODE == 'mixup': #('mixup', 'cutmix')
                    mixfunc = mixup_data
                elif self.cfg.MODEL.MIXUP_MODE == 'cutmix': #('mixup', 'cutmix')
                    mixfunc = cutmix_data
                
                images, targets_a, targets_b, lam = mixfunc(images, targets, alpha = self.cfg.MODEL.MIXUP_ALPHA,prob = self.cfg.MODEL.MIXUP_PROB)
            else:
                lam = 1.0

        if 'SingleView' in self.cfg.MODEL.NAME  :
            outputs = self.model(images)

        elif 'SVMeta' in self.cfg.MODEL.NAME:
            outputs = self.model(images,meta_infos)

        elif 'SVAtt' in self.cfg.MODEL.NAME or  'SVDB' in self.cfg.MODEL.NAME or  'SVreid' in self.cfg.MODEL.NAME :
            
            outputs = self.model(images,targets,lam)
        elif 'SVBNN' in self.cfg.MODEL.NAME :
            
            outputs = self.model(x = images,x_rb = images_bal, alpha = lam)
            
            
        else:
            raise ValueError('unknown model type {self.cfg.MODEL.NAME}')

        if  self.cfg.MODEL.MIXUP is True or 'SVBNN' in self.cfg.MODEL.NAME :
            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            
        else:
            loss = self.criterion(outputs,targets)
            
        
        
        
        
        if isinstance(loss,(tuple,list)):
            self.total_loss.update(loss[0].item(), images.size(0))    
                    
            self.global_loss.update(loss[1]['global'].item(), images.size(0))   
            
            if 'crop' in loss[1].keys():
                self.crop_loss.update(loss[1]['crop'].item(), images.size(0))   
            if 'drop' in loss[1].keys():                
                self.drop_loss.update(loss[1]['drop'].item(), images.size(0))   
            self.center_loss.update(loss[1]['center'].item(), images.size(0))   
            
        else:

            self.total_loss.update(loss.item(), images.size(0))
        
        
        
        
        self.optimizer.zero_grad()
        if isinstance(outputs,(list,tuple)):
            _, preds = torch.max(outputs[0], 1)
            loss[0].backward()
        else:
            #tensor
            _, preds = torch.max(outputs, 1)
            loss.backward()

        self.y_pred.extend(preds.cpu().numpy())
        self.y_true.extend(targets.cpu().numpy())


        
        #clip_grad_norm_(net.parameters(), 0.5)
    
        self.optimizer.step()
        #self.optimizer.zero_grad()
            
        if self.scheduler is not None:
            self.scheduler.step()          
        
        self.n_correct += torch.sum((preds == targets).float()).item()
        self.n_count += images.size(0)
    
        return
        

    def evaluate(self, data_dl):
        self.model.eval()
        y_true_eval = list()
        y_pred_eval = list()
        #n_count_eval = 0.0
        #n_correct_eval = 0.0
        total_loss_eval = AvgerageMeter()
        loss_global_eval = AvgerageMeter()
        loss_local_eval = AvgerageMeter()
        
        with torch.no_grad():
            for batch in data_dl:
                images, targets,meta_infos = self.parse_batch(batch)

                if 'SingleView' in self.cfg.MODEL.NAME  or  'SVBNN' in self.cfg.MODEL.NAME :
                    outputs = self.model(images)

                elif 'SVMeta' in self.cfg.MODEL.NAME:
                    outputs = self.model(images,meta_infos)
                elif 'SVAtt' in self.cfg.MODEL.NAME or  'SVDB' in self.cfg.MODEL.NAME or 'SVreid' in self.cfg.MODEL.NAME :
            
                    outputs = self.model(images,targets)
                else:
                    raise ValueError('unknown model type {self.cfg.MODEL.NAME}')
            
                
                if 'SVreid' in self.cfg.MODEL.NAME:
                    cls_score,_ = outputs
                    loss = self.criterion.crit(cls_score, targets)
        

                    
                    total_loss_eval.update(loss.item(), images.size(0))
                    _, preds = torch.max(outputs[0], 1)
                elif isinstance(outputs,(list,tuple)):
                    
                    # ave prob to count loss
                    loss1 = self.criterion.crit(outputs[0],targets) 
                    loss2 = self.criterion.crit(outputs[1],targets)
                    loss_global_eval.update(loss1.item(), images.size(0))
                    loss_local_eval.update(loss2.item(), images.size(0))
                    loss = loss1 + loss2
                    total_loss_eval.update(loss.item(), images.size(0))
                    
                    outputs = F.softmax(0.5*(outputs[0] +outputs[1]) , dim=-1)
                    _, preds = torch.max(outputs, 1)
                    

                    
                else:
                    #tensor
                    loss = self.criterion(outputs,targets)
                    
                    total_loss_eval.update(loss.item(), images.size(0))
                    _, preds = torch.max(outputs, 1)

                y_true_eval.extend(targets.cpu().numpy())
                y_pred_eval.extend(preds.cpu().numpy())
        

                #n_correct_eval += torch.sum((preds == targets).float()).item()
                #n_count_eval += images.size(0)
             
        metric = {'loss': total_loss_eval.avg,'y_true_eval':y_true_eval, 'y_pred_eval':y_pred_eval}
        if self.cfg.MODEL.LOSS_ATT is True:
            metric['loss_global'] = loss_global_eval.avg
            metric['loss_local'] = loss_local_eval.avg
            
        return metric

    #@(Events.EPOCH_COMPLETED)
    def handle_new_epoch(self):
        self.batch_cnt = 0
        
        if self.train_epoch % self.cfg.MISC.VALID_EPOCH  == 0:

            self.logger.info("-" * 96)
            val_metric = self.evaluate(self.val_dl)
        
            
            train_acc,valid_acc = self.print_stat(self.y_pred,self.y_true,val_metric['y_pred_eval'],val_metric['y_true_eval'])
            
            #use bal acc as metric instead of loss
            val_ms = float(valid_acc)
            val_loss = val_metric['loss']
            is_best = val_ms > self.best_metric
            
            
            if (is_best and  self.train_epoch>= self.cfg.MISC.SAV_EPOCH) or (self.train_epoch % self.cfg.MISC.SAV_EPOCH == 0):
                dict_sav = {'train_loss':self.total_loss.avg, 'valid_loss':val_loss , 'train_acc':train_acc, 'valid_acc':valid_acc}
                self.save(dict_sav,is_best)

            if is_best :
                self.best_metric = val_ms 
            

            
            self.logger.info("| end of epoch {:3d} | time: {:5.02f}s | val loss {:5.04f} |val ms {:5.03f} | best ms {:5.03f} |".format(
                self.train_epoch, (time.time() - self.epoch_start_time), val_loss,val_ms, self.best_metric)
            )
             
            if self.cfg.MODEL.LOSS_ATT is True:
                self.logger.info("val global {:2.04f}  | val local {:2.04f} | val total {:2.04f}".format(val_metric['loss_global'],val_metric['loss_local'],val_ms))
                
                   
                   
            self.logger.info('Train/val Acc: {:.4f}, {:.4f}'.format(train_acc,valid_acc))


            self.logger.info("-" * 96)

        self.epoch_start_time = time.time()
        self.train_epoch += 1
        self.total_loss.reset()
        self.global_loss.reset() 
        self.drop_loss.reset()
        self.crop_loss.reset() 
        self.center_loss.reset()
        
        
        self.y_true = list()
        self.y_pred = list()
        self.n_count = 0.0
        self.n_correct = 0.0
    def save(self,dict_sav,is_best= False):

        train_loss = dict_sav['train_loss']
        valid_loss = dict_sav['valid_loss'] 
        train_acc  = dict_sav['train_acc']
        valid_acc = dict_sav['valid_acc']

        model_path = osp.join(self.cfg.MISC.OUT_DIR, f"{self.cfg.MODEL.NAME}-Fold-{self.nf}-Epoch-{self.train_epoch}-trainloss-{train_loss:.4f}-loss-{valid_loss:.4f}-trainacc-{train_acc:.4f}-acc-{valid_acc:.4f}.pth")


        torch.save(self.model.state_dict(),model_path)


        if is_best:     

            if self.cfg.DATASETS.K_FOLD ==1:
                best_model_fn = osp.join(self.cfg.MISC.OUT_DIR, f"{self.cfg.MODEL.NAME}-best.pth")
            else:
                best_model_fn = osp.join(self.cfg.MISC.OUT_DIR, f"{self.cfg.MODEL.NAME}-Fold-{self.nf}-best.pth")
            torch.save(self.model.state_dict(),best_model_fn)








    def parse_batch(self,batch):
        if len(batch)==2:
             images, targets = batch
             meta_infos = None
        elif len(batch)==3:
            images, targets,meta_infos = batch
        else:
            raise ValueError('parse batch error')
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        if meta_infos is not None:
            meta_infos = meta_infos.to(self.device)
            
            
        return images,targets,meta_infos

    def print_stat(self,y_pred_tr,y_true_tr,y_pred_vl,y_true_vl):
        ps_tr = self.calc_stat(y_pred_tr,y_true_tr)
        ps_vl = self.calc_stat(y_pred_vl,y_true_vl)

        np.set_printoptions(precision=4)

        self.logger.info(f"Metrics  Epoch: {self.train_epoch}")
        #logger.info(f"Balance Acc 1 2 3 : {bal_acc1:.4f} {bal_acc2:.4f} {bal_acc3:.4f}")


        cm_train = ps_tr['cm']
        cm_valid = ps_vl['cm']
        
        if cm_train.shape[0]<=10: #only n_class<=10 print
            self.logger.info('confusion matix train\n')
            self.logger.info('{}\n'.format(cm_train))
            self.logger.info('confusion matix valid\n')
            self.logger.info('{}\n'.format(cm_valid))
    
    
            self.logger.info("Num All Class Train: {}".format(np.sum(cm_train,axis = 1)))
            self.logger.info("Acc All Class1 Train: {}".format(ps_tr['cls_acc1']))
            self.logger.info("Acc All Class2 Train: {}".format(ps_tr['cls_acc2']))
            self.logger.info("Acc All Class3 Train: {}".format(ps_tr['cls_acc3']))
            
            
        self.logger.info(f"Balance Acc 1 2 3 Train : {ps_tr['bal_acc1']:.4f} {ps_tr['bal_acc2']:.4f} {ps_tr['bal_acc3']:.4f}")
        
        if cm_valid.shape[0]<=10: #only n_class<=10 print
            self.logger.info("Num All Class Valid: {}".format(np.sum(cm_valid,axis = 1)))
            self.logger.info("Acc All Class1 Valid: {}".format(ps_vl['cls_acc1']))
            self.logger.info("Acc All Class2 Valid: {}".format(ps_vl['cls_acc2']))
            self.logger.info("Acc All Class3 Valid: {}".format(ps_vl['cls_acc3']))
        self.logger.info(f"Balance Acc 1 2 3 Valid : {ps_vl['bal_acc1']:.4f} {ps_vl['bal_acc2']:.4f} {ps_vl['bal_acc3']:.4f}")
        self.logger.info(f"Ave Acc : {ps_vl['avg_acc']:.4f}")

        return ps_tr['bal_acc1'],ps_vl['bal_acc1']

    def calc_stat(self, y_pred,y_true):
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
    
    
    
    
    

     
    
    
    
 