# encoding: utf-8


from torch.utils.data import DataLoader,WeightedRandomSampler

#from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset

import copy
import numpy as np
from .samplers import resample_idx_with_meta, ImbalancedDatasetSampler
from .transforms import build_transforms


def make_data_loader(cfg):
    # dataset
    dataseto = init_dataset(cfg)
    
    
    # augmentation 
    if not isinstance(dataseto, list):
        train_transform = build_transforms(cfg, is_train=True,n_aug = cfg.DATASETS.N_AUG)
        val_transform   = build_transforms(cfg, is_train=False)
        
        dataseto_v = copy.deepcopy(dataseto)
        dataseto.transform = train_transform#val_transform#
        dataseto_v.transform = val_transform
        
        
        if cfg.MODEL.SSL_FIXMATCH is True:
            # SSL add weak aug
            train_transform_weak = build_transforms(cfg, is_train=True,weak_aug = True,n_aug = cfg.DATASETS.N_AUG)    
            dataseto.transform_weak = train_transform_weak
        
        
        # loader
        train_ds_all = [dataseto]
        valid_ds_all = [dataseto_v]
        
        
#        # resampler
#        train_ds_all, valid_ds_all = resample_idx_with_meta(dataseto,dataseto_v, K_fold = cfg.DATASETS.K_FOLD, pct = cfg.DATASETS.PCT)
    
    else:
        #already return train and valid dataset (list and include transform)
        train_ds_all = [dataseto[0]]
        valid_ds_all = [dataseto[1]]
        cfg.DATASETS.K_FOLD = 1
    
    #train_sampler_all, valid_sampler_all = resample_idx_with_meta(dataseto, K_fold = cfg.DATASETS.K_FOLD, pct = cfg.DATASETS.PCT)

    #return train_ds_all, valid_ds_all

    # DataLoader
    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size  = cfg.DATALOADER.BATCH_SIZE

    train_loaders = list()
    val_loaders   = list()

    for nf in range(cfg.DATASETS.K_FOLD):


        
        train_ds = train_ds_all[nf]
        valid_ds = valid_ds_all[nf]

        
        if 'SVBNN' in cfg.MODEL.NAME:
            #there are two sampler
            
            
            train_loader_uni = DataLoader(train_ds, batch_size = batch_size, num_workers=num_workers,shuffle = True, drop_last=True)    
            
            train_loader_bal = DataLoader(train_ds, batch_size = batch_size, sampler=ImbalancedDatasetSampler(train_ds),num_workers=num_workers, drop_last=True)
            train_loaders.append([train_loader_uni,train_loader_bal])
            
        else:
            if cfg.DATALOADER.SAMPLER =='imbalance':
                train_loader = DataLoader(train_ds, batch_size = batch_size, sampler=ImbalancedDatasetSampler(train_ds),num_workers=num_workers, drop_last=True)
                #raise ValueError('inblance sampler not implemented')
            elif cfg.DATALOADER.SAMPLER =='uniform':
                
                train_loader = DataLoader(train_ds, batch_size = batch_size, num_workers=num_workers,shuffle = True, drop_last=True)    
            elif cfg.DATALOADER.SAMPLER =='weighted_meta':
                train_loader = DataLoader(train_ds, batch_size = batch_size, \
                                          sampler=WeightedRandomSampler(weights= 1.0/train_ds.w_im, num_samples=len(train_ds), replacement=True),\
                                          num_workers=num_workers, drop_last=True)    
            else:
                raise ValueError('Unknown Sampler {cfg.DATALOADER.SAMPLER}')
            train_loaders.append(train_loader)
            
            
        valid_loader = DataLoader(valid_ds, batch_size = batch_size, num_workers=num_workers, shuffle=False)
        
        val_loaders.append(valid_loader)
    
    return train_loaders, val_loaders



def make_data_loader_test(cfg):
    # as tta is needed, here we return dataset, not dataloader
    dataseto = init_dataset(cfg)
    
    
    # augmentation 
    if not isinstance(dataseto, list):
        train_transform = build_transforms(cfg, is_train=True,n_aug = cfg.MISC.N_TTA)
        val_transform   = build_transforms(cfg, is_train=False)
        
        if cfg.MISC.TTA is True:
        
            dataseto.transform = train_transform
        else:
            dataseto.transform = val_transform
            
        # resampler
        if cfg.MISC.ONLY_TEST is False:
            _, valid_ds_all = resample_idx_with_meta(dataseto,dataseto, K_fold = cfg.DATASETS.K_FOLD, pct = cfg.DATASETS.PCT)
        else:
            valid_ds_all = [dataseto]
            #cfg.DATASETS.K_FOLD = 1
      
    else:
        #already return train and valid dataset (list and include transform)
        valid_ds_all = [dataseto[1]] #TODO. test phase not implmented
        
        if cfg.MISC.TTA is True:
            valid_ds_all[0].transform = dataseto[0].transform
        
        cfg.DATASETS.K_FOLD = 1
    




    val_dss   = list()
    fns_kfds      = list()
    for nf in range(cfg.DATASETS.K_FOLD):
        if len(valid_ds_all)==1:
            valid_ds = valid_ds_all[0]
            if cfg.DATASETS.NAMES == 'ISIC': # random split
                fns_kfd = np.array(dataseto.fname)
            else:
                fns_kfd = np.array(dataseto[1].fname)

        else:
            valid_ds = valid_ds_all[nf]
            fns_kfd = np.array(dataseto.fname)[valid_ds.indices]


        
        
        val_dss.append(valid_ds)
    
        fns_kfds.append(fns_kfd)
        
        
    return val_dss,fns_kfds






def make_data_loader_reidtest(cfg):
    # as tta is needed, here we return dataset, not dataloader
    dataseto = init_dataset(cfg)
    
    
    # augmentation 
    if not isinstance(dataseto, list):
        #not inplementered
        #pass
        
    
        train_transform = build_transforms(cfg, is_train=True)
        val_transform   = build_transforms(cfg, is_train=False)
        
        if cfg.MISC.TTA is True:
        
            dataseto.transform = train_transform
        else:
            dataseto.transform = val_transform
            
        # resampler
        if cfg.MISC.ONLY_TEST is False:
            _, valid_ds_all = resample_idx_with_meta(dataseto,dataseto, K_fold = cfg.DATASETS.K_FOLD, pct = cfg.DATASETS.PCT)
        else:
            valid_ds_all = [dataseto]
            #cfg.DATASETS.K_FOLD = 1
      
    else:
        #already return train and valid dataset (list and include transform)
        
        train_ds_all = [dataseto[0]] 
        valid_ds_all = [dataseto[1]] #TODO. test phase not implmented
        
        
        
        if cfg.MISC.TTA is True:
            valid_ds_all[0].transform = dataseto[0].transform
        else:
            #need dist calc of training and testing
            train_ds_all[0].transform = dataseto[1].transform
        
        cfg.DATASETS.K_FOLD = 1
    


    train_dss = list()
    val_dss   = list()
    
    fns_kfds_train = list()
    fns_kfds_valid = list()
    
    for nf in range(cfg.DATASETS.K_FOLD):
        if len(valid_ds_all)==1:
            train_ds = train_ds_all[0]
            valid_ds = valid_ds_all[0]
            if cfg.DATASETS.NAMES == 'ISIC': # random split
                fns_kfd_train = np.array(dataseto.fname)
                fns_kfd_valid = np.array(dataseto.fname)
            else:
                fns_kfd_train = np.array(dataseto[0].fname)
                fns_kfd_valid = np.array(dataseto[1].fname)

        else:
            
            train_ds = train_ds_all[nf]
            valid_ds = valid_ds_all[nf]
            fns_kfd_train = np.array(dataseto.fname)[train_ds.indices]
            fns_kfd_valid = np.array(dataseto.fname)[valid_ds.indices]


        
        train_dss.append(train_ds)
        val_dss.append(valid_ds)
        fns_kfds_train.append(fns_kfd_train)
        fns_kfds_valid.append(fns_kfd_valid)
        
        
    return train_dss,val_dss,fns_kfds_train,fns_kfds_valid