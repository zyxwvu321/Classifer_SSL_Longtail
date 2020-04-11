# -*- coding: utf-8 -*-
# Mixup and cutmix

import numpy as np
import torch
import globalvar as gl

__all__ = ['mixup_data', 'mixup_criterion']


@torch.no_grad()
def mixup_data(x, y, alpha=0.2,prob =1.0):
    """Returns mixed inputs, pairs of targets, and lambda
    """
    r = np.random.rand(1)
    if alpha > 0  and r < prob:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0,))
    y_a, y_b = y, y.flip(dims=(0,))
    
    #rand_index = torch.randperm(x.size()[0]).cuda()
    #mixed_x = lam * x + (1 - lam) * x[rand_index]
    #y_a,y_b = y, y[rand_index]
    
    
    return mixed_x, y_a, y_b, lam


@torch.no_grad()
def cutmix_data(x, y, alpha=0.2,prob =1.0):
    
    
    """Returns mixed inputs, pairs of targets, and lambda
    """
    

    r = np.random.rand(1)
    if alpha > 0 and r < prob:
        # generate mixed sample
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(x.size()[0]).cuda()
        y_a = y
        y_b = y[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
#        # compute output
#        output = model(input)
#        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        return x, y_a, y_b, lam
    else:
        lam = 1.0
        return x ,y,y,lam
            
            
            
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    
    loss1 = criterion(pred, y_a)
    loss2 = criterion(pred, y_b)
    #cfg = gl.get_value('cfg')
    
    if isinstance(loss1,(tuple,list)):
        if lam !=1.0:

            loss1_a = loss1[1]['global']
            loss2_a = loss2[1]['global'] 
            
            loss_total = lam * loss1_a + (1 - lam) * loss2_a
            
            loss_dict = dict()
            for key in loss1[1].keys():
                loss_dict[key] = lam * loss1[1][key] + (1 - lam) * loss2[1][key]
            
            return (loss_total, loss_dict)
        else:
            
            return loss1
            
        
            
    else:
        return lam * loss1 + (1 - lam) * loss2







def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2