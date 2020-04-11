#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:35:04 2020

@author: minjie
"""

import torch
def init_pretrain(model,fname):
    """Initializes model with pretrained weights.
    
    
    """

    model_dict = model.state_dict()
    pretrain_dict = torch.load(fname)['state_dict']
    
    
    
    pretrain_dict = {k.replace('module.',''): v for k, v in pretrain_dict.items() if k.replace('module.','') in model_dict and model_dict[k.replace('module.','')].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)