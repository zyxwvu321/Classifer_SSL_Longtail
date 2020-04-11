import torch
import torch.nn as nn
import warnings
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import SGD, Adam

AVAI_OPTIMS = ['adam','sgd']
AVAI_SCHEDULERS = ['onecycle']
def find_modules(m, cond):
    if cond(m): return [m]
    return sum([find_modules(o,cond) for o in m.children()], [])

def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return isinstance(l, lin_layers)

def is_bn_layer(l):
    bn_layers = (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)
    return isinstance(l, bn_layers)

def build_optimizer(cfg,model, num_stps = None):
       
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module): model.
        optim (str, optional): optimizer. Default is "adam".
        lr (float, optional): learning rate. Default is 0.0003.
        weight_decay (float, optional): weight decay (L2 penalty). Default is 5e-04.
        momentum (float, optional): momentum factor in sgd. Default is 0.9,
        staged_lr (bool, optional): uses different learning rates for base and new layers. Base
            layers are pretrained layers while new layers are randomly initialized, e.g. the
            identity classification layer. Enabling ``staged_lr`` can allow the base layers to
            be trained with a smaller learning rate determined by ``base_lr_mult``, while the new
            layers will take the ``lr``. Default is False.
        backbone_layers (str or list): attribute names in ``model``. 
        base_lr_mult (float, optional): learning rate multiplier for base layers. Default is 0.1.

    """


    optim = cfg.SOLVER.OPTIMIZER
    if optim.lower() not in AVAI_OPTIMS:
        raise ValueError('Unsupported optim: {}. Must be one of {}'.format(optim, AVAI_OPTIMS))
    
    schedule = cfg.SOLVER.SCHEDULER
    if schedule.lower() not in AVAI_SCHEDULERS:
        raise ValueError('Unsupported scheduler: {}. Must be one of {}'.format(schedule, AVAI_SCHEDULERS))   

    if not isinstance(model, nn.Module):
        raise TypeError('model given to build_optimizer must be an instance of nn.Module')
        
    lr = cfg.SOLVER.BASE_LR
    staged_lr = cfg.SOLVER.STAGED_LR
    backbone_layers = cfg.SOLVER.BACKBONE_NAMES 
    base_lr_mult = cfg.SOLVER.BASE_MULT
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    momentum     = cfg.SOLVER.MOMENTUM

    # if stage_lr
    if staged_lr is True:
        if isinstance(backbone_layers, str):
            if backbone_layers is None:
                warnings.warn('new_layers is empty, therefore, staged_lr is useless')
            backbone_layers = [backbone_layers]
        
        base_params = []
        new_params = []
        
        for name, module in model.named_children():
            if name not in backbone_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
        
        param_groups = [
            {'params': new_params},
            {'params': base_params, 'lr': lr * base_lr_mult}  
        ]
        lrs = [lr, lr * base_lr_mult]
    else:
        param_groups = model.parameters()
        lrs = lr

    #build optimizer
    if optim.lower() == 'adam':
        optimizer = torch.optim.Adam(param_groups,lr=lr,weight_decay=weight_decay)

    elif optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(param_groups,lr=lr,momentum=momentum,weight_decay=weight_decay)

    else:
        raise ValueError(f'unknown optimizer: {optim}')

    # LR scheduler

    if schedule.lower() == 'onecycle':
        if optim.lower() == 'sgd':
            scheduler = OneCycleLR(optimizer, max_lr = lrs, epochs=cfg.SOLVER.EPOCHS, steps_per_epoch=num_stps)
        else:
            scheduler = OneCycleLR(optimizer, max_lr = lrs, epochs=cfg.SOLVER.EPOCHS, steps_per_epoch=num_stps, cycle_momentum=False)       
    else:
        raise ValueError(f'unknown scheduler: {schedule}')     

    return optimizer, scheduler