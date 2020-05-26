# encoding: utf-8
"""
loss layer definition
"""
import torch
import torch.nn as nn
import globalvar as gl
import numpy as np
#import torch.nn.functional as F

from .class_loss import FocalLoss,CrossEntropyLoss_labelsmooth,Att_CELoss,PCsoftmaxCE,REID_CELoss,FixMatch_CELoss
#from modeling.diceloss import DiceLoss


#from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth,CosEmbedLoss,FocalLossLabelSmooth,FocalLossLabelSigmoid
#from .center_loss import CenterLoss

#from .reanked_loss import RankedLoss
#from .arcface_loss import ArcFace,Am_softmax
#_(self, in_features, out_features, device_id=[0], s = 64.0, m = 0.50, easy_margin = False, ls_eps=0.0)


def make_loss(cfg):   

    cfg = gl.get_value('cfg')
    if not isinstance(cfg.DATASETS.LABEL_W, int):
        label_w = np.array(cfg.DATASETS.LABEL_W).astype('float32')
        label_w = torch.tensor(label_w).float()
    else:
        label_w = None
        
    n_class = cfg.DATASETS.NUM_CLASS

    if cfg.MODEL.LOSS_TYPE  =='ce':
        if cfg.MODEL.SSL_FIXMATCH is True:
            criterion = nn.CrossEntropyLoss(weight =label_w,ignore_index=-1,reduction = 'none')
        elif cfg.DATALOADER.SAMPLER =='imbalance' or 'SVBNN' in cfg.MODEL.NAME:
            criterion = nn.CrossEntropyLoss()
        elif cfg.DATALOADER.SAMPLER  in  ['uniform','weighted_meta']:
            criterion = nn.CrossEntropyLoss(weight =label_w)
        else:
            raise ValueError('Unknown Sampler {cfg.DATALOADER.SAMPLER}')
            
    elif cfg.MODEL.LOSS_TYPE  =='pcs':
        criterion = PCsoftmaxCE(n_class,label_w)
        
    elif cfg.MODEL.LOSS_TYPE =='focalloss':
        if cfg.MODEL.SSL_FIXMATCH is True:
            pass
            #criterion = nn.CrossEntropyLoss(weight =label_w,ignore_index=-1,reduction = 'none')
        
        elif 'SVBNN' in cfg.MODEL.NAME:
            criterion = FocalLoss(num_classes= n_class)
        else:
            
            criterion = FocalLoss(num_classes= n_class ,weight = label_w)
            
    elif cfg.MODEL.LOSS_TYPE =='ce_smooth':
        criterion = CrossEntropyLoss_labelsmooth(num_classes = n_class, label_w = label_w)
        
    elif  cfg.MODEL.LOSS_TYPE =='bce':
        criterion = nn.BCEWithLogitsLoss()
        
        

    else:
        raise ValueError(f"Unknown crit {cfg.MODEL.LOSS_TYPE}")  

    use_cuda = torch.cuda.is_available()
    if cfg.MODEL.DEVICE != 'cuda':
        use_cuda = False

    device = torch.device('cuda' if use_cuda else 'cpu')
    criterion = criterion.to(device)
    
    
    if cfg.MODEL.LOSS_ATT is True:
        criterion = Att_CELoss(criterion,num_classes= n_class )
        
    elif'reid' in cfg.MODEL.NAME: 
        k_center = cfg.MODEL.REID_KCENTER 
        criterion = REID_CELoss(criterion,num_classes= n_class,k_center = k_center )
        
    elif cfg.MODEL.SSL_FIXMATCH is True:
        criterion = FixMatch_CELoss(criterion,num_classes= n_class,pseudo_th = cfg.MODEL.PSEUDO_TH )
            
    return criterion
    
    


    """ sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func



def make_loss_cls(cfg):    # predict dry and wet
    
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=2) # ignore_index=-100 
        print("label smooth on, 2 numclasses:")
    else:
        xent = nn.CrossEntropyLoss() # ignore_index=-100 
        print("CE on, 2 numclasses:")
    def loss_func(score, target):
        return xent(score, target)
    return loss_func
 """








""" ef make_loss_with_center(cfg, num_classes):    # modified by gu
    k_softmax = cfg.SOLVER.LOSSK_SOFTMAX
    k_center = cfg.SOLVER.LOSSK_CENTER
    k_triple = cfg.SOLVER.LOSSK_TRIPLE
    k_rll = cfg.SOLVER.LOSSK_RLL
    k_cls = cfg.SOLVER.LOSSK_CLS



    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    elif cfg.MODEL.NAME == 'mobilenetv2' or cfg.MODEL.NAME == 'eff1' :
        feat_dim = 1280
    elif cfg.MODEL.NAME == 'eff3':
        feat_dim = 1536
    elif cfg.MODEL.NAME == 'mobilenetv3'or cfg.MODEL.NAME == 'mobilenetv3ch1':
        feat_dim = 960
    else:
        feat_dim = 2048
    
    if cfg.MODEL.USE_FC in ['yes','on']:
        feat_dim = 256
    
    
    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center'  and cfg.MODEL.IF_COSEMBED in ['no','off']:
        triplet = TripletLoss(cfg.SOLVER.MARGIN,p_drop = cfg.SOLVER.TRIPLE_PDROP)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
#        ranked_loss = RankedLoss(0.4,1.2,10.0)
    
    elif cfg.MODEL.IF_COSEMBED in ['yes','on']:
        triplet = TripletLoss(cfg.SOLVER.MARGIN,p_drop = cfg.SOLVER.TRIPLE_PDROP)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
        #cosembed   =  CosEmbedLoss()
        #ranked_loss = RankedLoss(1.3,2.0,1.0) # ranked_loss
        ranked_loss = RankedLoss(0.6,1.414,1.0,p_drop = cfg.SOLVER.RLL_PDROP) # ranked_loss
    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))



    if cfg.MODEL.IF_LABELSMOOTH in  ['on','yes'] and cfg.MODEL.IF_FOCAL_LOSS in  ['off','no']:# and cfg.MODEL.IF_ARCFACE in  ['off','no']:
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
        if cfg.MODEL.PRED_CLS in  ['on','yes']:
            xent_cls = CrossEntropyLabelSmooth(num_classes=3, pred_stat = True)     # new add by luo
        
        
##    elif cfg.MODEL.IF_LABELSMOOTH in  ['on','yes'] and cfg.MODEL.IF_ARCFACE in  ['yes','on']:
##        #xent = ArcFaceLoss(feat_dim,num_classes,ls_eps=0.1)     # new add by luo
##        xent = Am_softmax(feat_dim,num_classes)     # new add by luo
##        
#        
#        print("label smooth on arcface, numclasses:", num_classes)
    elif cfg.MODEL.IF_FOCAL_LOSS in  ['on','yes']:
        xent = FocalLossLabelSigmoid(num_classes=num_classes)    
        print("focal loss label smooth on, numclasses:", num_classes)
    
    k_softmax = cfg.SOLVER.LOSSK_SOFTMAX
    k_center = cfg.SOLVER.LOSSK_CENTER
    k_triple = cfg.SOLVER.LOSSK_TRIPLE
    k_rll = cfg.SOLVER.LOSSK_RLL
    k_cls = cfg.SOLVER.LOSSK_CLS
     
    def loss_func_n(score, feat, target,feat_afterbn,cids):
     
        #return  xent(score, target) + cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + 0.4*ranked_loss(feat_afterbn, target) #cosembed(feat_afterbn, target,cids) 
        #return  xent(score, target) + 0.00*cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + ranked_loss(feat_afterbn, target) #cosembed(feat_afterbn, target,cids) 
               
        return k_softmax*xent(score, target) + \
            k_triple*triplet(feat, target)[0] + \
            k_center*cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + k_rll*ranked_loss(feat_afterbn, target) 
    

    def loss_func_withclasspred(score, score_cls, feat, target,cids):
        return k_softmax*xent(score, target) + k_cls*xent_cls(score_cls, cids)  + \
                        k_triple*triplet(feat, target)[0] + \
                        k_center*cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) 
                      
    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH  in  ['on','yes']:
                return k_softmax*xent(score, target) + \
                        k_center*cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return k_softmax*F.cross_entropy(score, target) + \
                        k_center*cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH in  ['on','yes']:
             
                
                return k_softmax*xent(score, target) + \
                        k_triple*triplet(feat, target)[0] + \
                        k_center*cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) 

#                return xent(score, target) + \
#                        0.00000*cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)                      
            else:
                return k_softmax*F.cross_entropy(score, target) + \
                        k_triple*triplet(feat, target)[0] + \
                        k_center*cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))        
            
    
    if cfg.MODEL.IF_COSEMBED in ['yes','on']:

        return loss_func_n, center_criterion
    if  cfg.MODEL.PRED_CLS in  ['on','yes']:
        return loss_func_withclasspred, center_criterion
    return loss_func, center_criterion """