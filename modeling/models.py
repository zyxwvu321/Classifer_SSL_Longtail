
"""
Defines architectures for breast cancer classification models. 
"""

import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
from utils.utils import get_device
#import torchvision
import modeling.layers as layers
import torchvision.models as models

import torch.nn.functional as F

from modeling.backbones.pytorchinsight import sk_resnet50
from modeling.backbones.pytorchinsight import sge_resnet50
from modeling.backbones.pytorchinsight import old_resnet50 as resnet50d


from models_reid.senet import se_resnet50,se_resnet101,se_resnext50_32x4d,se_resnext101_32x4d
#from configs import effnetb4_singleview,effnetb4_metasingleview
from efficientnet_pytorch import EfficientNet


from .backbones.ResNeSt import resnest50
from .backbones.bit_pytorch import ResNetV2 as resnet_bit


from modeling.cbam import CBAM


#import collections as col

from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .wsdan.resnet  import resnet50
from .wsdan.attention import calculate_pooling_center_loss, attention_crop_drop2,mask2bbox
from .wsdan.utils import  batch_augment

from  .adl.resnet import resnet50 as resnet50_adl

from .blocks import DBLayer,BasicConv2d,BAP,FCNorm
from .blocks import Bottleneck as resblock



import modeling.wsdan.resnet as resnet


#from constants import VIEWS

from utils.utils import init_cnn
import globalvar as gl
class ISICModel_singleview(nn.Module):
    def __init__(self,n_class,arch,use_CBAM = False):
        super(ISICModel_singleview, self).__init__()
        self.mode = 'singleview'
        
        cfg = gl.get_value('cfg')
        self.cfg  = cfg
        if arch == 'resnet50': 
            
            
            if cfg.MODEL.USE_ADL  is True:
                model_backbone = resnet50_adl(pretrained = True,num_classes=n_class,
                                                     ADL_position=cfg.MODEL.ADL_POSITION,drop_rate=cfg.MODEL.ADLRATE,drop_thr=cfg.MODEL.ADLTHR)
            else:
                model_backbone = models.resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
            
        elif arch == 'sk_resnet50':
            model_backbone = sk_resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
        elif arch == 'resnet50d':
            model_backbone = resnet50d(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )    
            
        elif arch == 'sge_resnet50':
            model_backbone = sge_resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch =='resnext50_32x4d':

            model_backbone = models.resnext50_32x4d(pretrained = True)
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip

        
                 
        elif arch == 'effnetb4':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b4')      
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
        elif arch == 'effnetb3':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b3')       #(3072,512)
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch == 'effnetb5':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b5')       #(4096,512)
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch == 'effnetb2':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b2')       #(2816,512)
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch == 'resnest50':
            model_backbone = resnest50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip      
            
        elif arch =='resnetbit':
            model_backbone = resnet_bit(pretrained = True, block_units=[3, 4, 6, 3], width_factor=1,model_type = 'resnet50')
            model_backbone.head = None
            self.backbone = model_backbone
            self.backbone_lc = nn.ReLU(inplace=True) #skip       
        elif arch =='resnet101bit':
            model_backbone = resnet_bit(pretrained = True, block_units=[3, 4, 23, 3], width_factor=1,model_type = 'resnet101')
            model_backbone.head = None
            self.backbone = model_backbone
            self.backbone_lc = nn.ReLU(inplace=True) #skip 
            
        self.imfeat_dim = cfg.MODEL.IMG_FCS  #(4096,512)
        if use_CBAM is False and cfg.MODEL.USE_ADL  is False:
            
            if cfg.MODEL.POOLMODE == 'maxavg':
                self.head_im = nn.Sequential(layers.MaxAvgPool(),
                                  nn.BatchNorm1d(self.imfeat_dim[0],eps = 1e-5,momentum=0.1,affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(in_features=self.imfeat_dim[0],out_features =self.imfeat_dim[1],bias=False),
                                  nn.BatchNorm1d(self.imfeat_dim[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(in_features = self.imfeat_dim[1],out_features =n_class,bias=True))
            elif cfg.MODEL.POOLMODE=='avg':
                self.head_im = nn.Sequential(layers.AvgPool(),
                                  nn.BatchNorm1d(self.imfeat_dim[0]//2,eps = 1e-5,momentum=0.1,affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(in_features=self.imfeat_dim[0]//2,out_features =self.imfeat_dim[1],bias=False),
                                  nn.BatchNorm1d(self.imfeat_dim[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(in_features = self.imfeat_dim[1],out_features =n_class,bias=True))
            
            
            
        elif cfg.MODEL.USE_ADL  is True:
            
            self.head_im = nn.Sequential(layers.AvgPool(),
                                  nn.BatchNorm1d(self.imfeat_dim[0]//2,eps = 1e-5,momentum=0.1,affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(in_features=self.imfeat_dim[0]//2,out_features =self.imfeat_dim[1],bias=False),
                                  nn.BatchNorm1d(self.imfeat_dim[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(in_features = self.imfeat_dim[1],out_features =n_class,bias=True))
        else:
            self.head_im = nn.Sequential(CBAM(gate_channels =self.imfeat_dim[0]//2),
                                  layers.AvgPool(),
                                  nn.BatchNorm1d(self.imfeat_dim[0]//2,eps = 1e-5,momentum=0.1,affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(in_features=self.imfeat_dim[0]//2,out_features =self.imfeat_dim[1],bias=False),
                                  nn.BatchNorm1d(self.imfeat_dim[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(in_features = self.imfeat_dim[1],out_features =n_class,bias=True))
        self.meta_fc = nn.ReLU(inplace=True)
        self.final_conv = nn.ReLU(inplace=True) #skip 

        #if cfg.MODEL.BACKBONE_PRETRAIN_PATH is not None  and os.path.exists(cfg.MODEL.BACKBONE_PRETRAIN_PATH):
        #    self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE_PRETRAIN_PATH))

        self.init()
        if cfg.MODEL.PRETRAIN_PATH is not None  and os.path.exists(cfg.MODEL.PRETRAIN_PATH):
            self.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))


    def forward(self, x,x_lc = None):

        result_backbone = self.backbone(x)        
        
        result = self.head_im(result_backbone)

        return result
    
    def init(self):
        #pass
        #self.two_view_resnet.apply(_kaiming_init_)
        #self.head.apply(_kaiming_init_)
        init_cnn(self.head_im)
        init_cnn(self.head_im[-1],negative_slope = 1.0)



class ISICModel_singleview_meta(nn.Module):
    def __init__(self,n_class,arch,use_CBAM = False):
        super(ISICModel_singleview_meta, self).__init__()
        self.mode = 'metasingleview'
        cfg = gl.get_value('cfg')
        
        self.use_heatmap = cfg.MISC.CALC_HEATMAP
        
        if arch == 'resnet50':

            model_backbone = models.resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch == 'sk_resnet50':
            model_backbone = sk_resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
        elif arch =='resnext50_32x4d':

            model_backbone = models.resnext50_32x4d(pretrained = True)
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip

    
        elif arch == 'effnetb4':

            model_backbone = EfficientNet.from_pretrained('efficientnet-b4')  #(3584,512)
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch == 'effnetb3':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b3')       #(3072,512)
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
   
        elif arch == 'effnetb5':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b5')       #(4096,512)
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch == 'effnetb2':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b2')       #(2816,512)
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch == 'resnest50':
            model_backbone = resnest50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        
        elif arch =='resnetbit':
            model_backbone = resnet_bit(pretrained = True, block_units=[3, 4, 6, 3], width_factor=1,model_type = 'resnet50')
            model_backbone.head = None
            self.backbone = model_backbone
            self.backbone_lc = nn.ReLU(inplace=True) #skip       
        elif arch =='resnet101bit':
            model_backbone = resnet_bit(pretrained = True, block_units=[3, 4, 23, 3], width_factor=1,model_type = 'resnet101')
            model_backbone.head = None
            self.backbone = model_backbone
            self.backbone_lc = nn.ReLU(inplace=True) #skip         
        self.imfeat_dim = cfg.MODEL.IMG_FCS  #(4096,512)
        self.meta_dim = sum(cfg.MODEL.META_DIMS) #(29)
        self.meta_dim_fc = cfg.MODEL.META_FCS  #(64,128)
        self.final_dim  = cfg.MODEL.FINAL_DIM # (320)

        if use_CBAM is False:
            self.head_im = nn.Sequential(layers.MaxAvgPool(),
                                    nn.BatchNorm1d(self.imfeat_dim[0],eps = 1e-5,momentum=0.1,affine=True, track_running_stats=True),
                                    nn.Dropout(p=0.25),
                                    nn.Linear(in_features=self.imfeat_dim[0],out_features =self.imfeat_dim[1],bias=False),
                                    nn.BatchNorm1d(self.imfeat_dim[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.5))
                                    
        else:
            self.head_im = nn.Sequential(CBAM(gate_channels =self.imfeat_dim[0]//2),
                        layers.AvgPool(),
                        nn.BatchNorm1d(self.imfeat_dim[0]//2,eps = 1e-5,momentum=0.1,affine=True, track_running_stats=True),
                        nn.Dropout(p=0.25),
                        nn.Linear(in_features=self.imfeat_dim[0]//2,out_features =self.imfeat_dim[1],bias=False),
                        nn.BatchNorm1d(self.imfeat_dim[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))


        self.meta_fc = nn.Sequential(nn.BatchNorm1d(self.meta_dim,momentum=0.1, affine=True, track_running_stats=True),
                                     nn.Linear(in_features = self.meta_dim,out_features = self.meta_dim_fc[0],bias=False),
                                     nn.BatchNorm1d(self.meta_dim_fc[0],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.25),

                                     nn.Linear(in_features = self.meta_dim_fc[0],out_features = self.meta_dim_fc[1],bias=False),
                                     nn.BatchNorm1d(self.meta_dim_fc[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5))
        
        
        self.final_conv = nn.Sequential(nn.Linear(in_features = self.meta_dim_fc[1] +self.imfeat_dim[1] , out_features =self.final_dim,bias=False),
                                     nn.BatchNorm1d(self.final_dim,eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(in_features = self.final_dim,out_features =n_class,bias=True))

        self.init()
        if cfg.MODEL.PRETRAIN_PATH is not None  and os.path.exists(cfg.MODEL.PRETRAIN_PATH):
            self.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))


    def forward(self, x,meta_info = None):
        # if in training n_aug>1: transform
        if x.dim()==5:
            n_aug = x.size(1)
            x = x.reshape(x.size(0)*x.size(1), *x.size()[2:])
        else:
            n_aug = 1
        
        
        result_backbone = self.backbone(x)
        result_imfeat = self.head_im(result_backbone)
        
        if self.use_heatmap is True:
                # register the hook
            h_IMG = result_backbone.register_hook(self.activations_hook_IMG)
            #h_META = result_metafeat.register_hook(self.activations_hook_META)
        
        

        if meta_info is None:
            meta_info = torch.zeros((result_imfeat.size(0),self.meta_dim)).type_as(x)
        elif n_aug>= 1:
            if meta_info.dim()==3:
                meta_info = meta_info.reshape(meta_info.size(0)*meta_info.size(1), *meta_info.size()[2:])
            else:
                meta_info = meta_info.repeat(1,n_aug).reshape(result_imfeat.size(0),-1)
        
        
        result_metafeat = self.meta_fc(meta_info)

        result_immeta = torch.cat((result_imfeat, result_metafeat),dim = 1)
        result = self.final_conv(result_immeta)

        if n_aug>1:
            result = result.reshape(-1,n_aug,result.size(1)).mean(dim=1)

        return result
    
    def init(self):
        #pass
        #self.two_view_resnet.apply(_kaiming_init_)
        #self.head.apply(_kaiming_init_)
        init_cnn(self.head_im)
        init_cnn(self.meta_fc)
        init_cnn(self.final_conv)
        init_cnn(self.final_conv[-1],negative_slope = 1.0)


    def activations_hook_IMG(self, grad):
        self.gradients_IMG = grad    
    def activations_hook_META(self, grad):
        self.gradients_META = grad    
        
    # method for the gradient extraction
    def get_activations_gradient_IMG(self):
        return self.gradients_IMG
    def get_activations_gradient_META(self):
        return self.gradients_META
    
    # method for the activation exctraction
    def get_activations_IMG(self, x):
        # if in training n_aug>1: transform
        if x.dim()==5:    
            x = x.reshape(x.size(0)*x.size(1), *x.size()[2:])
        
        result_backbone = self.backbone(x)
        
        return result_backbone
    
    
    
    def get_activations_META(self, x):
        pass


class SplitBoneModel_meta(nn.Module):
    def __init__(self,n_class,arch,use_CBAM = False):
        super(SplitBoneModel_meta, self).__init__()
       
        self.mode = 'twoviewmeta'
        self.cfg = gl.get_value('cfg')
        if arch =='resnet34':
            model_resnet34 = torchvision.models.resnet34(pretrained = True)
            model_resnet34 = (nn.Sequential(*list(model_resnet34.children())[:-2]) )

            self.backbone = model_resnet34

           
        elif arch =='resnet50':
        
            model_resnet50 = torchvision.models.resnet50(pretrained = True)
            model_resnet50 = (nn.Sequential(*list(model_resnet50.children())[:-2]) )
            self.backbone = model_resnet50


        elif arch =='effb3':
            # use same backbone
            model_effb3 = EfficientNet.from_pretrained('efficientnet-b3')
            self.backbone = model_effb3

        else:
            raise ValueError(f"arch = {arch} is not supported.")  
       
        
        self.imfeat_dim = self.cfg.MODEL.IMG_FCS  #(4096,512)
        self.meta_dim = sum(self.cfg.MODEL.META_DIMS) #(29)
        self.meta_dim_fc = self.cfg.MODEL.META_FCS  #(64,128)
        self.final_dim  = self.cfg.MODEL.FINAL_DIM # (320)


        
        if use_CBAM is False:
            self.head_im = nn.Sequential(layers.AllViewsMaxAvgPool(),
                                  nn.BatchNorm1d(self.imfeat_dim[0],eps = 1e-5,momentum=0.1,affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(in_features=self.imfeat_dim[0],out_features =self.imfeat_dim[1],bias=False),
                                  nn.BatchNorm1d(self.imfeat_dim[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=0.5))
        else:                       
            self.head_im = nn.Sequential(layers.AllViewsCBAMAvgPool(featdim = self.imfeat_dim[0]),
                                  nn.BatchNorm1d(self.imfeat_dim[0],eps = 1e-5,momentum=0.1,affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(in_features=self.imfeat_dim[0],out_features =self.imfeat_dim[1],bias=False),
                                  nn.BatchNorm1d(self.imfeat_dim[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=0.5))

        self.meta_fc = nn.Sequential(nn.BatchNorm1d(self.meta_dim,momentum=0.1, affine=True, track_running_stats=True),
                                     nn.Linear(in_features = self.meta_dim,out_features = self.meta_dim_fc[0],bias=False),
                                     nn.BatchNorm1d(self.meta_dim_fc[0],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.25),

                                     nn.Linear(in_features = self.meta_dim_fc[0],out_features = self.meta_dim_fc[1],bias=False),
                                     nn.BatchNorm1d(self.meta_dim_fc[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5))
        
        
        self.final_conv = nn.Sequential(nn.Linear(in_features = self.meta_dim_fc[1] +self.imfeat_dim[1] , out_features =self.final_dim,bias=False),
                                     nn.BatchNorm1d(self.final_dim,eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(in_features = self.final_dim,out_features =self.n_class,bias=True))
        self.init()
        
#       # hook for the gradients of the activations
#    def activations_hook_AP(self, grad):
#        self.gradients_AP = grad    
#    def activations_hook_LAT(self, grad):
#        self.gradients_LAT = grad    
#        
#        
#    # method for the gradient extraction
#    def get_activations_gradient_AP(self):
#        return self.gradients_AP
#    def get_activations_gradient_LAT(self):
#        return self.gradients_LAT
#    
#    # method for the activation exctraction
#    def get_activations_AP(self, x):
#        return self.two_view_resnet(x)['AP']
#    def get_activations_LAT(self, x):
#        return self.two_view_resnet(x)['LAT']
        

    def forward(self, x,meta_info=None):
        #h = self.all_views_gaussian_noise_layer(x)
        result_backbone = self.two_view_resnet(x)
        
        imgs = torch.chunk(x,self.cfg.MODEL.N_VIEW,dim=1)
        result_backbone = dict()
        for view_name, img in zip(self.cfg.MODEL.VIEW_NAME, imgs):
            result_backbone[view_name]  = self.backbone(img)
        
            
        
#        if self.use_heatmap:
#                # register the hook
#            h_AP = result_backbone['AP'].register_hook(self.activations_hook_AP)
#            h_LAT = result_backbone['LAT'].register_hook(self.activations_hook_LAT)
        
        result_imfeat = self.head_im(result_backbone)
        
        if meta_info is  None:
            meta_info = torch.zeros((result_imfeat.size(0),self.meta_dim)).type_as(x)
            

        result_metafeat = self.meta_fc(meta_info)

        result_immeta = torch.cat((result_imfeat, result_metafeat ),dim = 1)

        result = self.final_conv(result_immeta)

        return result    
    

    
    
    
    def init(self):
        #pass
        #self.two_view_resnet.apply(_kaiming_init_)
        #self.head.apply(_kaiming_init_)
        init_cnn(self.head_im)
        init_cnn(self.meta_fc)
        init_cnn(self.final_conv)
        init_cnn(self.final_conv[-1],negative_slope = 1.0)
        
        #self.fc2.apply(_kaiming_init_)















class ISICModel_sv_att(nn.Module):
    def __init__(self,n_class,arch):
        super(ISICModel_sv_att, self).__init__()
        self.mode = 'sv_att'
        
        self.cfg = gl.get_value('cfg')
        self.M = self.cfg.MODEL.ATT_PARTS
        
        if arch in ['resnet50']:
            self.backbone =  getattr(resnet, arch)(pretrained=True).get_features()
            self.feat_dim = 512 * self.backbone[-1][-1].expansion
        
        elif arch == 'sk_resnet50':
            model_backbone = sk_resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            self.feat_dim = 2048
        
        
        
        self.center_feat = torch.zeros(n_class,self.feat_dim * self.M )
        
        

        device = get_device(self.cfg)
        self.center_feat = self.center_feat.to(device)
        

        # Attention Maps
        self.attentions = BasicConv2d(self.feat_dim, self.cfg.MODEL.ATT_PARTS, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.head_im = nn.Linear(self.feat_dim*self.M, n_class, bias=False)
        

        self.init()
        if self.cfg.MODEL.PRETRAIN_PATH is not None  and os.path.exists(self.cfg.MODEL.PRETRAIN_PATH):
            self.load_state_dict(torch.load(self.cfg.MODEL.PRETRAIN_PATH))
            


    def forward(self, x,target,lam = 1.0):


        # raw images forward
        y_pred_raw, feature_matrix, attention_map  = self.raw_predict(x)

        
        
        if self.training:
        
            # Update Feature Center
            feature_center_batch = F.normalize(self.center_feat[target], dim=-1)
            
            if lam ==1.0:
                self.center_feat[target] += self.cfg.SOLVER.ALPHA_CENTER * (feature_matrix.detach() - feature_center_batch)
    
            ##################################
            # Attention Cropping
            ##################################
            with torch.no_grad():
                crop_images = batch_augment(x, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
    
            ##################################
            # Attention Dropping
            ##################################
            with torch.no_grad():
                drop_images = batch_augment(x, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
    
    
            # crop images forward
            y_pred_crop, _, _ = self.raw_predict(crop_images)
    
    
            # drop images forward
            y_pred_drop, _, _ = self.raw_predict(drop_images)
            
            
            return    y_pred_raw,   y_pred_crop, y_pred_drop, feature_matrix,feature_center_batch
        else:
            ##################################
            # Object Localization and Refinement
            ##################################
            crop_images = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _, _ = self.raw_predict(crop_images)
            
            return    y_pred_raw,   y_pred_crop
    


    def raw_predict(self,x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.backbone(x)
        
        attention_maps = self.attentions(feature_maps)
        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification
        y_pred_raw = self.head_im(feature_matrix * 100.0)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() +  1e-12)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)
        
        return y_pred_raw, feature_matrix, attention_map
        
    def init(self):
        #pass
        #self.two_view_resnet.apply(_kaiming_init_)
        #self.head.apply(_kaiming_init_)
        init_cnn(self.head_im,negative_slope = 1.0)
        init_cnn(self.attentions)



class ISICModel_sv_db(nn.Module):
    def __init__(self,n_class,arch):
        super(ISICModel_sv_db, self).__init__()
        self.mode = 'sv_db'
        
        cfg = gl.get_value('cfg')
        
        if arch == 'resnet50': 

            model_backbone = models.resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        
        elif arch =='resnext50_32x4d':

            model_backbone = models.resnext50_32x4d(pretrained = True)
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip

       
                 
        elif arch == 'effnetb4':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b4')      
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            

        self.imfeat_dim = cfg.MODEL.IMG_FCS  #(4096,512)

  
        #self.head_im = nn.Conv2d(in_features = self.imfeat_dim[1],out_features =n_class,bias=False)

        #self.head_im = nn.Conv2d(in_channels = self.imfeat_dim[0]//2, out_channels = n_class,kernel_size = 1,stride=1, bias=True)
        
        self.head_im = nn.Sequential(
                      nn.BatchNorm2d(self.imfeat_dim[0]//2,eps = 1e-5,momentum=0.1,affine=True, track_running_stats=True),
                      nn.Dropout2d(p=0.25),
                      nn.Conv2d(in_channels=self.imfeat_dim[0]//2,out_channels =self.imfeat_dim[1],kernel_size = 1,bias=False),
                      nn.BatchNorm2d(self.imfeat_dim[1],eps = 1e-5,momentum=0.1, affine=True, track_running_stats=True),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(p=0.5),
                      nn.Conv2d(in_channels = self.imfeat_dim[1],out_channels =n_class,kernel_size = 1,bias=True))
        
        
        
        
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.db_block = DBLayer(p_peak= cfg.MODEL.DB_PEAKDROP, p_drop = cfg.MODEL.DB_PEAKDROP,patch_g = cfg.MODEL.DB_PATCHG,alpha = cfg.MODEL.DB_ALPHA)



        self.init()
        if cfg.MODEL.PRETRAIN_PATH is not None  and os.path.exists(cfg.MODEL.PRETRAIN_PATH):
            self.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))


    def forward(self, x,targets):

        result_backbone = self.backbone(x)        
        
        
        
        #result = self.head_im(result_backbone)

        if self.training is True:
            result = self.head_im(result_backbone)
            result = self.db_block(result,targets)
            result = self.gap(result).reshape(x.size(0),-1)
            
        else:
            
            result = self.head_im(result_backbone)
            result = self.gap(result).reshape(x.size(0),-1)
        
        
        


        return result
    
    def init(self):
        #pass
        #self.two_view_resnet.apply(_kaiming_init_)
        #self.head.apply(_kaiming_init_)
        init_cnn(self.head_im)
        init_cnn(self.head_im[-1],negative_slope = 1.0)
        
        
        




class ISICModel_singleview_reid(nn.Module):
    # test if bnnneck im reid etc. works
    def __init__(self,n_class,arch,use_CBAM = False):
        super(ISICModel_singleview_reid, self).__init__()
        self.mode = 'singleview_reid'
        
        cfg = gl.get_value('cfg')
        self.cfg  = cfg
        if arch == 'resnet50': 
            
            
            if cfg.MODEL.USE_ADL  is True:
                model_backbone = resnet50_adl(pretrained = True,num_classes=n_class,
                                                     ADL_position=cfg.MODEL.ADL_POSITION,drop_rate=cfg.MODEL.ADLRATE,drop_thr=cfg.MODEL.ADLTHR)
            else:
                model_backbone = models.resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
            
        elif arch == 'sk_resnet50':
            model_backbone = sk_resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
        elif arch == 'resnet50d':
            model_backbone = resnet50d(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
            
        elif arch == 'sge_resnet50':
            model_backbone = sge_resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch =='resnext50_32x4d':

            model_backbone = models.resnext50_32x4d(pretrained = True)
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip

        elif  arch =='se_resnext50':
    
            model_backbone = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=2) 
            param_dict = torch.load('../models/se_resnext50_32x4d-a260b3a4.pth')
     
            for i in param_dict:
                if 'classifier' in i  or 'last_linear' in i:
                    continue
                model_backbone.state_dict()[i].copy_(param_dict[i]) 
            
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
                 
        elif arch == 'effnetb4':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b4')      
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            

        self.imfeat_dim = cfg.MODEL.IMG_FCS  #(4096,512)
        
        self.use_fc = cfg.MODEL.REID_USE_FC
        
        
        self.num_classes = n_class
        self.pdrop_lin = cfg.MODEL.REID_PDROP_LIN
        self.neck_feat = cfg.MODEL.REID_NECK_FEAT
        
        
        
        
        
        if self.use_fc is True:
            self.in_planes = self.imfeat_dim[1]
            self.after_backbone = nn.Sequential(layers.AvgPool(),
                                         nn.Dropout(p=self.pdrop_lin),
                                         nn.Linear(self.imfeat_dim[0]//2,  self.in_planes, bias=False))
            self.bottleneck = nn.BatchNorm1d(self.imfeat_dim[1])
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        else:
            self.in_planes = self.imfeat_dim[0]//2
            self.after_backbone = layers.AvgPool()
            self.bottleneck = nn.BatchNorm1d(self.in_planes )
            self.classifier = nn.Sequential(nn.Dropout(p=self.pdrop_lin), nn.Linear(self.in_planes , self.num_classes, bias=False))
    
        self.bottleneck.bias.requires_grad_(False)  # no shift    
        
        self.center_feat = torch.zeros(n_class,self.in_planes)
        device = get_device(self.cfg)
        self.center_feat = self.center_feat.to(device)
        
        #self.head_im = nn.Sequential(self.after_backbone,self.bottleneck)

            
        init_cnn(self.after_backbone)
        init_cnn(self.bottleneck)
        init_cnn(self.classifier)
        
        
        self.meta_fc = nn.ReLU(inplace=True)
        self.final_conv = nn.ReLU(inplace=True) #skip 

        #if cfg.MODEL.BACKBONE_PRETRAIN_PATH is not None  and os.path.exists(cfg.MODEL.BACKBONE_PRETRAIN_PATH):
        #    self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE_PRETRAIN_PATH))


        if cfg.MODEL.PRETRAIN_PATH is not None  and os.path.exists(cfg.MODEL.PRETRAIN_PATH):
            self.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))


    def forward(self, x,target,lam = 1.0):

        result_backbone = self.backbone(x)        
        
        global_feat = self.after_backbone(result_backbone)
        feat = self.bottleneck(global_feat)
        cls_score = self.classifier(feat)
        
        
        
        if self.training:
            
            if self.cfg.MODEL.REID_CENTERFEATNORM is True:
                feature_center_batch = self.center_feat[target]
                if lam ==1.0:
                    
                    feature_center_batch = F.normalize(feature_center_batch, dim=-1)
                    feat = F.normalize(feat.detach(), dim=-1)
                    self.center_feat[target] += self.cfg.SOLVER.ALPHA_CENTER * (feat.detach() -  feature_center_batch)
                    
                    #self.center_feat[target] += self.cfg.SOLVER.ALPHA_CENTER * (global_feat.detach() -  feature_center_batch)
                    
                #return cls_score,global_feat,feature_center_batch
                return cls_score,feat,feature_center_batch
            else:
            
            
                feature_center_batch = self.center_feat[target]
                if lam ==1.0:
                    
                   self.center_feat[target] += self.cfg.SOLVER.ALPHA_CENTER * (global_feat.detach() -  feature_center_batch)
                    
                return cls_score,global_feat,feature_center_batch
                
        else:
            if self.neck_feat == 'after':
                return cls_score,feat
            #elif self.neck_feat == 'cls':
            #    return cls_score
            elif self.neck_feat == 'before':
                return cls_score,global_feat
            else: 
                raise ValueError('unknown valid feat mode')

class SV_BNN(nn.Module):
    def __init__(self,n_class,arch, cls_layer = 'FC'):
        super(SV_BNN, self).__init__()
        self.mode = 'SV_BNN'
        
        cfg = gl.get_value('cfg')
        self.cfg  = cfg
        if arch == 'resnet50': 
            model_backbone = models.resnet50(pretrained = True)
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
            
        elif arch == 'sk_resnet50':
            model_backbone = sk_resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            
        elif arch == 'resnet50d':
            model_backbone = resnet50d(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )    
            
        elif arch == 'sge_resnet50':
            model_backbone = sge_resnet50(pretrained = True)
            #in_features =  4096
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
        elif arch =='resnext50_32x4d':

            model_backbone = models.resnext50_32x4d(pretrained = True)
            self.backbone = (nn.Sequential(*list(model_backbone.children())[:-2]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip

        
                 
        elif arch == 'effnetb4':
            model_backbone = EfficientNet.from_pretrained('efficientnet-b4')      
            self.backbone = model_backbone#(nn.Sequential(*list(model_backbone.children())[:-3]) )
            self.backbone_lc = nn.ReLU(inplace=True) #skip
            

        self.imfeat_dim = cfg.MODEL.IMG_FCS  #(4096,512)

       
        self.blk_cb = nn.Sequential(resblock(self.imfeat_dim[0]//2,self.imfeat_dim[0]//8),
                                    layers.AvgPool())
                                    
        
        self.blk_rb = nn.Sequential(resblock(self.imfeat_dim[0]//2,self.imfeat_dim[0]//8),
                                    layers.AvgPool())
        
        
        # default only ave pool
        if cls_layer =='FC':
            self.classifer = nn.Linear(in_features = self.imfeat_dim[0],out_features =n_class,bias = True)
        elif  cls_layer =='FCNorm':
            self.classifer = FCNorm(in_features = self.imfeat_dim[0],out_features =n_class)

            
        self.init()
        if cfg.MODEL.PRETRAIN_PATH is not None  and os.path.exists(cfg.MODEL.PRETRAIN_PATH):
            self.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))


    def forward(self, x,x_rb=None,alpha = 0.5):
        backbone_cb = self.backbone(x)
        if x_rb is None: # valid
            backbone_rb = backbone_cb
        else:
            backbone_rb = self.backbone(x_rb)
            
        
        result = 2.0* torch.cat((self.blk_cb(backbone_cb)*alpha,(1.0-alpha) *self.blk_rb(backbone_rb)),dim=1)
        result = self.classifer(result)
            
            
           
        return result

    
    def init(self):
        #pass
        #self.two_view_resnet.apply(_kaiming_init_)
        #self.head.apply(_kaiming_init_)
        init_cnn(self.blk_cb)
        init_cnn(self.blk_rb)
        init_cnn(self.classifer)
        #init_cnn(self.head_im[-1],negative_slope = 1.0)