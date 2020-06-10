# encoding: utf-8
"""
build transform
"""

#import torchvision.transforms as T
#from PIL import Image
#from .transforms import RandomErasing,RandomErasingCorner
from .data_preprocessing import  TrainAugmentation_albu,TestAugmentation_albu,TrainAugmentation_bone,TestAugmentation_bone



import torchvision.transforms as transforms
from data.transforms.RandAugment.augmentations import RandAugment,Lighting


_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}

def get_transform(resize, phase='train'):
    if phase == 'train':
        tfms =  transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            #Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        
        # Add RandAugment with N, M(hyperparameter)
        #tfms.transforms.insert(1, RandAugment(2, 9))
        return tfms
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])




def build_transforms(cfg, is_train=True, weak_aug = False,n_aug = 1):

    if cfg.INPUT.USE_FGTFMS is True:
        if is_train is True:
            transform = get_transform( cfg.INPUT.SIZE_TRAIN_PRED, 'train')
        else:
            transform = get_transform( cfg.INPUT.SIZE_TRAIN_PRED, 'val')
        return transform

        
        
    if cfg.DATASETS.NAMES =='ISIC':
        if is_train is True:
            
            if weak_aug is False:
                transform = TrainAugmentation_albu(sz_hw = cfg.INPUT.SIZE_TRAIN_IN, \
                            mean = cfg.INPUT.PIXEL_MEAN, std = cfg.INPUT.PIXEL_STD,  \
                            crp_scale = cfg.INPUT.CRP_SCALE, crp_ratio = cfg.INPUT.CRP_RATIO, n_aug = n_aug,out_augpos = cfg.DATASETS.OUT_AUGPOS)    
            else:
                transform = TrainAugmentation_albu(sz_hw = cfg.INPUT.SIZE_TRAIN_IN, \
                            mean = cfg.INPUT.PIXEL_MEAN, std = cfg.INPUT.PIXEL_STD,  \
                            crp_scale = cfg.INPUT.CRP_SCALE_WEAK, crp_ratio = cfg.INPUT.CRP_RATIO,weak_aug = True, n_aug = n_aug)    

            
        else:
            transform  = TestAugmentation_albu(size = cfg.INPUT.SIZE_TRAIN_IN, mean = cfg.INPUT.PIXEL_MEAN, std = cfg.INPUT.PIXEL_STD,out_augpos = cfg.DATASETS.OUT_AUGPOS)  
            
            
    elif cfg.DATASETS.NAMES =='BoneXray':
        #size = configs.image_size, mean = configs.image_mean, std = configs.image_std, ext_p =configs.ext_p
        if is_train is True:
            transform = TrainAugmentation_bone(sz_in_hw = cfg.INPUT.SIZE_TRAIN_IN, sz_out_hw = cfg.INPUT.SIZE_TRAIN_PRED, \
                            mean = cfg.INPUT.PIXEL_MEAN, std = cfg.INPUT.PIXEL_STD,  \
                            minmax_h = cfg.INPUT.MINMAX_H, w2h_ratio = cfg.INPUT.W2H_RATIO)    
        else:
            transform  = TestAugmentation_bone(sz_in_hw = cfg.INPUT.SIZE_TRAIN_IN,sz_out_hw = cfg.INPUT.SIZE_TRAIN_PRED, mean = cfg.INPUT.PIXEL_MEAN, std = cfg.INPUT.PIXEL_STD)  
    else:
        raise ValueError('unknown transform for dataset {cfg.DATASETS.NAMES}')
    #   local att
    #train_transform_lc = TrainAugmentation_albu(sz_in_hw = configs.sz_in_hw_lc, sz_out_hw = configs.sz_out_hw_lc, mean = configs.image_mean, std = configs.image_std, 
    #                                         minmax_h= configs.minmax_h_lc,w2h_ratio = configs.w2h_ratio_lc)    

    return transform





