#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:36:15 2020
dataset transform
@author: minjie
"""



import albumentations as A
from albumentations.pytorch import ToTensor as ToTensor_albu
import cv2
import torch
from multiprocessing import Pool
from utils.parse_meta import parse_kpds
import numpy as np

def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})




class TrainAugmentation_albu:
    def __init__(self, sz_hw = (384,384),mean=0, std=1.0, crp_scale=(0.08, 1.0),crp_ratio = (0.75, 1.3333), weak_aug = False,n_aug = 1,out_augpos = False):
        """
        Args:
            weak_aug, week aug for fixmatch
            
        """


        if isinstance(sz_hw, int):
            sz_hw = (sz_hw,sz_hw)
            
        self.mean = mean
        self.std  = std
        self.sz_hw = sz_hw
        self.crp_scale = crp_scale
        self.crp_ratio = crp_ratio
        self.n_aug = n_aug # number of repeated augmentation
        self.out_augpos = out_augpos
    
        if self.sz_hw[0] == self.sz_hw[1]:

            self.T_aug = A.Compose([A.Rotate(p=0.5),
                                    A.RandomResizedCrop(height = self.sz_hw[0], width = self.sz_hw[1],  scale=self.crp_scale, ratio=self.crp_ratio,
                                                        interpolation = cv2.INTER_CUBIC,p = 1.0),
                                    A.Flip(p = 0.5),
                                    A.RandomRotate90(p = 0.5)])
        else:
            self.T_aug = A.Compose([A.Rotate(p=0.5),
                                    A.RandomResizedCrop(height = self.sz_hw[0], width = self.sz_hw[1],  scale=self.crp_scale, ratio=self.crp_ratio,
                                                        interpolation = cv2.INTER_CUBIC,p = 1.0),
                                    A.Flip(p = 0.5)])

        self.I_aug = A.Compose([ A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,  p=0.5), 
                                 A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=15, val_shift_limit=20,p = 0.5),
                                 
                                 A.OneOf([A.Blur(blur_limit=5, p=0.3),
                                 A.GaussNoise(var_limit=(5.0, 10.0), p=0.3),
                                 A.IAASharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.4)],p=0.5)])

       

        self.N_aug = A.Compose([A.Normalize(mean=mean, std=std,  p=1.0),
                                ToTensor_albu()])
        
        if weak_aug is False:    
            self.augment = A.Compose([ self.T_aug, self.I_aug,self.N_aug])  
            
            
            
            
#            self.augment = A.Compose([ self.T_aug, self.I_aug])    
#            self.augment = A.Compose(self.augment, bbox_params={'format': 'albumentations', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['category_id']})
            if self.out_augpos is True:
                self.augment = A.Compose(self.augment,\
                                     keypoint_params = A.KeypointParams(format= 'xys', \
                                                                        remove_invisible=False, angle_in_degrees=True))#label_fields=['category_id'], \
        else:
            #weak augment
            self.T_aug =  A.RandomResizedCrop(height = self.sz_hw[0], width = self.sz_hw[1],  scale=self.crp_scale, ratio=self.crp_ratio,
                                                        interpolation = cv2.INTER_CUBIC,p = 1.0)
            self.augment = A.Compose([ self.T_aug, self.N_aug])    
    
    def __call__(self, img):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            
            labels: labels of boxes.
        """        
        if self.n_aug==1:
            #augmented = self.augment(image = img)
            if self.out_augpos is False:
                augmented = self.augment(image = img)
                return augmented['image']
            else:
                hh,ww,_ = img.shape
                points = [[ww/2.0,hh/2.0,1.0],[0.0,0.0,1.0]]
                hw_in = img.shape[:2]
                
                augmented = self.augment(image = img,keypoints=points)
                
                image_aug = augmented['image']
                hw_out = image_aug.shape[1:]
                feat_kpds = torch.tensor(parse_kpds(augmented['keypoints'],hw_in,hw_out))
                
            return (image_aug,feat_kpds)
        else:
            # test multi-aug
            if self.out_augpos is False:
                return torch.stack([self.augment(image = img)['image'] for _ in range(self.n_aug)]) 
            else:
                img_out = []
                feat_out = []
                trans_out = []
                hh,ww,_ = img.shape
                #points = [[ww/2.0,hh/2.0,1.0],[0.0,0.0,1.0]]
                points = [[ww/2.0,hh/2.0,1.0],[0.0,0.0,1.0],[ww,0.0, 1.0]] # add one point for cv2.getAffineTransform
                
                hw_in = img.shape[:2]
                
                for _ in range(self.n_aug):
    
                    augmented = self.augment(image = img,keypoints=points)
                    image_aug = augmented['image']
                    hw_out = image_aug.shape[1:]     
                    #feat_kpds = torch.tensor(parse_kpds(augmented['keypoints'],hw_in,hw_out))
                    feat_kpds = torch.tensor(parse_kpds(augmented['keypoints'][:2],hw_in,hw_out))
                    
                    pts2 = augmented['keypoints']
                    pts1 = np.float32([pt[:2] for pt in points])
                    pts2 = np.float32([pt[:2] for pt in pts2])
                    
                    trans = cv2.getAffineTransform(pts2,pts1)
                    trans_out.append(trans)
                    img_out.append(image_aug)
                    feat_out.append(feat_kpds)
                    
                return (torch.stack(img_out), {'feat_out':torch.stack(feat_out), 'trans_out': np.stack(trans_out)})
                
            #return torch.stack([self.augment(image = img)['image'] for _ in range(self.n_aug)]) 
            
#            imgs = []
#            pts  = []
#            
#            hh,ww,_ = img.shape
#            for _ in range(self.n_aug):
#                #points = [ww/2.0,hh/2.0,1.0]
#                points = [[0.0,0.0,1.0], [0.0,hh,1.0], [ww,0.0,1.0],[ww,hh,1.0]]
#                augmented = self.augment(image = img,keypoints=points,category_id = ['0'])
#                imgs.append(augmented['image'])
#                pts.append(augmented['keypoints'])
    
#        # NOTE: use bbox will have prob that box is outside crop region.
#        bboxes= [[0.45, 0.45, 0.55, 0.55]]
#       
#        augmented = self.T_aug(image = img,bboxes = bboxes,category_id = ['0'])
        
#        hh,ww,_ = img.shape
#        points = [[ww/2.0,hh/2.0,1.0]]
#        augmented = self.augment(image = img,keypoints=points,category_id = ['0'])
    
    
        #return augmented['image']

class TestAugmentation_albu:
    def __init__(self, size, mean=0, std=1.0,out_augpos = False):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        if isinstance(size, int):
            size = (size,size)
            
        self.mean = mean
        self.size = size
        self.out_augpos = out_augpos
        
        
        self.augment = A.Compose([A.Resize( size[0], size[1],  interpolation=cv2.INTER_CUBIC,  p=1),
                                 A.Normalize(mean=mean, std=std, p=1.0),
                                 ToTensor_albu()
                                 ])    
        if self.out_augpos is True:   
    
            self.augment = A.Compose(self.augment,\
                                     keypoint_params = A.KeypointParams(format= 'xys', \
                                                                        remove_invisible=False, angle_in_degrees=True))

    def __call__(self, img):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            
            labels: labels of boxes.
        """     
        if self.out_augpos is False:  
            augmented = self.augment(image = img)
            return augmented['image'] 
        else:

            hh,ww,_ = img.shape
            points = [[ww/2.0,hh/2.0,1.0],[0.0,0.0,1.0]]
            hw_in = img.shape[:2]
            augmented = self.augment(image = img,keypoints=points)
            image_aug = augmented['image']
            
            hw_out = image_aug.shape[1:]
            feat_kpds = torch.tensor(parse_kpds(augmented['keypoints'],hw_in,hw_out))
            
    
                
            return (image_aug,feat_kpds)



class TrainAugmentation_bone:
    def __init__(self, sz_in_hw = (512,512), sz_out_hw = (448,448),mean=0, std=1.0, minmax_h = (0,128), w2h_ratio = 1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        if isinstance(sz_in_hw, int):
            sz_in_hw = (sz_in_hw,sz_in_hw)
        if isinstance(sz_out_hw, int):
            sz_out_hw = (sz_out_hw,sz_out_hw)
            
        self.mean = mean
        self.sz_in_hw = sz_in_hw
        self.sz_out_hw = sz_out_hw
        #self.crp_scale = crp_scale
        #self.crp_ratio = crp_ratio
        self.minmax_h  = minmax_h
        self.w2h_ratio = w2h_ratio


        self.I_aug = A.Compose([A.Resize( sz_in_hw[0], sz_in_hw[1],  interpolation=1,  p=1),
                                 A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,  p=0.5),                                  
                                 A.OneOf([A.Blur(blur_limit=5, p=0.3),
                                 A.GaussNoise(var_limit=(5.0, 10.0), p=0.3),
                                 A.IAASharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.4)],p=0.5)])

  
        self.T_aug = A.RandomSizedCrop(min_max_height = (self.minmax_h[0],self.minmax_h[1]),height = self.sz_out_hw[0], width = self.sz_out_hw[1],\
                                       w2h_ratio = self.w2h_ratio,p = 1.0)
        

        self.N_aug = A.Compose([A.Normalize(mean=mean, std=std,  p=1.0),
                                ToTensor_albu()])
        
        self.augment = A.Compose([self.I_aug, self.T_aug,self.N_aug])    
    

    
    def __call__(self, img):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            
            labels: labels of boxes.
        """        
        augmented = self.augment(image = img)
        return augmented['image']



class TestAugmentation_bone:
    #def __init__(self, size, mean=0, std=1.0, ext_p =(-0.125,0.25)):
    def __init__(self,  sz_in_hw = (512,512), sz_out_hw = (448,448), mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        if isinstance(sz_in_hw, int):
            sz_in_hw = (sz_in_hw,sz_in_hw)
            
        if isinstance(sz_out_hw, int):
            sz_out_hw = (sz_out_hw,sz_out_hw)
            
        self.augment = A.Compose([A.Resize( sz_in_hw[0], sz_in_hw[1],  interpolation=1,  p=1),
                                 A.CenterCrop(sz_out_hw[0], sz_out_hw[1],  p=1.0),
                                 A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                                 ToTensor_albu()
                                 ])    

#    
    def __call__(self, img):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            
            labels: labels of boxes.
        """        
        augmented = self.augment(image = img)
        return augmented['image'] 
    
