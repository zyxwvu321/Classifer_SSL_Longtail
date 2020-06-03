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
def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})




class TrainAugmentation_albu:
    def __init__(self, sz_hw = (384,384),mean=0, std=1.0, crp_scale=(0.08, 1.0),crp_ratio = (0.75, 1.3333), weak_aug = False,n_aug = 1):
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
#            self.augment = A.Compose(self.augment,\
#                                     keypoint_params = A.KeypointParams(format= 'xys', label_fields=['category_id'], \
#                                                                        remove_invisible=False, angle_in_degrees=True))
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
            augmented = self.augment(image = img)
            return augmented['image']
        else:
            # test multi-aug
            return torch.stack([self.augment(image = img)['image'] for _ in range(self.n_aug)]) 
            
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
    
    
        return augmented['image']

class TestAugmentation_albu:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        if isinstance(size, int):
            size = (size,size)
            
        self.mean = mean
        self.size = size


        
        self.augment = A.Compose([A.Resize( size[0], size[1],  interpolation=cv2.INTER_CUBIC,  p=1),
                                 A.Normalize(mean=mean, std=std, p=1.0),
                                 ToTensor_albu()
                                 ])    

    def __call__(self, img):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            
            labels: labels of boxes.
        """        
        augmented = self.augment(image = img)
        return augmented['image'] 
    




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
    
