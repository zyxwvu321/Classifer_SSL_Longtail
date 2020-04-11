# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:14:12 2020

@author: cmj
"""

import numpy as np
#import logging
import pathlib
import cv2

import os.path as osp
from pathlib import Path
import torch
from torch.utils.data import Dataset
from utils.utils import get_annotation_bbox
import pandas as pd
from utils.parse_meta import parse_age,parse_sex

def read_classes(classes_path):
    with open(classes_path,encoding = 'utf_8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names



class BoneXrayDataset_withmeta(Dataset): #return

    def __init__(self, root, info_csv = './dat/bone_meta.csv', transform=None, is_test = False):
        """Dataset for Custom data.
        Args:
            root: the root of dataset, 
            info_csv: img and its infos,
            fn_map:pth file 
        """
        super(BoneXrayDataset_withmeta, self).__init__()
        self.root = Path(root)
        self.info_csv = info_csv

        self.transform = transform

        df_info = pd.read_csv(self.info_csv)
        im_info = np.array(df_info.values)
        self.label    = im_info[:,4].astype('int64')
        

        self.flist = [str(self.root/fn) for fn in im_info[:,2]]
        self.flist2 = [str(self.root/fn) for fn in im_info[:,3]]

        self.fname = [Path(fn).stem for fn in self.flist]
        
        self.meta_age = parse_age(im_info[:,13], th_list=np.arange(5,90,5))
        #self.meta_pos = parse_pos(im_info[:,5], all_pos =['anterior torso','head/neck','lower extremity','palms/soles','posterior torso','upper extremity'])
        self.meta_sex = parse_sex(im_info[:,14])
        #self.color_gain = im_info[:,11:14].astype('float32')
        #self.meta_boxsz = parse_boxsz(im_info[:,1:3], self.bbox_info)
            
        self.uids = np.arange(len(self.flist)).astype('int32')
        
        self.is_test = is_test


    def __getitem__(self, index):
        
        image_fn1 = self.flist[index]
        image_fn2 = self.flist2[index]
        image1 = cv2.imread(image_fn1)
        image2 = cv2.imread(image_fn2)
        
        if image1 is None or image2 is None:
            print(f'image is None :{image_fn1} {image_fn2}' )
        
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        
        
        if self.is_test:
            label = -1
        else:
            label = self.label[index]
        
        
        if self.transform is not None:
            image1   = self.transform(image1)
            image2   = self.transform(image2)
        

        image = torch.cat((image1,image2),dim = 0)
        
        
        label = torch.tensor(label)
        
        meta_info = np.hstack((self.meta_age[index], self.meta_sex[index]))
        meta_info = torch.from_numpy(meta_info)
        return image, label,meta_info


    def __len__(self):
        return len(self.flist)

   
    def get_annotation(self, index):
        
        label = self.label[index]
        return label   

class CustomDataset:

    def __init__(self, root, transform=None, is_test=False,input_channels = 1):
        """Dataset for Custom data.
        Args:
            root: the root of dataset, 
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.input_channels = input_channels
        
                
        self.flist_AP = [str(fn) for fn in sorted(list(Path(root).glob('*_AP.png')))]        
        #self.ids = [osp.split(fn)[1].split('_')[0] for fn in self.flist_AP]
        
        self.ids = [Path(fn).stem.replace('_AP','') for fn in self.flist_AP]


        #print number of Pos & Neg set
        labels = [self._get_annotation(image_id)  for image_id in self.ids ]
        
        n_pos = (np.array(labels)==1).sum()
        n_neg =(np.array(labels)==0).sum()
        n_other = len(labels) - n_pos - n_neg
        print(f'Number of Positive: {n_pos},  Number of Negative: {n_neg},Number of Unknown: {n_other} ')


    def __getitem__(self, index):
        image_id = self.ids[index]
        
        
        labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        
 
        if self.transform:
            image['AP']   = self.transform(image['AP'])
            image['LAT']  = self.transform(image['LAT'])
#        print('APmean : {}'.format(image['AP'].mean()))
#        print('LATmean : {}'.format(image['LAT'].mean()))
#        print('APstd : {}'.format(image['AP'].std()))
#        print('LATstd : {}'.format(image['LAT'].std()))
#    
        
    
        
        #return (image['AP'],image['LAT']),  labels
        if self.input_channels == 1:
            return torch.stack((image['AP'][0],image['LAT'][0]),dim = 0), labels
        elif self.input_channels == 3:
            return torch.cat((image['AP'],image['LAT']),dim = 0), labels
        else:
            raise(ValueError(f'unknown input_channels: {self.input_channels}'))

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image['AP']   = self.transform(image['AP'])
            image['LAT']  = self.transform(image['LAT'])
        return image

    def _get_annotation(self, image_id):
        
        if image_id.lower().startswith('pos'):
            labels = 1
        elif image_id.lower().startswith('neg'):
            labels = 0
        else:
            labels = -1
            
            #raise ValueError(f"Unknown file label: {image_id}")  
        return labels

    def __len__(self):
        return len(self.flist_AP)

   
    
    def _read_image(self, image_id):
        image = dict()
        
        image_file_AP = self.root / str(image_id + '_AP.png')
        image_AP = cv2.imread(str(image_file_AP))
        image_file_LAT = self.root / str(image_id + '_LAT.png')
        image_LAT = cv2.imread(str(image_file_LAT))
        
        
        if image_AP is None or image_LAT is None:
            raise ValueError(f'error image_file {image_id}')
        
        image['AP'] = cv2.cvtColor(image_AP, cv2.COLOR_BGR2RGB)
        image['LAT'] = cv2.cvtColor(image_LAT, cv2.COLOR_BGR2RGB)
        
        return image


class CustomDataset_bbox:
    def __init__(self, root, transform=None):
    
        self.root = Path(root)
        self.transform = transform        
        self.flist = [str(fn) for fn in sorted(list(Path(root).rglob('*.png')))]         
    
    
        self.ids = [Path(fn).stem for fn in self.flist]


        #print number of Pos & Neg set
        self.bboxes = [self._get_annotation(image_fn)  for image_fn in self.flist]
    
    
    
    def __getitem__(self, index):
        image_fn = self.flist[index]
        
        
        boxes = self._get_annotation(image_fn)
        image = self._read_image(image_fn)
        
 
        if self.transform:
           image, boxes = self.transform(image, boxes)


        return image, boxes[0]
        
    def _read_image(self, image_fn):

        
        #image_file = self.root / (image_id +'.png')
        image = cv2.imread(str(image_fn))
        
        if image is None:
            raise ValueError(f'error image_file {image_fn}')
        return image

    def _get_annotation(self, image_fn):
        #ann_file = self.root / (image_id +'.xml')
        ann_file = str(image_fn).replace('.png','.xml')
        if osp.exists(ann_file):
            boxes = get_annotation_bbox(str(ann_file))
            
            #raise ValueError(f"Unknown file label: {image_id}")  
            return boxes
        else:
            print(f'warning no box for {image_fn}')
            return np.array([[0.0,0.0,1.0,1.0]]).astype('float32')


   
    def __len__(self):
        return len(self.flist)    


