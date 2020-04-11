#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:29:38 2019

@author: minjie
"""
import cv2
import torch
import numpy as np
import modeling.models as models
from pathlib import Path
import xml.etree.ElementTree as ET

from fastai import *
from fastai.vision import *
import pydicom
import matplotlib.pyplot as plt
from utils.utils import del_mkdirs
from tqdm import tqdm

def get_annotation_bbox(annotation_file):
        
    objects = ET.parse(annotation_file).findall("object")
    boxes = []
    
    
    for object in objects:
        
        
        bbox = object.find('bndbox')

        # VOC dataset format follows Matlab, in which indexes start from 0
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        boxes.append(np.array([[x1, y1], [x2, y2]]))

    
    if len(boxes)!=1:
        raise ValueError(f"More than one box in {annotation_file}")  
    
    return boxes[0]

#%% convert dicom to png
path_dicom = './data/ori/ori_png'


fd_list = [str(Path(path_dicom)/'pos'),str(Path(path_dicom)/'neg')]





w_AP = []


w_LAT = []

for fds in fd_list:
    flist = [str(fn) for fn in sorted(Path(fds).glob('*.xml'))]
    
    for fn in flist:
        box = get_annotation_bbox(fn)
        img = cv2.imread(fn.replace('.xml','.png'))
        hh,ww,_ = img.shape
        
        w_ratio = (box[1,0]-box[0,0])/ww
        w_center = (box[1,0]+box[0,0])/(2.0*ww)
        
        
        if fn.endswith('1.xml'):
            
            
            w_LAT.append([fn,ww,hh, w_center, w_ratio ])
            
        elif fn.endswith('0.xml'):
            w_AP.append([fn,ww,hh, w_center, w_ratio ])
        else:
            raise ValueError(f"Unknown pos {fn}")  
import pandas as pd
df = pd.DataFrame(w_AP, columns=['name','ww','hh','w_center','w_ratio'])
df.to_csv('w_AP.csv', index=False)
df = pd.DataFrame(w_LAT, columns=['name','ww','hh','w_center','w_ratio'])
df.to_csv('w_LAT.csv', index=False)
                
        