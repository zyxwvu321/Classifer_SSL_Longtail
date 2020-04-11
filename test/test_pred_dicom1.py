#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:29:38 2019

@author: minjie
"""
import cv2
import torch
import modeling.models as models
from pathlib import Path

from fastai import *
from fastai.vision import *
import pydicom
import matplotlib.pyplot as plt
from utils.utils import del_mkdirs
from tqdm import tqdm
from transforms.data_preprocessing import  TestAugmentation_albu
#model = models.LocateBoneModel()
from transforms.data_preprocessing import  TestAugmentation_albu
import shutil


#%% convert dicom to png
path_dicom = './data/ori/ori_dicom'


fd_list = [str(Path(path_dicom)/'pos'),str(Path(path_dicom)/'neg')]


fd_list_png = str(Path(fd_list[0].replace('dicom','png')).parents[0])



del_mkdirs([fd_list_png])

#del_mkdirs([str(Path(fd_list_png[0]).parents[0])])

for fds in fd_list:
    flist = [str(fn) for fn in sorted(Path(fds).glob('*.dcm'))]
    
    for fn in flist:
        
        with pydicom.dcmread(fn) as dc:
            img_dicom = dc.pixel_array
            #print(img_dicom.shape, img_dicom.shape[0]/img_dicom.shape[1])
            
            
            fname_png = fn.replace('.dcm','.png').replace('dicom','png').replace('0.png','_AP.png').replace('1.png','_LAT.png').replace('pos/','Pos').replace('neg/','Neg')
  
            cv2.imwrite(fname_png, (img_dicom/16384.0*255).astype('uint8'))
            #x = pil2tensor(img_dicom,np.float32).div_(16384)
            #Image_AP = Image(x)
            


path_xml= './data/ori/ori_png0'


fd_list = [str(Path(path_xml)/'pos'),str(Path(path_xml)/'neg')]
for fds in fd_list:
    flist = [str(fn) for fn in sorted(Path(fds).glob('*.xml'))]
    
    for fn in flist:
        fname_xml = fn.replace('0.xml','_AP.xml').replace('1.xml','_LAT.xml').replace('pos/','Pos').replace('neg/','Neg').replace('ori_png0','ori_png')
        shutil.copyfile(fn,fname_xml)