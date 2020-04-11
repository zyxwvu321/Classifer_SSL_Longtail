#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 18:17:18 2020

@author: minjie
"""
import numpy as np

import xml.etree.ElementTree as ET
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
        boxes.append(np.array([x1, y1, x2, y2]).astype('int'))

    
#    if len(boxes)!=1:
#        raise ValueError(f"More than one box in {annotation_file}")  
    boxes = np.array(boxes)
    return boxes

def extend_box(box,sz,ex_p):
    # sz= (h,w)
    box_ext = box.copy()
    
    box_w,box_h = box[2]-box[0],box[3]-box[1]
    box_ext[0] = box[0] - int(box_w * ex_p *0.5)
    box_ext[2] = box[2] + int(box_w * ex_p *0.5)
    
    box_ext[1] = box[1] - int(box_h * ex_p *0.5)
    box_ext[3] = box[3] + int(box_h * ex_p *0.5)
    
    box_ext[0] = max(0,box_ext[0])
    box_ext[1] = max(0,box_ext[1])
    box_ext[2] = min(sz[1]-1,box_ext[2])
    box_ext[3] = min(sz[0]-1,box_ext[3])
    return box_ext




def extend_box_square(box,sz,ex_p):
    # sz= (h,w)
    box_ext = box.copy()
    
    box_w,box_h = box[2]-box[0],box[3]-box[1]
    
    max_wh = max(box_w,box_h)
    
    xx = (box[2]+box[0])/2.0
    yy = (box[3]+box[1])/2.0
    box_ext[0] = xx - int(max_wh * (1+ex_p) *0.5)
    box_ext[2] = xx + int(max_wh * (1+ex_p) *0.5)
    
    box_ext[1] = yy - int(max_wh * (1+ex_p) *0.5)
    box_ext[3] = yy + int(max_wh * (1+ex_p) *0.5)
    
    box_ext[0] = max(0,box_ext[0])
    box_ext[1] = max(0,box_ext[1])
    box_ext[2] = min(sz[1]-1,box_ext[2])
    box_ext[3] = min(sz[0]-1,box_ext[3])
    box_ext= np.round(box_ext)
    
    return box_ext


def extend_box_square_crp(img,box,ex_p):
    # sz= (h,w)
    assert img.dtype == 'uint8'
    h_in,w_in,ch_in = img.shape
    
    box_ext = box.copy()
    
    box_w,box_h = box[2]-box[0],box[3]-box[1]
    
    max_wh = max(box_w,box_h)
    
    xx = (box[2]+box[0])/2.0
    yy = (box[3]+box[1])/2.0
    box_ext[0] = xx - int(max_wh * (1+ex_p) *0.5)
    box_ext[2] = xx + int(max_wh * (1+ex_p) *0.5)
    
    box_ext[1] = yy - int(max_wh * (1+ex_p) *0.5)
    box_ext[3] = yy + int(max_wh * (1+ex_p) *0.5)
    
    box_ext= np.round(box_ext).astype('int32')
    
    
    
    h_out =  box_ext[3] - box_ext[1] + 1
    w_out =  box_ext[2] - box_ext[0] + 1
    
    
    img_crp =  np.full((h_out, w_out, ch_in), 128, dtype=np.uint8)   
    
    h_st = max(0, -box_ext[1])
    w_st = max(0, -box_ext[0])
    h_ed = min(h_out, h_out - (box_ext[3]-h_in+1))
    w_ed = min(w_out, w_out - (box_ext[2]-w_in+1))
    
    
    
    box_ext[0] = max(0,box_ext[0])
    box_ext[1] = max(0,box_ext[1])
    box_ext[2] = min(w_in-1,box_ext[2])
    box_ext[3] = min(h_in-1,box_ext[3])
    
    img_crp[h_st:h_ed,w_st:w_ed,:] = img[box_ext[1]:box_ext[3]+1,box_ext[0]:box_ext[2]+1,:]
    
    return img_crp
