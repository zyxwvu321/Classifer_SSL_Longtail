import numpy as np
#import logging

import cv2

import os.path as osp
from pathlib import Path
import torch
import pandas as pd
import globalvar as gl

from PIL import Image

from utils.parse_meta import parse_age,parse_sex, parse_pos,parse_boxsz

from collections import Counter
import math
from torch.utils.data import Dataset
#from collections import Counter
from utils.image import gaussian_radius, draw_umich_gaussian,draw_dense_reg
#from utils.utils import extend_box_square
#from utils.box_utils import extend_box_square_crp





class ISIC_withmeta(Dataset): #return

    def __init__(self, root, meta_list, dict_label = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']  ,info_csv = './dat/all18_info.csv', 
        transform=None, transform_weak = None,is_test = False):
        """Dataset for Custom data.
        Args:
            root: the root of dataset, 
            info_csv: img and its infos,

            meta_list: used meta infos
        """

        super(ISIC_withmeta, self).__init__()
        #self.root = Path(root)
        self.meta_list = meta_list
        #self.info_csv = info_csv

        self.transform = transform
        self.transform_weak = transform_weak
        self.cfg = gl.get_value('cfg')
        
        
        
        if not isinstance(root,(list,tuple)):
            root = [root]
            info_csv = [info_csv]
            
        self.flist = []
        self.fname = []
        self.meta_feat = []
        self.label_name = dict_label
        
        
        for root0,info_csv0 in zip(root,info_csv):
            df_info = pd.read_csv(info_csv0)
            im_info = np.array(df_info.values)
            
            if len(self.flist)==0:
                self.flist = [str(Path(root0)/(fn + '.jpg')) for fn in im_info[:,0]]
                self.fname = [Path(fn).stem for fn in self.flist]
            else:
                flist =[str(Path(root0)/(fn + '.jpg')) for fn in im_info[:,0]]
                self.flist += flist
                self.fname += [Path(fn).stem for fn in flist]
                
            
    
            #self.bbox_info = np.array(im_info[:,7:11]).astype('float32')
            
            meta_feat = list()
            if 'age' in meta_list:
                meta_age = parse_age(im_info[:,4], th_list=np.arange(5,90,5))
                meta_feat.append(meta_age)
            if 'pos' in meta_list:
                meta_pos = parse_pos(im_info[:,5], all_pos =['anterior torso','head/neck','lower extremity','palms/soles','posterior torso','upper extremity'])
                meta_feat.append(meta_pos)
            if 'sex' in meta_list:
                meta_sex = parse_sex(im_info[:,6])
                meta_feat.append(meta_sex)
            if 'color_gain' in meta_list:
                color_gain = im_info[:,7:10].astype('float32')
                meta_feat.append(color_gain)
    #        if 'boxsz' in meta_list:
    #            meta_boxsz = parse_boxsz(im_info[:,1:3], self.bbox_info)
    #            meta_feat.append(meta_boxsz)
            meta_feat = np.hstack(tuple(meta_feat))
            
            if len(self.meta_feat)==0:
                self.meta_feat = meta_feat
                self.label    = im_info[:,3].astype('int64')
            else:
                self.meta_feat = np.vstack((self.meta_feat,meta_feat))
                self.label    = np.hstack((self.label,im_info[:,3].astype('int64')))
            #self.label    = im_info[:,3].astype('int64')
        
        
        
        
        
        freqs = Counter(self.label)
        
        if is_test is False:
            hist_label = np.array([freqs[k] for k in range(len(dict_label))])
            self.hist_label  =hist_label

            label_w = hist_label.mean()/hist_label
            self.label_w = np.clip(label_w, 0.20, 15.0)
        

        
        
            cfg = gl.get_value('cfg')
            cfg.DATASETS.LABEL_W = list(label_w)
            self.label_samp_w = label_w[self.label]  # for sampler
            self.label_samp_w[self.label==-1] = cfg.DATASETS.SAMPLERATIO_UNLABEL # unlabel data, weight
        
        self.is_test = is_test


    def __getitem__(self, index):
        
        image_fn = self.flist[index]
        image = cv2.imread(image_fn)
        
        if image is None:
            print(f'image is None :{image_fn}' )
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hh,ww,_ = image.shape
        
        if self.cfg.INPUT.USE_FGTFMS is True:
            image = Image.fromarray(image.astype('uint8'))
        
        
        if self.is_test:
            label = -1
        else:
            label = self.label[index]
        
        
        
        
        meta_info = self.meta_feat[index]          
        label = torch.tensor(label)
        meta_info = torch.from_numpy(meta_info)    
        
        if self.transform_weak is None or self.is_test is True:
            if self.transform is not None:
                image   = self.transform(image)
            return image,  label, meta_info
        else:
            
            if self.transform is not None:
                image_s   = self.transform(image)
            if self.transform_weak is not None:
                image_w   = self.transform_weak(image)

            return image_s,  label, meta_info,image_w
    
    def __len__(self):
        return len(self.flist)

    def get_annotation(self, index):
        
        label = self.label[index]
        return label
    
        
"""            
        if self.transform_localbbox  is not None:
            image_roi   = self.transform_localbbox(image_roi) """

       # meta_info = {'age':self.meta_age[index], 'pos': self.meta_pos[index], 'sex':self.meta_sex[index], 'color_gain': self.color_gain[index],  'boxsz': self.meta_boxsz[index]}
        #meta_info = np.hstack((self.meta_age[index], self.meta_pos[index], self.meta_sex[index], self.color_gain[index],  self.meta_boxsz[index]))

        

class CustomDataset_bbox:
    def __init__(self, root, box_root, transform=None,cfg = None,is_test=False):
        self.cfg = cfg
        self.root = Path(root)
        self.box_root = Path(box_root)
        self.transform = transform        
        self.flist = [str(fn) for fn in sorted(list(Path(root).rglob('*.jpg')))]         
        
        self.is_test = is_test
    
        #print number of Pos & Neg set

    
    
    
    def __getitem__(self, index):
        image_fn = self.flist[index]
        image = cv2.imread(image_fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        box_fn = str(Path(self.box_root)/(Path(image_fn).stem + '.txt'))
        
        if osp.exists(box_fn):
            xywh = np.loadtxt(box_fn)
        
        
            xx,yy,ww,hh = xywh
            x1,y1,x2,y2 = xx-ww/2,yy-hh/2,xx+ww/2,yy+hh/2

        
            boxes = np.array([[x1,y1,x2,y2]]).astype('float32')
        elif self.is_test is True:
            boxes = np.array([[0.0,0.0,1.0,1.0]]).astype('float32')
        else:
            raise ValueError('box file not exist')
 
        if self.transform:
           image, boxes = self.transform(image, boxes)


        return image, boxes[0]
        


   
    def __len__(self):
        return len(self.flist)    




class CustomDataset_bbox_centernet:
    def __init__(self, root, box_root, transform=None,configs=None):
    
        self.root = Path(root)
        self.box_root = Path(box_root)
        self.transform = transform        
        self.configs = configs
        self.flist = [str(fn) for fn in sorted(list(Path(root).rglob('*.jpg')))]         
    
    
        #print number of Pos & Neg set

    
    
    
    def __getitem__(self, index):
        image_fn = self.flist[index]
        image = cv2.imread(image_fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        box_fn = str(Path(self.box_root)/(Path(image_fn).stem + '.txt'))
        
        if osp.exists(box_fn):
            xywh = np.loadtxt(box_fn)
        
        
            xx,yy,ww,hh = xywh
            x1,y1,x2,y2 = xx-ww/2,yy-hh/2,xx+ww/2,yy+hh/2

        
            boxes = np.array([[x1,y1,x2,y2]]).astype('float32')
        else:
            boxes = np.array([[0.0,0.0,1.0,1.0]]).astype('float32')
        
 
        if self.transform:
           image, boxes = self.transform(image, boxes)

        
        #generate box_gt for loss
        #box x1,y1,x2,y2, [0,1]
        output_h,output_w,grid_wh = self.configs.hh,self.configs.ww,self.configs.grid_wh
        hin,win = self.configs.image_size
        
        hm = np.zeros((self.configs.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.configs.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        dense_xy = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.configs.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.configs.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.configs.max_objs), dtype=np.uint8)


        
        num_objs = min(boxes.shape[0], self.configs.max_objs)
        
        
#        gt_det = []
        for k in range(num_objs):
          bbox = boxes[k]
          h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
          if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h*grid_wh), math.ceil(w*grid_wh)))
            radius = max(0, int(radius))
            #radius = self.opt.hm_gauss if self.opt.mse_loss else radius
            ct = np.array(
              [(bbox[0] + bbox[2]) / 2.0 * grid_wh, (bbox[1] + bbox[3]) / 2.0* grid_wh], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            ct_int = np.clip(ct_int, 0, grid_wh-1)
            
            draw_umich_gaussian(hm[k], ct_int, radius)
            
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1

            
            draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
            draw_dense_reg(dense_xy, hm.max(axis=0), ct_int, reg[k], radius)
            
#            gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
#                           ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
#        
        #ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        #if self.opt.dense_wh:
        hm_a = hm.max(axis=0, keepdims=True)
        dense_mask = np.concatenate([hm_a, hm_a], axis=0)
        
        ret = {'hm': hm, 'wh': wh, 'xy': reg, 'ind': ind,'dense_xy': dense_xy,'dense_wh': dense_wh,'dense_mask':dense_mask, 'boxes': boxes}
        
        #ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
        #del ret['wh']
        #elif self.opt.cat_spec_wh:
          #ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
          #del ret['wh']
        #if self.opt.reg_offset:
          #ret.update({'reg': reg})
#        if self.opt.debug > 0 or not self.split == 'train':
#          gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
#                   np.zeros((1, 6), dtype=np.float32)
#          meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
#          ret['meta'] = meta
#        return ret        
#        
        
        return image, ret
        


   
    def __len__(self):
        return len(self.flist)   


class CustomDataset:

    def __init__(self, root, transform=None, label_name =None,is_test = False):
        """Dataset for Custom data.
        Args:
            root: the root of dataset, 
        """
        self.root = Path(root)
        self.transform = transform
        

        self.dict_label_inv = dict()
        for ky,val in label_name.items():
            self.dict_label_inv[val] = ky

                
        self.flist = [str(fn) for fn in sorted(list(Path(root).rglob('*.jpg')))]        
                
        self.ids = [Path(fn).stem for fn in self.flist]
        self.is_test = is_test

        #print number of Pos & Neg set
        if is_test:
            labels = [-1]*len(self.ids)
        else:
            labels = [self._get_annotation(fn)  for fn in self.flist ]
            freq = Counter(labels)
            lb_freq = list()
            for ky in label_name.keys():
                print(f'Number of {label_name[ky]} : {freq[ky]}')
                lb_freq.append(freq[ky])
            self.lb_freq = lb_freq

    def __getitem__(self, index):
        fn = self.flist[index]
        
        if self.is_test:
            label = -1
        else:
            label = self._get_annotation(fn)
        image = self._read_image(fn)
        
 
        if self.transform:
            image   = self.transform(image)
           


        return image, label


    def get_image(self, index):
        fn = self.flist[index]
        image = self._read_image(fn)
        if self.transform:
            image   = self.transform(image)
      
        return image
    
    def get_annotation(self, index):
        fn = self.flist[index]
        label = self._get_annotation(fn)
        return label
        
    
    def _get_annotation(self, fn):
        label = self.dict_label_inv[str(Path(fn).parts[-2])]
        
        return label

    def __len__(self):
        return len(self.flist)

   
    
    def _read_image(self, fn):
        
        image = cv2.imread(str(fn))        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #image = image.astype('float32')/255.0
        return image



