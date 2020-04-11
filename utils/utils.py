import shutil
import os
import numpy as np
import random 
import torch
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Iterable
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device(cfg):
    use_cuda = torch.cuda.is_available()
    if cfg.MODEL.DEVICE != 'cuda':
        use_cuda = False

    device = torch.device('cuda' if use_cuda else 'cpu')
    return device

def del_mkdirs(fd_list):
    for fd in fd_list:
        if os.path.exists(fd):
            shutil.rmtree(fd)
            while os.path.exists(fd): # check if it exists
                pass

        os.makedirs(fd)


def set_seed(manualSeed):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        #cudnn.enabled = False # if identity result is desired
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
  
        
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






def init_cnn_(m, f,negative_slope):
    if isinstance(m, (nn.Conv1d,nn.Conv2d,nn.Conv3d)):
        f(m.weight,a = negative_slope) # a = 0 for relu afterwards
        if getattr(m, 'bias', None) is not None: 
            m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        f(m.weight) 
        if getattr(m, 'bias', None) is not None: 
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.InstanceNorm1d,nn.InstanceNorm2d,nn.InstanceNorm3d)):
    #elif isinstance(m, (nn.InstanceNorm1d,nn.InstanceNorm2d,nn.InstanceNorm3d)): #bn Not learn parameters
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        
    for l in m.children(): 
        init_cnn_(l, f,negative_slope)

def init_cnn(m, uniform=False,negative_slope = 0.0):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    init_cnn_(m, f,negative_slope)




def list_to_txt(ll,fn):
    with open(fn, 'w') as f:
        for fname in ll:
            f.write("%s\n" % fname)


def setify(o): return o if isinstance(o,set) else set(listify(o))

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res


def get_files(path, extensions=None, recurse=False, include=None):
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
            if include is not None and i==0: d[:] = [o for o in d if o in include]
            else:                            d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, f, extensions)



class AvgerageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res



class ListContainer():
    def __init__(self, items): self.items = listify(items)
    def __getitem__(self, idx):
        try: return self.items[idx]
        except TypeError:
            if isinstance(idx[0],bool):
                assert len(idx)==len(self) # bool mask
                return [o for m,o in zip(idx,self.items) if m]
            return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res

def children(m): return list(m.children())

class Hook():
    def __init__(self, m, f): self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[])
    means,stds = hook.stats
    if mod.training:
        means.append(outp.data.mean())
        stds .append(outp.data.std())
        
class Hooks(ListContainer):
    def __init__(self, ms, f): super().__init__([Hook(m, f) for m in ms])
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self: h.remove()
        
        
class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x):
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x


def tuplelist_to_array(ss):
    ss = ss.replace('[','').replace(']','').replace('(','').replace(')','')
    
    if ss=='':
        ss = []
    else:
        ss = ss.split(',')
        ss = np.array(ss).astype('float32')
    return ss

def parse_bool(ss):
    assert len(ss)==1, 'state has only more element'
    ss =  ss[0].text

    if ss.lower() == 'true': return True
    elif ss.lower() == 'false': return False
    else: raise ValueError('state true or false is not set')

def parse_int(ss):
    assert len(ss)==1, 'state has only more element'
    ss =  np.int(ss[0].text)
    return ss


def parse_pairpoint(obj):
    b_check  = parse_bool(obj.findall('check'))
    pt1      = tuplelist_to_array(obj.find('point1').text)
    pt2      = tuplelist_to_array(obj.find('point2').text)
    return (b_check, pt1, pt2)

def parse_polygon(obj):
    if len(obj)==1:
        obj = obj[0]
        xx       = tuplelist_to_array(obj.find('x').text)
        yy       = tuplelist_to_array(obj.find('y').text)
    elif len(obj)==0:
        print('warning,no polygon')
        xx = []
        yy = []
    else:
        raise ValueError('polygon has only more element')
    
    return np.vstack((xx, yy)).T


def read_homo_match_anno(annotation_file):
        
    match_type = parse_int(ET.parse(annotation_file).findall("type"))
    homo       = tuplelist_to_array(ET.parse(annotation_file).find("homo").text)
    if len(homo)==6: #affine
        homo       =  np.hstack((homo,np.array([0.0,0.0,1.0])))

    points = ET.parse(annotation_file).findall('point')
    points_prop = []
    for point in points:
        points_prop.append(parse_pairpoint(point))

    
    polygon1 = parse_polygon(ET.parse(annotation_file).findall('polygon1'))
    polygon2 = parse_polygon(ET.parse(annotation_file).findall('polygon2'))
    polygon_prop = [polygon1,polygon2]
    
    return match_type, points_prop, polygon_prop,homo

def iou_mask(im1,im2):
    assert len(im1.shape)==2 and len(im2.shape)==2, 'input mask with dim = 2'
    return ((im1>0) & (im2>0)).sum()/(((im1>0) | (im2>0)).sum() + 0.001)






    
    
    
    