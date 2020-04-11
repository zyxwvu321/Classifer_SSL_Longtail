# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:38:32 2019

@author: cmj
"""

# -*- coding: utf-8 -*-


import argparse
import os
import os.path as osp
import shutil
from pathlib import Path
from utils.utils import del_mkdirs
from tqdm import tqdm
import cv2
import numpy as np

dict_label = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AKIEC', 4: 'BKL', 5: 'DF', 6: 'VASC'}

fd_18  = './data/ISIC/train18/'
fd_19  = './data/ISIC/train19/'

fd_tar_train = './data/ISIC/train/'
fd_tar_valid = './data/ISIC/valid/'
np.random.seed(0)
val_pct = 0.2

for ky,val in dict_label.items():
    flist_19 = (Path(fd_19)/val).glob('*.jpg')
    flist_18 = (Path(fd_18)/val).glob('*.jpg')

    fn_18 = [fn.stem for fn in flist_18]
    fn_19 = [fn.stem for fn in flist_19]
    

    for fn in fn_18:
        if fn in fn_19:
            fn_19.remove(fn)
    
    del_mkdirs([str(Path(fd_tar_train)/val),str(Path(fd_tar_valid)/val)])
    
    
    np.random.shuffle(fn_18)
    np.random.shuffle(fn_19)
    
    idx_18 = int(len(fn_18) * val_pct)
    idx_19 = int(len(fn_19) * val_pct)
    fn_18_tr = fn_18[idx_18:]
    fn_18_te = fn_18[:idx_18]
    
    fn_19_tr = fn_19[idx_19:]
    fn_19_te = fn_19[:idx_19]
    
    print(f'{val} {len(fn_18_tr)} {len(fn_18_te)} {len(fn_19_tr)} {len(fn_19_te)}')

    for fn in fn_18_tr:
        src_fn = (Path(fd_18)/val)/(fn + '.jpg')
        tar_fn = (Path(fd_tar_train)/val)/(fn + '.jpg')
        shutil.copyfile(src_fn,tar_fn)
        

    for fn in fn_18_te:
        src_fn = (Path(fd_18)/val)/(fn + '.jpg')
        tar_fn = (Path(fd_tar_valid)/val)/(fn + '.jpg')
        shutil.copyfile(src_fn,tar_fn)


    for fn in fn_19_tr:
        src_fn = (Path(fd_19)/val)/(fn + '.jpg')
        tar_fn = (Path(fd_tar_train)/val)/(fn + '.jpg')
        shutil.copyfile(src_fn,tar_fn)
        

    for fn in fn_19_te:
        src_fn = (Path(fd_19)/val)/(fn + '.jpg')
        tar_fn = (Path(fd_tar_valid)/val)/(fn + '.jpg')
        shutil.copyfile(src_fn,tar_fn)











