from pathlib import Path
import os
import numpy as np
import cv2
from color_enh import shades_of_gray_method,sod_minkowski
os.makedirs('../data/color_enh',exist_ok = True)
flist = Path('../data/train19').rglob('*.jpg')
flist = [str(fn) for fn in flist]

idx = np.random.permutation(len(flist))
idx = idx[:50]


for id in idx:
    img = cv2.imread(flist[id])
    img = (img/255.0).astype('float32')
    #img_eh = shades_of_gray_method(img)
    img_eh = sod_minkowski(img)
    #img_eh = np.clip(img_eh,0.0,1.0)
    cv2.imwrite(f'../data/color_enh/{id}_ori.jpg',(img*255).astype('uint8'))
    cv2.imwrite(f'../data/color_enh/{id}_eh.jpg',(img_eh).astype('uint8'))

