# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import math
import random

from skimage.draw import polygon
import numpy as np
import torch

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465),std =  (0.229, 0.224, 0.225)):
        self.probability = probability
        self.mean = mean
        self.std = std
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]#(1.0 - self.mean[0])/self.std[0]#
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]#(1.0 - self.mean[1])/self.std[1]#
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]#(1.0 - self.mean[2])/self.std[2]#
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]#(1.0 - self.mean[0])/self.std[0]#
                return img

        return img






class RandomErasingCorner(object):
    """ Randomly selects a triangle region in an image's corner and erases its pixels.
        
        paste four corners
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image. e.g. 0.02x0.02
         sh: Maximum proportion of erased area against input image. e.g. 0.25x0.25
         r1: Minimum aspect ratio of erased area. e.g. 0.5   (0.25/sqrt(2)x  0.25*sqrt(2))
         
         # drl: Different ratio for four corners, e.g. (drl,drh)*s
         #drh:
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.25, r1=0.5, 
                 mean=(0.4914, 0.4822, 0.4465),std =  (0.229, 0.224, 0.225)):#drl=0.8, drh = 1.0,
        self.probability = probability
        self.mean = mean
        self.std = std
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        #self.drl = drl
        #self.drh = drh

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img
        
        ss  = np.random.uniform(self.sl, self.sh,(4,))
        asr = np.random.uniform(self.r1, 1 / self.r1,(4,))
        hh, ww = img.size()[1], img.size()[2]
        
        crp_h = hh*ss*np.sqrt(asr)
        crp_w = ww*ss*np.sqrt(1.0 / asr)
        
        #drs = np.random.uniform(self.drl,self.drh,(4,))
        
        
#        crp_h_lu,crp_w_lu =  int(round(drs[0] * crp_h)), int(round(drs[0] * crp_w))
#        crp_h_ru,crp_w_ru =  int(round(drs[1] * crp_h)), int(round(drs[1] * crp_w))        
#        crp_h_ld,crp_w_ld =  int(round(drs[2] * crp_h)), int(round(drs[2] * crp_w))        
#        crp_h_rd,crp_w_rd =  int(round(drs[3] * crp_h)), int(round(drs[3] * crp_w))
#        
        crp_h  = np.round(crp_h).astype('int')
        crp_w  = np.round(crp_w).astype('int')
        
        crp_h_lu,crp_w_lu =  crp_h[0],crp_w[0]
        crp_h_ru,crp_w_ru =  crp_h[1],crp_w[1]
        crp_h_ld,crp_w_ld =  crp_h[2],crp_w[2]
        crp_h_rd,crp_w_rd =  crp_h[3],crp_w[3]
        
        
        
        r_lu,c_lu = np.array([-1,-1, crp_h_lu]), np.array([-1, crp_w_lu,-1])        
        r_ru,c_ru = np.array([-1,-1, crp_h_ru]), np.array([ww-crp_w_ru-1, ww, ww])        
        r_ld,c_ld = np.array([hh-crp_h_ld-1, hh, hh]), np.array([-1, crp_w_ld,-1])        
        r_rd,c_rd = np.array([hh-crp_h_rd-1, hh, hh]), np.array([ ww, ww,ww-crp_w_rd-1])
        
        rr_lu, cc_lu = polygon(r_lu, c_lu)        
        rr_ru, cc_ru = polygon(r_ru, c_ru)        
        rr_ld, cc_ld = polygon(r_ld, c_ld)        
        rr_rd, cc_rd = polygon(r_rd, c_rd)
        
        r_all = torch.from_numpy(np.hstack((rr_lu,rr_ru,rr_ld,rr_rd)))        
        c_all = torch.from_numpy(np.hstack((cc_lu,cc_ru,cc_ld,cc_rd)))
        
        if img.size()[0] == 3:
            img[0, r_all, c_all] = (1.0 - self.mean[0])/self.std[0]#self.mean[0]#1.0
            img[1, r_all, c_all] = (1.0 - self.mean[1])/self.std[1]#self.mean[1]#1.0
            img[2, r_all, c_all] = (1.0 - self.mean[2])/self.std[2]#self.mean[2]#1.0
        else:
            img[0, r_all,c_all] = (1.0 - self.mean[0])/self.std[0]# self.mean[0]#
            
        return img




if __name__ == '__main__':
    import cv2
    img = cv2.imread('../data/fingerprint/lens_176176_newlens/ori/2/wet/L1/V2/0000_p_v2.bmp')
    img = np.transpose(img,(2,0,1))
    rec = RandomErasingCorner()
    num_aug = 20
    for n in range(num_aug):
        imgt = torch.from_numpy(img.copy()).float()/255.0
        img_aug = (rec(imgt).numpy()*255).astype('uint8')
        img_aug = np.transpose(img_aug,(1,2,0))
        cv2.imwrite(f'{n}.png', img_aug)
        
    
    