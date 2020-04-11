from skimage import exposure, img_as_float
import numpy as np
#import random
import cv2

def sod_minkowski(img):
    """Minkowski P-Norm Shades of Grey"""
    assert img.dtype =='uint8', 'dtype != uint8'
    assert img.shape[2]==3, 'ch!=3'
    
    img = (img/255.0).astype('float32')
    img0 = np.power(img,6.0)
    b,g,r = cv2.split(img0)
    bm,gm,rm = np.mean(b),np.mean(g),np.mean(r)
    gray = np.mean([bm,gm,rm])
    gray = np.power(gray, 1/6.0)
    
    bm,gm,rm =np.power(bm, 1/6.0),np.power(gm, 1/6.0),np.power(rm, 1/6.0)
    b,g,r = cv2.split(img)
    r = gray/rm*r
    g = gray/gm*g
    b = gray/bm*b

    # r = np.uint8(cv2.normalize(r, 0, 255, cv2.NORM_MINMAX)*255)
    # g = gray/np.mean(g)*g
    # g = np.uint8(cv2.normalize(g, 0, 255, cv2.NORM_MINMAX)*255)
    # b = gray/np.mean(b)*b
    # b = np.uint8(cv2.normalize(b, 0, 255, cv2.NORM_MINMAX)*255)
    return (np.clip(cv2.merge((b,g,r)) * 255,0,255)).astype('uint8'), (gray/rm, gray/gm,gray/bm)


def shades_of_gray_method(img):#, gamma):

    #img = img_as_float(image)
    #img = exposure.adjust_gamma(image, gamma=gamma)

    """
    Illuminant estimated using Minkowski norm
    -----------------------------------------
    """
    p = 6
    shape = img.shape
    N = shape[0] * shape[1]

    Re = np.power(np.power(img[:,:,0], p).mean(), 1/p)
    Ge = np.power(np.power(img[:,:,1], p).mean(), 1/p)
    Be = np.power(np.power(img[:,:,2], p).mean(), 1/p)

    e = np.array([Re, Ge, Be])
    e_ = np.sqrt((e**2).sum())
    e_gorro = e/e_

    d = 1/e_gorro

    img[:,:,0] = img[:,:,0]*d[0]
    img[:,:,1] = img[:,:,1]*d[1]
    img[:,:,2] = img[:,:,2]*d[2]

    return img