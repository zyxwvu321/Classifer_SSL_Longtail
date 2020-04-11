""" oxford 102  (flower) Dataset
"""
import os

from PIL import Image
from torch.utils.data import Dataset
#import torchvision.transforms as transforms
from scipy import io
from pathlib import Path
import numpy as np
DATAPATH = '../data/flower'
from data.transforms.build import get_transform

def id2fn(im_id):
    im_id = str(im_id)
    n_bit = 5-len(im_id)
    fn  = 'image_' +  '0'*n_bit + im_id +  '.jpg'
    return fn
    
    
#def get_transform(resize, phase='train'):
#    if phase == 'train':
#        return transforms.Compose([
#            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
#            transforms.RandomCrop(resize),
#            transforms.RandomHorizontalFlip(0.5),
#            transforms.ColorJitter(brightness=0.126, saturation=0.5),
#            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#        ])
#    else:
#        return transforms.Compose([
#            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
#            transforms.CenterCrop(resize),
#            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#        ])


class FlowerDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=(448,448)):
        assert phase in ['train', 'val']
        # tstid used for train, trnid+valid used for valid (6149  / 1020+1020)
        
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 102
        
        fn_label = Path(DATAPATH)/'imagelabels.mat'
        fn_split = Path(DATAPATH)/'setid.mat'
        
        self.labels = io.loadmat(str(fn_label))['labels'][0].astype('int64') - 1
        ids = io.loadmat(str(fn_split))
        
        trnid  =  ids['tstid'][0].astype('int64')  
        valid =  np.hstack((ids['valid'][0],ids['trnid'][0])).astype('int64') 
        
        
        if self.phase == 'train':
            self.image_ids = trnid
        else:
            self.image_ids = valid
        

        self.fname = [os.path.join(DATAPATH, 'images', id2fn(ids))  for ids in self.image_ids]   

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_ids[item]
        
        # image
        image = Image.open(os.path.join(DATAPATH, 'images', id2fn(image_id))).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.labels[image_id - 1] # count begin from zero

    def __len__(self):
        return len(self.image_ids)


if __name__ == '__main__':
    ds = FlowerDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
