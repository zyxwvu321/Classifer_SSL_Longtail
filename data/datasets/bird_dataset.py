""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
import os

from PIL import Image
from torch.utils.data import Dataset
#import torchvision.transforms as transforms

#from data.transforms.RandAugment.augmentations import RandAugment
from data.transforms.build import get_transform


DATAPATH = '../data/bird'
image_path = {}
image_label = {}


#def get_transform(resize, phase='train'):
#    if phase == 'train':
#        tfms =  transforms.Compose([
#            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
#            transforms.RandomCrop(resize),
#            transforms.RandomHorizontalFlip(0.5),
#            transforms.ColorJitter(brightness=0.126, saturation=0.5),
#            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#        ])
#    
#        
#        # Add RandAugment with N, M(hyperparameter)
#        tfms.transforms.insert(1, RandAugment(2, 9))
#        return tfms
#    else:
#        return transforms.Compose([
#            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
#            transforms.CenterCrop(resize),
#            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#        ])


class BirdDataset(Dataset):
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

    def __init__(self, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 200

        # get image path from images.txt
        with open(os.path.join(DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)

        # get train/test image id from train_test_split.txt
        with open(os.path.join(DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        image = Image.open(os.path.join(DATAPATH, 'images', image_path[image_id])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, image_label[image_id] - 1  # count begin from zero

    def __len__(self):
        return len(self.image_id)


if __name__ == '__main__':
    ds = BirdDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
