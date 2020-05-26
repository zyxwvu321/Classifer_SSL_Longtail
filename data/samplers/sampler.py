import torch
import torch.utils.data
import torchvision
import numpy as np

from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from collections import defaultdict

def count_label_freq(label, w_im):
    freqs = defaultdict(float)
    
    for lb,ww in zip(label,w_im):
        freqs[lb] += 1.0/ww
    return freqs

def resample_idx_with_meta(ds,ds_v, K_fold = 1,pct = 0.8,seed =123):
    # use meta info, if it is same, split to same train &valid part
    flist = ds.flist
    imlist = [Path(fn).stem for fn in flist]
    
    
    n_img = len(imlist)

    im_uids = ds.uids
    n_idx = im_uids.max() + 1
    
    
    if K_fold ==1:
        # 1-fold, split train and valid
        np.random.seed(seed)
        idx_all = np.random.permutation(n_idx)
        valid_ids = idx_all[int(round(n_idx*pct)):]
        
        idx_valid =np.array([idx    for idx in range(n_img)  if im_uids[idx] in valid_ids])
        idx_train = np.setdiff1d(np.arange(n_img),idx_valid)
        
        train_ds = torch.utils.data.Subset(ds,idx_train)
        valid_ds = torch.utils.data.Subset(ds_v,idx_valid)
        
        
        #Sampler_Train =  [SubsetRandomSampler(idx_train)]
        #Sampler_Valid =  [SubsetRandomSampler(idx_valid)]
        return train_ds, valid_ds
        #return Sampler_Train, Sampler_Valid
    else:
        # K-fold test
        train_ds =list()
        valid_ds =list()
        #Sampler_Train =list()
        #Sampler_Valid =list()

        
        idx_all = np.arange(n_idx) 
        kf = KFold(n_splits=K_fold,shuffle = True,random_state=seed)
        kf_indice = list(kf.split(idx_all))
        for n in range(K_fold):
            
            idx_valid = np.array([idx  for idx in range(n_img)  if im_uids[idx] in kf_indice[n][1]])
            idx_train = np.setdiff1d(np.arange(n_img),idx_valid)
            
            train_ds.append(torch.utils.data.Subset(ds,idx_train))
            valid_ds.append(torch.utils.data.Subset(ds_v,idx_valid))        
            
            #Sampler_Train.append(SubsetRandomSampler(idx_train))
            #Sampler_Valid.append(SubsetRandomSampler(idx_valid)) 
        
        return train_ds, valid_ds
        #return Sampler_Train, Sampler_Valid


    
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            return dataset.get_annotation(idx)

                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
        
