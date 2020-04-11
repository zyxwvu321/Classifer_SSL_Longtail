# encoding: utf-8
"""
@ Init Dataset
"""



from .ISIC_Dataset import ISIC_withmeta
from .BoneXray_Dataset import BoneXrayDataset_withmeta
from .bird_dataset import BirdDataset
from .flower_dataset import FlowerDataset

def init_dataset(cfg):
    ds_name = cfg.DATASETS.NAMES
    if 'ISIC' in ds_name:
        #dataset = CustomDataset_withmeta(root = cfg.DATASETS.ROOT_DIR, meta_list= cfg.DATASETS.META_LIST , dict_label = cfg.DATASETS.DICT_LABEL,\
        #      info_csv = cfg.DATASETS.INFO_CSV, fn_map =cfg.DATASETS.FN_MAP,is_test = cfg.MISC.ONLY_TEST)
        dataset = ISIC_withmeta(root = cfg.DATASETS.ROOT_DIR, meta_list= cfg.DATASETS.META_LIST , dict_label = cfg.DATASETS.DICT_LABEL,\
              info_csv = cfg.DATASETS.INFO_CSV, is_test = cfg.MISC.ONLY_TEST)

        
    elif 'BoneXray' in ds_name:
        dataset = BoneXrayDataset_withmeta(root = cfg.DATASETS.datasets, info_csv = cfg.DATASETS.INFO_CSV,is_test = cfg.MISC.ONLY_TEST)
    
    elif 'bird' in ds_name:
        # already imclude transform and train valid split
        dataset =  [BirdDataset(phase='train', resize=cfg.INPUT.SIZE_TRAIN_PRED), BirdDataset(phase='val', resize=cfg.INPUT.SIZE_TRAIN_PRED)]
    elif 'flower' in ds_name:
         
        dataset =  [FlowerDataset(phase='train', resize=cfg.INPUT.SIZE_TRAIN_PRED), FlowerDataset(phase='val', resize=cfg.INPUT.SIZE_TRAIN_PRED)]
    else: 
        raise KeyError("Unknown datasets: {}".format(ds_name))

    return dataset
