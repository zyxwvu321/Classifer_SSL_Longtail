from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'

# Name of Model
_C.MODEL.NAME = 'resnet50_SingleView'



# Name of backbone
_C.MODEL.BACKBONE = 'resnet50'



# Last stride of backbone
#_C.MODEL.LAST_STRIDE = 1

# Path to pretrained model
_C.MODEL.PRETRAIN_PATH = ''

# Path to pretrained model of backbone
_C.MODEL.BACKBONE_PRETRAIN_PATH = ''


# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

#N-View
_C.MODEL.N_VIEW = 1

_C.MODEL.VIEW_NAME = ['ap','lat']
# If train with BNNeck, options: 'bnneck' or 'no'
#_C.MODEL.NECK = 'bnneck'

# Model image part FCs
_C.MODEL.IMG_FCS = [4096,512]

# Model Final FC
_C.MODEL.FINAL_DIM = 256

# Meta Info
# Meta List


# Meta Dim
_C.MODEL.META_DIMS = [17,6,1,3]

# Meta FCs
_C.MODEL.META_FCS = [64,128]


# use CBAM
_C.MODEL.use_CBAM = False


# Loss
_C.MODEL.LOSS_TYPE = 'ce'



# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
#_C.MODEL.IF_WITH_CENTER = 'yes'

# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
#_C.MODEL.METRIC_LOSS_TYPE = 'center'

# For example, if loss type is cross entropy loss + triplet loss + center loss
# the setting should be: _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'

# If train with label smooth, options: 'yes', 'no'
_C.MODEL.IF_LABELSMOOTH = 'yes'

# If train with focal loss, options: 'yes', 'no'
_C.MODEL.IF_FOCAL_LOSS  = 'no'


# If train with margin based loss(e.g. RLL Loss), with one more loss item options: 'yes', 'no'
#_C.MODEL.IF_COSEMBED =  'no'

# If train with margin based loss(e.g. RLL Loss), with one more loss item options: 'yes', 'no'
#_C.MODEL.IF_STN =  'no'

# If train with arcface loss, use revised linear layers: 'yes', 'no'
_C.MODEL.IF_ARCFACE = 'no'

# If train with FC layers(256): 'yes', 'no'
_C.MODEL.USE_FC  = 'no'

# If train with class prediction also , predict dry/nor/wet : 'yes', 'no'
#_C.MODEL.PRED_CLS  = 'no' 

# Pooling mode: 'ave', 'max', 'maxave', now only support dry/wet prediction
#_C.MODEL.POOL_MODE = 'ave'

# If train with debug, plot mean/std of input and feature output: : 'yes', 'no'
_C.MODEL.IF_DEBUG = 'no'


#DB
_C.MODEL.DB_PEAKDROP = 0.5

_C.MODEL.DB_PATCHDROP = 0.25

_C.MODEL.DB_PATCHG = 2

_C.MODEL.DB_ALPHA = 0.1

# att

_C.MODEL.ATT_PARTS = 8
_C.MODEL.LOSS_ATT = False


# ADL
_C.MODEL.USE_ADL  = False
_C.MODEL.ADL_POSITION = ['40','42']#['30','40','42']

_C.MODEL.ADLTHR  = 0.90 
_C.MODEL.ADLRATE  = 0.5


_C.MODEL.POOLMODE ='maxavg'
_C.MODEL.MIXUP = False
_C.MODEL.MIXUP_MODE = 'mixup' #('mixup', 'cutmix')
_C.MODEL.MIXUP_ALPHA = 0.2
_C.MODEL.MIXUP_PROB = 1.0 # maybe by setting mixup prob to 0.5 and centerloss is applied whenmixup is off



# reid bnnneck
_C.MODEL.REID_PDROP_LIN = 0.0
_C.MODEL.REID_USE_FC = True
_C.MODEL.REID_NECK_FEAT = 'after' # before after 
_C.MODEL.REID_KCENTER = 0.0
_C.MODEL.REID_CENTERFEATNORM = True

# if  SSL enable
_C.MODEL.SSL_FIXMATCH = False
_C.MODEL.PSEUDO_TH = 0.95


# cls layer

_C.MODEL.CLS_LAYER = 'FC' # 'FCNorm'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# Size of the image during training
_C.INPUT.SIZE_TRAIN_IN = [384,384]

# Size of the image after augmentation, the input of model 
_C.INPUT.SIZE_TRAIN_PRED = [384,512]
_C.INPUT.SIZE_TRAIN_ATT  = [256,256]

# Size of the image during test
#_C.INPUT.SIZE_TEST = [176,176]

# Random probability for image horizontal flip
_C.INPUT.PROB = 0.0

# Random probability for random erasing
_C.INPUT.RE_PROB = 0.0

# Random probability for random erasing corner
_C.INPUT.REC_PROB = 0.0

# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]

# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# augmentation of the image
_C.INPUT.CRP_SCALE = [0.08,1.0]
_C.INPUT.CRP_SCALE_WEAK = [0.5,1.0]
_C.INPUT.CRP_RATIO = [0.75,1.33]
_C.INPUT.MINMAX_H  = [318,450]
_C.INPUT.W2H_RATIO = 1.33333


# use fine-grained transform 
_C.INPUT.USE_FGTFMS  = False


# augmentation of the attention cropped image
#_C.INPUT.CRP_SCALE_ATT = [0.7,1.0]
#_C.INPUT.CRP_RATIO_ATT = [0.75,1.33]
#_C.INPUT.MINMAX_H_ATT  = [224,320]
#_C.INPUT.W2H_RATIO_ATT = 1.33333


# Value of padding size
#_C.INPUT.PADDING = 10

# Random probability for rotation
#_C.INPUT.ROT_PROB = 0.0

# Max rotation angle(uniform distribution, (-deg,deg))
#_C.INPUT.ROT_DEG = 45

# if rotated, rot90 prob
#_C.INPUT.ROT90_PROB = 0.0

# if ElasticTransform is added
#_C.INPUT.ELASTIC_PROB = 0.0


# if use albu toolbox for augmentation
#_C.INPUT.TRANS_ALBU = 'yes'

# if use one channel input and train from scratch
#_C.INPUT.IF_ONE_CHANNEL = 'no'



# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()

# List of the dataset names for training
_C.DATASETS.NAMES = 'ISIC'

# Root directory where datasets should be used 
_C.DATASETS.ROOT_DIR = ['../data/all18_coloradj']

# Root directory where valid datasets should be used , '' None
_C.DATASETS.ROOT_DIR_VALID = ''

# label mapping
_C.DATASETS.DICT_LABEL = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'] 

# label weights
_C.DATASETS.LABEL_W = 1

# dataset infos
_C.DATASETS.INFO_CSV = ['./dat/all18_info.csv']

#_C.DATASETS.FN_MAP   = './dat/fn_maps_ISIC18.pth'

_C.DATASETS.NUM_CLASS = 7

_C.DATASETS.META_LIST = ['age','pos','sex','color_gain']

# K-Fold
_C.DATASETS.K_FOLD = 1

# split pecterage 
_C.DATASETS.PCT  = 0.8

# sample weight of unlabel data
_C.DATASETS.SAMPLERATIO_UNLABEL = 1.0
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# Sampler for data loading
_C.DATALOADER.SAMPLER = 'uniform' #['uniform' 'imbalance'] 

#BATCH_SIZE
_C.DATALOADER.BATCH_SIZE = 16



# Number of instance for one batch
#_C.DATALOADER.NUM_INSTANCE = 4

# Sampling mode
#_C.DATALOADER.RANDSAMPLER = 'RandomIdentitySampler'




# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# Name of optimizer
_C.SOLVER.OPTIMIZER = 'SGD'

# Base learning rate
_C.SOLVER.BASE_LR = 1e-3

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005

# MOMENTUM
_C.SOLVER.MOMENTUM = 0.9 

# If Stagelr 
_C.SOLVER.STAGED_LR = False

# backbone mult
_C.SOLVER.BASE_MULT =  0.1

# backbone names
_C.SOLVER.BACKBONE_NAMES = ['backbone']


# scheduler
_C.SOLVER.SCHEDULER = "OneCycle" #['OneCycle']

# Number of max epoches
_C.SOLVER.EPOCHS = 60

# Start Epoch, if continue training ,set this 
_C.SOLVER.START_EPOCH  = 1

# att
_C.SOLVER.ALPHA_CENTER = 0.05


# see 2 images per batch
_C.TEST = CN()

# Number of images per batch during test
_C.TEST.BATCH_SIZE = 128

# If test with re-ranking, options: 'yes','no'
#_C.TEST.RE_RANKING = 'no'

# Path to trained model
#_C.TEST.WEIGHT = ""

# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'

# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'



# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

_C.MISC = CN()
# Seed
_C.MISC.SEED = 0

# Valid Epoch
_C.MISC.VALID_EPOCH = 10

# Save Epoch
_C.MISC.SAV_EPOCH = 10

# iteration of display training log
_C.MISC.LOG_PERIOD = 1000

# Log File Name
_C.MISC.LOGFILE = 'log.txt'

# evallog
_C.MISC.LOGFILE_EVAL = 'log_eval.txt'

# Output Dir
_C.MISC.OUT_DIR =  "../checkpoints/log_d"
    
# If Only Test
_C.MISC.ONLY_TEST = False

# If TTA
_C.MISC.TTA = False

# If TTA
_C.MISC.N_TTA = 1

  

# Start Fold for Training
_C.MISC.START_FOLD = 0
