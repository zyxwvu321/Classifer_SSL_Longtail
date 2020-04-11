
from .models import ISICModel_singleview,ISICModel_singleview_meta,SplitBoneModel_meta,ISICModel_sv_att,ISICModel_sv_db, \
    ISICModel_singleview_reid,SV_BNN

import torch

def build_model(cfg):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    
    if 'SVBNN' in cfg.MODEL.NAME:
        model = SV_BNN(n_class = cfg.DATASETS.NUM_CLASS, arch = cfg.MODEL.BACKBONE, cls_layer = cfg.MODEL.CLS_LAYER)
        
        
    elif 'SingleView' in cfg.MODEL.NAME:
        model = ISICModel_singleview(n_class = cfg.DATASETS.NUM_CLASS, arch = cfg.MODEL.BACKBONE,use_CBAM = cfg.MODEL.use_CBAM)

    elif 'SVMeta' in cfg.MODEL.NAME:
        model = ISICModel_singleview_meta(n_class = cfg.DATASETS.NUM_CLASS, arch = cfg.MODEL.BACKBONE,use_CBAM = cfg.MODEL.use_CBAM)
    
    elif 'MVMeta' in cfg.MODEL.NAME:
        model = SplitBoneModel_meta(n_class = cfg.DATASETS.NUM_CLASS, arch = cfg.MODEL.BACKBONE,use_CBAM = cfg.MODEL.use_CBAM)

    
    elif 'SVAtt' in cfg.MODEL.NAME:
        #model = resnet50(pretrained=True,use_bap=True)
        model = ISICModel_sv_att(n_class = cfg.DATASETS.NUM_CLASS, arch = cfg.MODEL.BACKBONE)

    elif 'SVDB' in cfg.MODEL.NAME:
        model = ISICModel_sv_db(n_class = cfg.DATASETS.NUM_CLASS, arch = cfg.MODEL.BACKBONE)
    
    elif 'SVreid' in cfg.MODEL.NAME:
        model = ISICModel_singleview_reid(n_class = cfg.DATASETS.NUM_CLASS, arch = cfg.MODEL.BACKBONE)
    else:
        raise ValueError('unknown model type { cfg.MODEL.NAME}')

    use_cuda = torch.cuda.is_available()
    if cfg.MODEL.DEVICE != 'cuda':
        use_cuda = False

    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    return model

    




"""     use_stn = True if cfg.MODEL.IF_STN in  ['on','yes'] else False
    use_arcFace = True if cfg.MODEL.IF_ARCFACE in  ['on','yes'] else False
    use_fc = True if cfg.MODEL.USE_FC in  ['on','yes'] else False
    pred_cls = True if cfg.MODEL.PRED_CLS in  ['on','yes'] else False
    
    pdrop_lin = cfg.SOLVER.PDROP_LIN
    
    if_debug = True if cfg.MODEL.IF_DEBUG in  ['on','yes'] else False
    print(f'stn = {use_stn}')
    print(f'arcFace = {use_arcFace}')
    print(f'pred_cls = {pred_cls}')    
    print(f'pdrop_lin = {pdrop_lin}')    
    print(f'if_debug = {if_debug}')
    
    
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                     use_stn = use_stn,use_arcFace = use_arcFace,use_fc = use_fc,pred_cls = pred_cls,pdrop_lin = pdrop_lin,if_debug = if_debug)
     """
    