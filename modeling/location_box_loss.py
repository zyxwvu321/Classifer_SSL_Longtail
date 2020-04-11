import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


    
def model_box_to_xy_ori(box,ori_im_shape,prior_w = 0.5, prior_cx=0.5, prior_cy=0.5,size_variance = 0.1,center_variance = 0.1):
    
    box[:,0] = (box[:,0] * center_variance +  prior_cx * 2.0)/2.0
    box[:,1] = (box[:,1] * center_variance +  prior_cy * 2.0)/2.0
    box[:,2] = torch.exp(box[:,2] * size_variance) * prior_w
    box[:,3] = torch.exp(box[:,3] * size_variance) * prior_w
    
    box_numpy = box.cpu().numpy()
    
    box_xyxy = np.zeros((box_numpy.shape[0],4),dtype = 'float32')
    
    
    
    
    box_xyxy[:,0] = box_numpy[:,0] - box_numpy[:,2]/2.0
    box_xyxy[:,1] = box_numpy[:,1] - box_numpy[:,3]/2.0
    box_xyxy[:,2] = box_numpy[:,0] + box_numpy[:,2]/2.0
    box_xyxy[:,3] = box_numpy[:,1] + box_numpy[:,3]/2.0
    
    return box_xyxy

def model_box_to_xy_ori_l1(box):
    
    
    
    box_numpy = box.cpu().numpy()
    
    box_xyxy = np.zeros((box_numpy.shape[0],4),dtype = 'float32')
    
    
    
    
    box_xyxy[:,0] = box_numpy[:,0] - box_numpy[:,2]/2.0
    box_xyxy[:,1] = box_numpy[:,1] - box_numpy[:,3]/2.0
    box_xyxy[:,2] = box_numpy[:,0] + box_numpy[:,2]/2.0
    box_xyxy[:,3] = box_numpy[:,1] + box_numpy[:,3]/2.0
    
    return box_xyxy


def norm_box_to_abs(box_norm,ori_im_shape):
    box = box_norm.copy()
    box[0] *=ori_im_shape[1]
    box[1] *=ori_im_shape[0]
    box[2] *=ori_im_shape[1]
    box[3] *=ori_im_shape[0]
    return box


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

    
    
def iou_ori(box_pred,box_gt,ori_im_shape):
    # one box input
    b_pred = torch.from_numpy(box_pred)
    b_gt = torch.from_numpy(box_gt)
    
    
    
    b_gt[:,0] *=ori_im_shape[1]
    b_gt[:,1] *=ori_im_shape[0]
    b_gt[:,2] *=ori_im_shape[1]
    b_gt[:,3] *=ori_im_shape[0]

    b_pred[:,0] *=ori_im_shape[1]
    b_pred[:,1] *=ori_im_shape[0]
    b_pred[:,2] *=ori_im_shape[1]
    b_pred[:,3] *=ori_im_shape[0]


    return iou_of(b_pred,b_gt)
    


#class Location_Box_Loss(nn.Module):
#    def __init__(self,prior_w = 0.5, prior_cx=0.5, prior_cy=0.5,size_variance = 0.1,center_variance = 0.1):
#        """Implement Loss of box locolization
#
#        smooth L1 regression loss.
#        """
#        super(Location_Box_Loss, self).__init__()
#        self.prior_w = prior_w
#        
#        self.prior_cx = prior_cx
#        self.prior_cy = prior_cy
#        
#        self.size_variance = size_variance
#        self.center_variance = center_variance
#
#    def forward(self, gt_box, predicted_vec):
#        """
#        # predicted_vec  x_center(-1,1)  /center_variance
#                         y_center(-1,1) /center_variance
#                         w ( -inf, inf)  0-1x, 1-2.7x -1 -0.36x  /size_variance
#        """
#        ww_gt = gt_box[:,2] - gt_box[:,0]
#        ww_gt_tr   = torch.log(ww_gt/self.prior_w)/self.size_variance
#        
#        hh_gt = gt_box[:,3] - gt_box[:,1]
#        hh_gt_tr   = torch.log(hh_gt/self.prior_w)/self.size_variance
#        
#        xx_gt_tr = ((gt_box[:,2] + gt_box[:,0]) - self.prior_cx * 2.0)/ self.center_variance   # (0.1) -> (-1,1)        
#        yy_gt_tr = ((gt_box[:,3] + gt_box[:,1]) - self.prior_cy * 2.0)/ self.center_variance   # (0.1) -> (-1,1)        
#        
#        loss_xx = F.smooth_l1_loss(predicted_vec[:,0], xx_gt_tr, size_average=True) 
#        loss_yy = F.smooth_l1_loss(predicted_vec[:,1], yy_gt_tr, size_average=True) 
#        
#        loss_ww = F.smooth_l1_loss(predicted_vec[:,2], ww_gt_tr, size_average=True) 
#        loss_hh = F.smooth_l1_loss(predicted_vec[:,3], hh_gt_tr, size_average=True) 
#        
#        
#        loss = loss_xx + loss_yy +  loss_ww + loss_hh
#        return loss,loss_xx,loss_yy,loss_ww,loss_hh

    


class Location_Box_Loss(nn.Module):
    def __init__(self,prior_w = 0.5, prior_cx=0.5, prior_cy=0.5,size_variance = 0.1,center_variance = 0.1):
        """Implement Loss of box locolization

        smooth L1 regression loss.
        """
        super(Location_Box_Loss, self).__init__()
        self.prior_w = prior_w
        
        self.prior_cx = prior_cx
        self.prior_cy = prior_cy
        
        self.size_variance = size_variance
        self.center_variance = center_variance

    def forward(self, gt_box, predicted_vec):
        """
        # predicted_vec  x_center(-1,1)  /center_variance
                         y_center(-1,1) /center_variance
                         w ( -inf, inf)  0-1x, 1-2.7x -1 -0.36x  /size_variance
        """
        ww_gt = gt_box[:,2] - gt_box[:,0] # (0,1)
        #ww_gt_tr   = torch.log(ww_gt/self.prior_w)/self.size_variance
        
        hh_gt = gt_box[:,3] - gt_box[:,1]
        #hh_gt_tr   = torch.log(hh_gt/self.prior_w)/self.size_variance
        
        xx_gt_tr = (gt_box[:,2] + gt_box[:,0])/2.0  #(0.1) 
        yy_gt_tr = (gt_box[:,3] + gt_box[:,1])/2.0   # (0.1)         
        
        #loss_xx = F.smooth_l1_loss(predicted_vec[:,0], xx_gt_tr, size_average=True) 
        #loss_yy = F.smooth_l1_loss(predicted_vec[:,1], yy_gt_tr, size_average=True) 
        
        #loss_ww = F.smooth_l1_loss(predicted_vec[:,2], ww_gt, size_average=True) 
        #loss_hh = F.smooth_l1_loss(predicted_vec[:,3], hh_gt, size_average=True) 
  
        
        loss_xx = F.l1_loss(predicted_vec[:,0], xx_gt_tr, size_average=True) 
        loss_yy = F.l1_loss(predicted_vec[:,1], yy_gt_tr, size_average=True) 
        
        loss_ww = F.l1_loss(predicted_vec[:,2], ww_gt, size_average=True) 
        loss_hh = F.l1_loss(predicted_vec[:,3], hh_gt, size_average=True)       
        
        loss = loss_xx + loss_yy +  loss_ww + loss_hh
        return loss,loss_xx,loss_yy,loss_ww,loss_hh



class Location_Box_CenterNet(nn.Module):
    def __init__(self):
        """Implement Loss of box locolization

        hm focal-loss xywh L1 regression loss.
        """
        super(Location_Box_CenterNet, self).__init__()


    def forward(self, gt_box, pred):
        
       #score
       hm = gt_box['hm'] 
       pred_s = pred[:,0:1,...]
       pos_inds = hm.eq(1).float()
       neg_inds = hm.lt(1).float()

       neg_weights = torch.pow(1.0 - hm, 4)
       pos_loss = torch.log(pred_s) * torch.pow(1 - pred_s, 2) * pos_inds
       neg_loss = torch.log(1 - pred_s) * torch.pow(pred_s, 2) * neg_weights * neg_inds

       num_pos  = pos_inds.float().sum()
       pos_loss = pos_loss.sum()
       neg_loss = neg_loss.sum()
       
       loss_s = -(pos_loss + neg_loss) / num_pos

  

       dense_xy = gt_box['dense_xy']
       dense_wh = gt_box['dense_wh']
       dense_mask = gt_box['dense_mask']
       pred_xy = pred[:,1:3,...]
       pred_wh = pred[:,3:,...]

     
       loss_xy = F.l1_loss(pred_xy*dense_mask,dense_xy*dense_mask,size_average=False) / num_pos
       loss_wh = F.l1_loss(pred_wh*dense_mask,dense_wh*dense_mask,size_average=False) / num_pos
       
       loss = loss_s + loss_xy + loss_wh
       return loss,loss_s,loss_xy,loss_wh


