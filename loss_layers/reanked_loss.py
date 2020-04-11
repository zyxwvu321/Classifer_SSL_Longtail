# encoding: utf-8
"""
@author:  zzg
@contact: xhx1247786632@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F
import globalvar as gl
def normalize_rank(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def rank_loss(dist_mat, labels, margin,alpha,tval):
    """ 
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]  0.6,1.414,10.0,
      
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)


    total_ap = 0.0
    total_an = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])
        
        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]
        
        ap_is_pos = torch.clamp(torch.add(dist_ap,margin-alpha),min=0.0)
        ap_pos_num = ap_is_pos.size(0) +1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum,float(ap_pos_num))

        an_is_pos = torch.lt(dist_an,alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval*(-1*an_less_alpha+alpha))
        an_weight_sum = torch.sum(an_weight) +1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm,an_weight))
        loss_an = torch.div(an_ln_sum,an_weight_sum)
        total_ap = total_ap +loss_ap
        total_an = total_an +loss_an
    
    loss_ap = loss_ap*250.0/N
    loss_an = loss_an*250.0/N
    total_loss = loss_ap+loss_an
    writer = gl.get_value('writer')
    #writer.add_scalar("rll_ap", loss_ap.item())
    #writer.add_scalar("rll_an", loss_an.item())

    writer.add_scalars('rll_apan', {'rll_ap': loss_ap.item(),
                                    'rll_an': loss_an.item()})

    writer.add_scalar("rll_total", total_loss.item())
            
    return total_loss

class RankedLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    
    def __init__(self, margin=None, alpha=None, tval=None,p_drop = 0.0):
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        
        self.p_drop = p_drop
        if self.p_drop>0:    
            self.m_drop = nn.Dropout2d(p = self.p_drop) 
    def __call__(self, global_feat, labels, normalize_feature=True):
        if normalize_feature:
            global_feat = F.normalize(global_feat,p=2,dim=1)   #normalize_rank(global_feat, axis=-1)
        if self.p_drop>0:    
            global_feat0 = global_feat.permute(1,0)[None,:]
            global_feat0 = self.m_drop(global_feat0)[0].permute((1,0)) *  self.p_drop
            
        else:
            global_feat0 = global_feat
        

        
        dist_mat = euclidean_dist_rank(global_feat0, global_feat0)
        total_loss = rank_loss(dist_mat,labels,self.margin,self.alpha,self.tval)
        
        return total_loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

