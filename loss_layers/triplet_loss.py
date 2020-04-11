# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
import globalvar as gl
from torch.nn import functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
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

def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())

    return distmat
def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def hard_example_mining_cosemb(dist_mat, labels, cids):#, return_inds=False):
    """For each anchor, find hard positive and negative sample.
    using cosine embedding formula
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos_lb = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg_lb = labels.expand(N, N).ne(labels.expand(N, N).t())
    
    
    is_pos_cid = cids.expand(N, N).eq(cids.expand(N, N).t())
    is_neg_cid = cids.expand(N, N).ne(cids.expand(N, N).t())
    
    
    #is_diff_im = (1.0 - torch.eye(N)).type_as(is_pos_lb)
    is_diff_im = (torch.ones((N,N)).triu(diagonal=1)).type_as(is_pos_lb)
    
    
#    dist_plpc = dist_mat[is_pos_lb & is_pos_cid & is_diff_im].reshape(N,-1) #same finger same status
#    
#    dist_plnc = dist_mat[is_pos_lb & is_neg_cid & is_diff_im].reshape(N,-1) #same finger diff status
#    
#    dist_nlpc = dist_mat[is_neg_lb & is_pos_cid & is_diff_im].reshape(N,-1) #diff finger same status
#    
#    dist_nlnc = dist_mat[is_neg_lb & is_neg_cid & is_diff_im].reshape(N,-1) #diff finger diff status
#    
#
#    n_plpc, n_plnc, n_nlpc,n_nlnc =  1,2,3,6   #1,4,14,28
#    th_plpc,th_plnc, th_nlpc,th_nlnc  = -0.25, -0.5, 0.0 , 0.0
#    
#    dist_plpc = dist_plpc.topk(n_plpc,dim=1,largest = True)[0]
#    dist_plnc = dist_plnc.topk(n_plnc,dim=1,largest = True)[0]
#    dist_nlpc = dist_nlpc.topk(n_nlpc,dim=1,largest = False)[0]
#    dist_nlnc = dist_nlnc.topk(n_nlnc,dim=1,largest = False)[0]
#
#    dist_plpc = torch.clamp(dist_plpc +th_plpc, min =  0.0)
#    dist_plnc = torch.clamp(dist_plnc +th_plnc, min =  0.0)
#    
#    dist_nlpc = torch.clamp(1.0-dist_nlpc +th_nlpc, min =  0.0)
#    dist_nlnc = torch.clamp(1.0-dist_nlnc +th_nlnc, min =  0.0)
#    
    
    dist_plpc = dist_mat[is_pos_lb & is_pos_cid & is_diff_im].contiguous() #same finger same status
    
    dist_plnc = dist_mat[is_pos_lb & is_neg_cid & is_diff_im].contiguous() #same finger diff status
    
    dist_nlpc = dist_mat[is_neg_lb & is_pos_cid & is_diff_im].contiguous() #diff finger same status
    
    dist_nlnc = dist_mat[is_neg_lb & is_neg_cid & is_diff_im].contiguous() #diff finger diff status
    

    n_plpc, n_plnc, n_nlpc,n_nlnc =  1,2,3,6   #1,4,14,28
    th_plpc,th_plnc, th_nlpc,th_nlnc  = -0.1, -0.25, 0.0 , 0.0
    
    dist_plpc = dist_plpc.topk(min(N*n_plpc//2,len(dist_plpc)),dim=0,largest = True)[0]
    dist_plnc = dist_plnc.topk(min(N*n_plnc//2,len(dist_plnc)),dim=0,largest = True)[0]
    dist_nlpc = dist_nlpc.topk(min(N*n_nlpc//2,len(dist_nlpc)),dim=0,largest = False)[0]
    dist_nlnc = dist_nlnc.topk(min(N*n_nlnc//2,len(dist_nlnc)),dim=0,largest = False)[0]

    dist_plpc = torch.clamp(dist_plpc +th_plpc, min =  0.0)
    dist_plnc = torch.clamp(dist_plnc +th_plnc, min =  0.0)
    
    dist_nlpc = torch.clamp(1.0-dist_nlpc +th_nlpc, min =  0.0)
    dist_nlnc = torch.clamp(1.0-dist_nlnc +th_nlnc, min =  0.0)

    #return dist_plpc.sum(dim=1), dist_plnc.sum(dim=1),dist_nlpc.sum(dim=1),dist_nlnc.sum(dim=1)
    return dist_plpc.sum(), dist_plnc.sum(),dist_nlpc.sum(),dist_nlnc.sum()


class CosEmbedLoss(object):


    def __init__(self, margin=0.0):
        self.margin = margin
        self.ranking_loss =nn.CosineEmbeddingLoss(margin=margin, reduction='mean')
      

    def __call__(self, feat_afterbn, labels, cids):

        #feat_afterbn = normalize(feat_afterbn, axis=-1)
        
        #dist_mat = euclidean_dist(feat_afterbn, feat_afterbn)
        
        dist_mat = cosine_distance(feat_afterbn, feat_afterbn)
        dist_plpc, dist_plnc,dist_nlpc,dist_nlnc = hard_example_mining_cosemb(dist_mat, labels, cids)
        

        N = dist_mat.size(0)
#        loss_p1 = dist_plpc.mean()
#        loss_p2 = dist_plnc.mean()
#        loss_n1 = dist_nlpc.mean()
#        loss_n2 = dist_nlnc.mean()
        loss_p1 = dist_plpc/N*2
        loss_p2 = dist_plnc/N*2
        loss_n1 = dist_nlpc/N*2
        loss_n2 = dist_nlnc/N*2
        
        loss_n = loss_p1 +  loss_p2 + loss_n1 +loss_n2
        
        writer = gl.get_value('writer')
        writer.add_scalar("cos_p1", loss_p1.item())
        writer.add_scalar("cos_p2", loss_p2.item())
        writer.add_scalar("cos_n1", loss_n1.item())
        writer.add_scalar("cos_n2", loss_n2.item())
        writer.add_scalar("cos_emb", loss_n.item())
        
     
        return loss_n

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None,p_drop = 0.0):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        self.p_drop = p_drop
        if self.p_drop>0:    
            self.m_drop = nn.Dropout2d(p = self.p_drop) 
    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            
        # add dropout for triple loss   
        if self.p_drop>0:    
            global_feat0 = global_feat.permute(1,0)[None,:]
            global_feat0 = self.m_drop(global_feat0)[0].permute((1,0)) *  self.p_drop
            dist_mat = euclidean_dist(global_feat0, global_feat0)
        else:
            dist_mat = euclidean_dist(global_feat, global_feat)
            
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        
        
        
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        
        writer = gl.get_value('writer')
        writer.add_scalar("dist_ap", dist_ap.mean().item())
        writer.add_scalar("dist_an", dist_an.mean().item())
        #writer.add_scalars("dist_apan", {"dist_ap": dist_ap.mean().item(),
        #                            "dist_an": dist_an.mean().item()})


        writer.add_scalar("loss_tri", loss.item())
        return loss, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, pred_stat = False):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.pred_stat = pred_stat

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets[:,None], 1)
        #targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
#        if self.use_gpu: 
#            targets = targets.cuda()
            
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        writer = gl.get_value('writer')
        
        if self.pred_stat is False:
        
            writer.add_scalar("loss_ces", loss.item())
        else:
            writer.add_scalar("loss_ces_cls", loss.item())
        return loss


class FocalLossLabelSigmoid(nn.Module):
    """Focal loss with Sigmoid

    Reference:
    
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.0, gamma = 5.0, gamma_p = 1.0,alpha = 10.0):
        super(FocalLossLabelSigmoid, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        self.gamma = gamma
        self.gamma_p = gamma_p
        self.alpha = alpha # for positive *alpha
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        probs = torch.sigmoid(inputs)

        targets = torch.zeros_like(probs).scatter_(1, targets[:,None], 1)
        
        
        #pt = probs*targets + (1.0-probs)*(1.0-targets)


        pt_p = probs*targets 
        pt_n =  (1.0-probs)*(1.0-targets)


        #targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        ww_p = -( (1-pt_p)**self.gamma_p ) 
        ww_n = -( (1-pt_n)**self.gamma ) 


        focal_loss_p = ww_p * torch.log(probs) *targets *self.alpha *0.1

        focal_loss_n = ww_n * torch.log(1.0 - probs) * (1.0-targets)*10.0



        loss_p = focal_loss_p.mean(0).sum()
        loss_n = focal_loss_n.mean()
        loss = loss_p + loss_n
        #loss = (- targets * log_probs).mean(0).sum()
        
        writer = gl.get_value('writer')
        writer.add_scalar("loss_fls_sigmoid_p", loss_p.item())
        writer.add_scalar("loss_fls_sigmoid_n", loss_n.item())
        writer.add_scalar("loss_fls_sigmoid", loss.item())
        return loss


class FocalLossLabelSmooth(nn.Module):
    """Focal loss with label smoothing regularizer.

    Reference:
    
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.0, gamma = 2.0,use_gpu=True):
        super(FocalLossLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.gamma = gamma
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)




        targets = torch.zeros_like(log_probs).scatter_(1, targets[:,None], 1)
        #targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
#        if self.use_gpu: 
#            targets = targets.cuda()

        pp = torch.exp(log_probs)
        pt = pp*targets + (1-pp)*(1-targets)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        focal_loss = -( (1-pt)**self.gamma ) * log_probs *targets
        loss = focal_loss.mean(0).sum()
        #loss = (- targets * log_probs).mean(0).sum()
        
        writer = gl.get_value('writer')
        writer.add_scalar("loss_fls", loss.item())
        return loss