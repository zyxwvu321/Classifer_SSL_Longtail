# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import globalvar as gl


##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)



def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]

class CrossEntropyLoss_labelsmooth(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.
    
    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by
    
    .. math::
        \begin{equation}
        (1 - \epsilon) \times y + \frac{\epsilon}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\epsilon` is a weight. When
    :math:`\epsilon = 0`, the loss function reduces to the normal cross entropy.
    
    Args:
        num_classes (int): number of classes.
        epsilon (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """
    
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True, label_w = None):
        super(CrossEntropyLoss_labelsmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        #revise for weighted version
        if label_w is not None:
            label_w_inv = 1.0/label_w
            self.label_w = torch.diag(label_w *(1.0- epsilon + epsilon/num_classes))
            
            sum_all = torch.sum(label_w_inv)
            
            num_n   = torch.zeros_like(label_w)
            for cc in range(num_classes):
                num_n[cc] = sum_all - label_w_inv[cc]
            
            k_n = epsilon*(num_classes-1.0)/num_classes
            
            w_n = k_n* 1.0/num_n
            
            
            self.label_w = (1.0 - torch.eye(num_classes)) * w_n.repeat((num_classes,1)) + self.label_w
            if use_gpu is  True:
                self.label_w =  self.label_w.cuda()
                
        else:
            self.label_w = label_w

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        
        if self.label_w is None:
            targets_onehot = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            if self.use_gpu: 
                targets_onehot = targets_onehot.cuda()
            
            
            targets_onehot = (1 - self.epsilon) * targets_onehot + self.epsilon / self.num_classes
            return (- targets_onehot * log_probs).mean(0).sum()
            
        else:
            label_w =  self.label_w.index_select(dim=0,index= targets)
            return (- log_probs * label_w).mean(0).sum()
            
#        if self.label_w is not None:
#            ww   = self.label_w[None,:].type_as(log_probs)
#
#            
#            return (- targets_onehot * log_probs * ww).mean(0).sum()
#        else:
#            
            
        
#class BCE_Loss(nn.Module):
#    def __init__(self, num_classes):
#        super().__init__()
#        self.num_classes = num_classes
#
#    def forward(self, pred, targ):
#        t = one_hot_embedding(targ, self.num_classes)
#        t = torch.Tensor(t[:,1:].contiguous()).cuda()
#        x = pred[:,1:]
#        w = self.get_weight(x,t)
#        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)#/(self.num_classes)
#    
#    def get_weight(self,x,t): return None

def pcsoftmax(inputs,weight,dim,use_gpu=True):
    assert isinstance(weight, torch.Tensor), 'weight is not tensor'
    p_cls = 1.0/weight
    p_cls = (p_cls/p_cls.sum())
    if use_gpu is True:
       p_cls =  p_cls.cuda()
    x_exp = torch.exp(inputs)
    weighted_x_exp = x_exp * p_cls
        # weighted_x_exp = x_exp
    x_exp_sum = torch.sum(weighted_x_exp, dim=dim, keepdim=True)

    probs = x_exp / (x_exp_sum  + 1e-10)
    #probs = probs/(probs.sum()+ 1e-10)
    return probs



class PCsoftmaxCE(nn.Module):
    
    def __init__(self, num_classes,weight,use_gpu=True):
        # include focal-loss and label smmooth
        super(PCsoftmaxCE, self).__init__()
        assert isinstance(weight, torch.Tensor), 'weight is not tensor'
        
        p_cls = 1.0/weight
        
        self.p_cls = (p_cls/p_cls.sum())
        if use_gpu is True:
            self.p_cls =  self.p_cls.cuda()
        self.num_classes = num_classes
        
        self.crit = nn.NLLLoss()

        
    def forward(self, inputs, targets):
        x_exp = torch.exp(inputs)
        weighted_x_exp = x_exp * self.p_cls
        # weighted_x_exp = x_exp
        x_exp_sum = torch.sum(weighted_x_exp, 1, keepdim=True)

        inputs_adj = torch.log(x_exp / (x_exp_sum+ 1e-10)  + 1e-6)
        

        
        return self.crit(inputs_adj,targets)
        #torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        
        
       


class BCElLoss(nn.Module):
    def __init__(self, num_classes,weight = None, alpha = 0.5,mult_val = 1.0):

        super(BCElLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.mult_val = mult_val
        self.alpha = alpha
    def forward(self, pred, targ):
        
        #tt = one_hot_embedding(targ, self.num_classes)
        #tt = torch.Tensor(tt.contiguous()).cuda()
#        with torch.no_grad():
   
        tt = torch.zeros_like(pred).scatter_(1,targ.unsqueeze(1), 1.0)
        
        pos_weight = torch.ones((1,self.num_classes)).cuda() * (self.num_classes-1) * self.alpha * 2
        if self.weight is None:
        
            return self.mult_val *F.binary_cross_entropy_with_logits(pred, tt,  reduction='mean', pos_weight=pos_weight )#/(self.num_classes)
        else:
            pos_weight = pos_weight * self.weight
            return self.mult_val *F.binary_cross_entropy_with_logits(pred, tt,   reduction='mean',pos_weight=pos_weight)#/(self.num_classes)

    

class FocalLoss(nn.Module):
    def __init__(self, num_classes,weight = None, use_gpu=True):
        # include focal-loss and label smmooth
        super(FocalLoss, self).__init__()

        self.num_classes = num_classes
        self.weight = weight
        self.softmax = nn.Softmax(dim=1)
        self.use_gpu = use_gpu

        # default
        if self.weight is None:
            self.weight = torch.ones(num_classes,dtype = torch.float32)

            self.fl_gamma =  2*torch.ones(num_classes,dtype = torch.float32)
            
        else:
            # it's numpy
            # for weight<1.0 est gamma
            self.fl_gamma =  torch.zeros(num_classes,dtype = torch.float32)
            for idx, ww in enumerate(self.weight):
                if ww<=1:
                    fl_gamma = np.floor(-np.log2(ww))
                    self.fl_gamma[idx] = fl_gamma
                    if fl_gamma>0:
                        self.weight[idx] = self.weight[idx] *pow(2.0,fl_gamma/2.0)
            self.weight = torch.from_numpy(self.weight).float()

        if use_gpu is True:
            self.weight = self.weight.cuda()
            self.fl_gamma = self.fl_gamma.cuda()
        
    def forward(self, pred, targ):
#        if self.use_gpu: 
#            targ = targ.cuda()
        probs = self.softmax(pred)
        #targets_onehot = torch.zeros_like(log_probs).scatter_(1, targ[:,None], 1)

        targ_prob = torch.gather(input = probs,dim=1,index = targ[:,None]).squeeze(1)

        
        fl_gamma = self.fl_gamma.index_select(dim = 0, index = targ)
        fl_weight = self.weight.index_select(dim = 0, index = targ)
        
        
        return torch.mean(-fl_weight * (1.0 - targ_prob+0.001).pow(fl_gamma) * torch.log(targ_prob + 0.001))
        
class Att_CELoss(nn.Module):

    
    def __init__(self, crit,num_classes):
        super(Att_CELoss, self).__init__()
        self.num_classes = num_classes
        
        self.crit = crit
        self.center_loss =  CenterLoss()
        
        

    def forward(self, inputs, targets):

        y_pred_raw,  y_pred_crop, y_pred_drop, feature_matrix,feature_center_batch = inputs
        loss1 = self.crit(y_pred_raw, targets)
        loss2 = self.crit(y_pred_crop, targets)
        loss3 = self.crit(y_pred_drop, targets)
        loss_center = self.center_loss(feature_matrix,feature_center_batch)

        loss = (loss1+loss2+loss3)/3.0 + loss_center
        loss_dict = {'global':loss1,'crop':loss2,'drop':loss3,'center':loss_center}
        return (loss,loss_dict)






class REID_CELoss(nn.Module):

    
    def __init__(self, crit,num_classes,k_center):
        super(REID_CELoss, self).__init__()
        self.num_classes = num_classes
        
        self.crit = crit
        self.center_loss =  CenterLoss()
        self.k_center = k_center
        

    def forward(self, inputs, targets):
        #if self.training is True:
        cls_score,global_feat,feature_center_batch = inputs
        loss_cls = self.crit(cls_score, targets)
    
        loss_center = self.center_loss(global_feat,feature_center_batch)
        loss = loss_cls + self.k_center*loss_center
        loss_dict = {'global':loss_cls,'center':loss_center}
        return (loss,loss_dict)

       # else:

            
class FixMatch_CELoss(nn.Module):

    
    def __init__(self, crit,num_classes,pseudo_th = 0.95):
        super(FixMatch_CELoss, self).__init__()
        self.num_classes = num_classes
        
        self.crit = crit
        
        self.pseudo_th = pseudo_th
#        cfg = gl.get_value('cfg')
#        if not isinstance(cfg.DATASETS.LABEL_W, int):
#            label_w = np.array(cfg.DATASETS.LABEL_W).astype('float32')
#            label_w = torch.tensor(label_w).float()
#        else:
#            label_w = None
#        self.label_w = label_w.cuda()

    def forward(self, inputs,  targets):
        #if self.training is True:
        
        mask_label = targets !=-1
        #targets[targets==-1] = -100 # CE has a default ignore_index = -100
        
        n_label = mask_label.sum()
        n_pseudo = inputs[0].size(0) - n_label
        
        
        inputs_s,inputs_w = inputs
        
        
        #with label
        Lx = (self.crit(inputs_s, targets) * mask_label.float()).sum()
        
        
        pseudo_label = torch.softmax(inputs_w, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.pseudo_th).float()
        
        mask_usepseudo = mask * (1.0 - mask_label.float())
        
        n_usepseudo = mask_usepseudo.sum()
        
        Lu = (self.crit(inputs_s, targets_u) *mask_usepseudo).sum()
        
#        Lu = (F.cross_entropy(inputs_s, targets_u,weight = self.label_w,
#                              reduction='none') * mask_usepseudo).sum()
        
        hist_usepseudo = np.bincount(targets_u[mask_usepseudo.bool()].cpu().numpy(),minlength = self.num_classes)
        
        
        loss = (Lx + Lu)/ inputs[0].size(0)
    
        
        
        
        loss_dict = {'n_usepseudo':n_usepseudo.item(),'n_pseudo':n_pseudo.item() ,'hist_usepseudo':hist_usepseudo}
        
        
        
        return (loss,loss_dict)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
#class FocalLoss(nn.Module):
#    def __init__(self, num_classes,weight = None, alpha = 0.5,mult_val = 1.0):
#        super(FocalLoss, self).__init__()
#
#        self.num_classes = num_classes
#        self.weight = weight
#        self.mult_val = mult_val
#        self.alpha = alpha
#    def forward(self, pred, targ):
#        
##        tt = one_hot_embedding(targ, self.num_classes)
##        tt = torch.Tensor(tt.contiguous()).cuda()
#        tt = torch.zeros_like(pred).scatter_(1,targ.unsqueeze(1), 1.0)
#        ww = self.get_weight(pred,tt)
#        
#        pos_weight = torch.ones((1,self.num_classes)).cuda() * (self.num_classes-1) * self.alpha * 2
#        if self.weight is None:
#            return self.mult_val  *F.binary_cross_entropy_with_logits(pred, tt, ww, reduction='mean',pos_weight=pos_weight)#/(self.num_classes)
#        else:
#            pos_weight = pos_weight * self.weight
#        
#            return self.mult_val  *F.binary_cross_entropy_with_logits(pred, tt, ww, reduction='mean',pos_weight=pos_weight)#/(self.num_classes)
#    def get_weight(self,xx,tt):
#        alpha,gamma = 0.5,2.0
#        pp = torch.sigmoid(xx)
#        pt = pp*tt + (1-pp)*(1-tt)
#        ww = alpha*tt + (1-alpha)*(1-tt)
#        
#       # ww = alpha*(1.0-tt) + (1.0-alpha)*tt
#        ww = ww * (1-pt).pow(gamma)
#        return ww.detach()
#    

#class FocalLoss(nn.Module):
#    r"""
#        This criterion is a implemenation of Focal Loss, which is proposed in 
#        Focal Loss for Dense Object Detection.
#
#            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#
#        The losses are averaged across observations for each minibatch.
#
#        Args:
#            alpha(1D Tensor, Variable) : the scalar factor for this criterion
#            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
#                                   putting more focus on hard, misclassiﬁed examples
#            size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                However, if the field size_average is set to False, the losses are
#                                instead summed for each minibatch.
#
#
#    """
#    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#        super(FocalLoss, self).__init__()
#        if alpha is None:
#            self.alpha = Variable(torch.ones(class_num, 1))
#        else:
#            if isinstance(alpha, Variable):
#                self.alpha = alpha
#            else:
#                self.alpha = Variable(alpha)
#        self.gamma = gamma
#        self.class_num = class_num
#        self.size_average = size_average
#
#    def forward(self, inputs, targets):
#        N = inputs.size(0)
#        C = inputs.size(1)
#        P = F.softmax(inputs,dim = 1)
#
#        class_mask = inputs.data.new(N, C).fill_(0)
#        class_mask = Variable(class_mask)
#        ids = targets.view(-1, 1)
#        class_mask.scatter_(1, ids.data, 1.)
#        #print(class_mask)
#
#
#        if inputs.is_cuda and not self.alpha.is_cuda:
#            self.alpha = self.alpha.cuda()
#        alpha = self.alpha[ids.data.view(-1)]
#
#        probs = (P*class_mask).sum(1).view(-1,1)
#
#        log_p = probs.log()
#        #print('probs size= {}'.format(probs.size()))
#        #print(probs)
#
#        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
#        #print('-----bacth_loss------')
#        #print(batch_loss)
#
#
#        if self.size_average:
#            loss = batch_loss.mean()
#        else:
#            loss = batch_loss.sum()
#        return loss

