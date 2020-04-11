from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


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
            
        
