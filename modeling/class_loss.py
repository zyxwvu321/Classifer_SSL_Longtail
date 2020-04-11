# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]

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

        # no weight no fl
        if self.weight is None:
            self.weight = torch.ones(num_classes,dtype = torch.float32)

            self.fl_gamma =  torch.zeros(num_classes,dtype = torch.float32)
            
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

