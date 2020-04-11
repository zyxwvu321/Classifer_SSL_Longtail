from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class Custom_Classify_Loss(nn.Module):

    
    def __init__(self, cls_freq = None, use_focal_loss = False, use_imbal_samp = False, epsilon=0.1, label_smooth=True):
        super(Custom_Classify_Loss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return (- targets * log_probs).mean(0).sum()
