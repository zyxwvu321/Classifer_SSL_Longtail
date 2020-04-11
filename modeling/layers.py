
"""
Defines layers used in models.py.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3
from modeling.cbam import CBAM


#from constants import VIEWS
import math

class InvertedResidual_Pool(nn.Module):
    def __init__(self, inp, oup, feat_sz, expand_ratio,  onnx_compatible=False):
        super(InvertedResidual_Pool, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        hidden_dim = round(inp * expand_ratio)
        
        
        self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),                    
                    ReLU(inplace=True),
                    nn.BatchNorm2d(hidden_dim),
                    
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, feat_sz, 1, 0, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    nn.BatchNorm2d(hidden_dim),
                    
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            

    def forward(self, x):
        return self.conv(x)
    
    
    
class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape):
        super(OutputLayer, self).__init__()
        if not isinstance(output_shape, (list, tuple)):
            output_shape = [output_shape]
        self.output_shape = output_shape
        self.flattened_output_shape = int(np.prod(output_shape))
        self.fc_layer = nn.Linear(in_features, self.flattened_output_shape)

    def forward(self, x):
        h = self.fc_layer(x)
        if len(self.output_shape) > 1:
            h = h.view(h.shape[0], *self.output_shape)
        h = F.log_softmax(h, dim=-1)
        return h


class BasicBlockV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out



class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {
            view_name: self._avg_pool(view_tensor)
            for view_name, view_tensor in x.items()
        }

    @staticmethod
    def _avg_pool(single_view):
        n, c, _, _ = single_view.size()
        return single_view.view(n, c, -1).mean(-1)

class AllViewsMaxAvgPool(nn.Module):
    """Max/ave-pool across all 2 views"""

    def __init__(self, twoview_name = ['ap','lat']):
        super(AllViewsMaxAvgPool, self).__init__()
        self.twoview_name = twoview_name

    def forward(self, x):

        result_pool = torch.cat([nn.AdaptiveMaxPool2d(1)(x[self.twoview_name[0]]), 
                                nn.AdaptiveAvgPool2d(1)(x[self.twoview_name[0]]),
                                nn.AdaptiveMaxPool2d(1)(x[self.twoview_name[1]]), 
                                nn.AdaptiveAvgPool2d(1)(x[self.twoview_name[1]]), 
                                ], dim=1).contiguous()
        return result_pool.view(result_pool.size(0),-1)
    
class AllViewsCBAMAvgPool(nn.Module):
    """Max/ave-pool across all 2 views"""

    def __init__(self, featdim, twoview_name = ['ap','lat']):
        super(AllViewsCBAMAvgPool, self).__init__()
        self.cbam_ap = CBAM(gate_channels =featdim//4)
        self.cbam_lat = CBAM(gate_channels =featdim//4)
        self.twoview_name = twoview_name
    def forward(self, x):

        result_pool = torch.cat([nn.AdaptiveAvgPool2d(1)(self.cbam_ap(x[self.twoview_name[0]])), 
                                 nn.AdaptiveAvgPool2d(1)(self.cbam_lat(x[self.twoview_name[1]])), 
                                 ], dim=1).contiguous()
    
        return result_pool.view(result_pool.size(0),-1)      
class MaxAvgPool(nn.Module):
    """Max/ave-pool """

    def __init__(self):
        
        super(MaxAvgPool, self).__init__()

    def forward(self, x):

        result_pool = torch.cat([nn.AdaptiveMaxPool2d(1)(x), 
                                nn.AdaptiveAvgPool2d(1)(x)], dim=1).contiguous()
        return result_pool.view(result_pool.size(0),-1)      

class AvgPool(nn.Module):
    """ave-pool """

    def __init__(self):
        
        super(AvgPool, self).__init__()

    def forward(self, x):

        result_pool = nn.AdaptiveAvgPool2d(1)(x).contiguous()
                                
        return result_pool.view(result_pool.size(0),-1)     
class SingleViewsMaxAvgPool(nn.Module):
    """Max/ave-pool across all 2 views"""

    def __init__(self):
        super(SingleViewsMaxAvgPool, self).__init__()

    def forward(self, x):

        result_pool = torch.cat([nn.AdaptiveAvgPool2d(1)(x),
                                nn.AdaptiveMaxPool2d(1)(x)],
                                dim=1).contiguous()
        return result_pool        

        
class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full:bool=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)






class RunningBatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('sums', torch.zeros(1,nf,1,1))
        self.register_buffer('sqrs', torch.zeros(1,nf,1,1))
        self.register_buffer('batch', torch.tensor(0.))
        self.register_buffer('count', torch.tensor(0.))
        self.register_buffer('step', torch.tensor(0.))
        self.register_buffer('dbias', torch.tensor(0.))

    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        s = x.sum(dims, keepdim=True)
        ss = (x*x).sum(dims, keepdim=True)
        c = self.count.new_tensor(x.numel()/nc)
        mom1 = 1 - (1-self.mom)/math.sqrt(bs-1)
        self.mom1 = self.dbias.new_tensor(mom1)
        self.sums.lerp_(s, self.mom1)
        self.sqrs.lerp_(ss, self.mom1)
        self.count.lerp_(c, self.mom1)
        self.dbias = self.dbias*(1-self.mom1) + self.mom1
        self.batch += bs
        self.step += 1

    def forward(self, x):
        if self.training: self.update_stats(x)
        sums = self.sums
        sqrs = self.sqrs
        c = self.count
        if self.step<100:
            sums = sums / self.dbias
            sqrs = sqrs / self.dbias
            c    = c    / self.dbias
        means = sums/c
        vars = (sqrs/c).sub_(means*means)
        if bool(self.batch < 20): vars.clamp_min_(0.01)
        x = (x-means).div_((vars.add_(self.eps)).sqrt())
        return x.mul_(self.mults).add_(self.adds)
    
    
    
from torch.nn.modules.utils import _pair

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        
    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

#
#batch_size = 5
#in_channels = 3
#h, w = 24, 24
#x = torch.randn(batch_size, in_channels, h, w)
#
#out_channels = 2
#output_size = 22
#kernel_size = 3
#stride = 1
#conv = LocallyConnected2d(
#    in_channels, out_channels, output_size, kernel_size, stride, bias=True)
#
#out = conv(x)
#
#out.mean().backward()
#print(conv.weight.grad)