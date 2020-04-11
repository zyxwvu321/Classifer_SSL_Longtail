import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#__all__ = ['CBAMLayer', 'SPPLayer','DBLayer']

'''
    Woo et al., 
    "CBAM: Convolutional Block Attention Module", 
    ECCV 2018,
    arXiv:1807.06521
'''

def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))

    
class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    Credit: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return mish(input)

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # channel attention
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


'''
    He et al.,
    "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition",
    TPAMI 2015,
    arXiv:1406.4729
'''
class SPPLayer(nn.Module):
    def __init__(self, pool_size, pool=nn.MaxPool2d):
        super(SPPLayer, self).__init__()
        self.pool_size = pool_size
        self.pool = pool
        self.out_length = np.sum(np.array(self.pool_size) ** 2)

    def forward(self, x):
        B, C, H, W = x.size()
        for i in range(len(self.pool_size)):
            h_wid = int(math.ceil(H / self.pool_size[i]))
            w_wid = int(math.ceil(W / self.pool_size[i]))
            h_pad = (h_wid * self.pool_size[i] - H + 1) / 2
            w_pad = (w_wid * self.pool_size[i] - W + 1) / 2
            out = self.pool((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))(x)
            if i == 0:
                spp = out.view(B, -1)
            else:
                spp = torch.cat([spp, out.view(B, -1)], dim=1)
        return spp



# AAAI 2020 paper: Fine-grained Recognition: Accounting for Subtle Differences between Similar Classes
class DBLayer(nn.Module):
    def __init__(self, p_peak= 0.5, p_drop = 0.25,patch_g = 2,alpha = 0.1):
        super(DBLayer, self).__init__()

        self.p_peak = p_peak
        self.p_drop = p_drop
        self.patch_g = patch_g
        self.alpha  = alpha
        #self.gap = nn.AdaptiveAvePool2d(1)
        
    def forward(self, x,targets):

        B,C,H,W = x.size()

        for nB in range(B):
            mask_cls = x[nB][targets[nB]]
            mask_cls_max = (mask_cls == mask_cls.max()).float()

            mask_cls_max = mask_cls_max * (torch.rand_like(mask_cls_max)<=self.p_peak).float()
            mask_cls_max = mask_cls_max[None,...]
            #peak suppression
            x[nB] -= (1-self.alpha ) *x[nB] * mask_cls_max


            # patch suppression
            mask_patch = torch.rand((H//self.patch_g, W//self.patch_g)).type_as(x)<=self.p_drop


            mask_patch = F.interpolate(mask_patch.float()[None,None,...],size = (H,W),mode = 'nearest')[0]

            mask_patch =   (1.0 - mask_cls_max) * mask_patch.float()

            x[nB] -= (1-self.alpha ) *x[nB] * mask_patch
        
        #x = self.gap(x).reshape(x.size[0],-1)
        return x





class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-12)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix

#block use in resnet50-d
def get_downsample(inplanes, planes, expansion, stride):
    downsample = None
    if stride != 1:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * expansion),
            nn.AvgPool2d(kernel_size = stride, stride = stride)
        )

    elif inplanes != planes * expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * expansion)
        )
    return downsample
        
class Bottleneck(nn.Module):
    

    def __init__(self, inplanes, planes, stride=1, act = 'relu'):
        super(Bottleneck, self).__init__()
        expansion = 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = conv3x3(planes, planes, stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding= 1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = conv1x1(planes, planes * self.expansion)
        self.conv3 =  nn.Conv2d(planes, planes*expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        if act =='relu':
            self.relu = nn.ReLU(inplace=True)
        elif act == 'swish':
            self.relu = Mish()
            
        self.downsample = get_downsample(inplanes, planes, expansion, stride)
        self.stride = stride
        
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# for LDAM Loss
class FCNorm(nn.Module):
    def __init__(self, in_features, out_features):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.linear(F.normalize(x), F.normalize(self.weight))
        return out