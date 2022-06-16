
# Import the backbone registry from mmcls
import torch
from torch.autograd import grad
from mmcv.cnn import CONV_LAYERS
from torch import nn
import torch.nn.functional as F
from .custom_grad import GetSubnet, percentile
import numpy as np
import math

@CONV_LAYERS.register_module('MultiLevelHN')
class MultiLevelHN(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros', sparsity=0.0, grad_type = 'STE', bw_scale=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        self.score = nn.Parameter(torch.Tensor(self.weight.size()))

        self.sparsity = np.array(sparsity)

        self.weight.requires_grad = False
        self.weight *= math.sqrt(1/self.sparsity[0])

        self.scale = nn.Parameter(torch.Tensor([1]))
        self.scale.requires_grad = False

        self.bw_scale = bw_scale
    
    def set_kthvalues(self):
        with torch.no_grad():
            self.kthvalues = percentile(self.score.abs(), self.sparsity)

    def forward(self, x):
        mask = 0.
        for k in self.kthvalues:
            mask = mask + GetSubnet.apply(self.score.abs(), k, self.bw_scale)

        w = self.weight * mask 
        w = w * self.scale
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


@CONV_LAYERS.register_module('IndependentScore')
class IndependentScore(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros', sparsity=0.0, grad_type = 'STE', bw_scale=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        self.scores = nn.ParameterList([nn.Parameter(torch.Tensor(self.weight.size())) for _ in range(len(sparsity))])

        self.sparsity = np.array(sparsity)

        self.weight.requires_grad = False
        self.weight *= math.sqrt(1/self.sparsity[0])

        self.scale = nn.Parameter(torch.Tensor([1]))
        self.scale.requires_grad = False

        self.bw_scale = bw_scale
    
    def set_kthvalues(self):
        with torch.no_grad():
            kthvalues = []
            for score, sparsity in zip(self.scores, self.sparsity):
                kthvalues.append(percentile(score.abs(), np.array([sparsity])).item())
            self.kthvalues = torch.Tensor(kthvalues).to(self.weight.device)

    def forward(self, x):
        mask = 0.
        for i, (k, _) in enumerate(zip(self.kthvalues, self.scores)):
            mask = mask + GetSubnet.apply(self.scores[i], k, self.bw_scale)

        w = self.weight * mask 
        w = w * self.scale
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
