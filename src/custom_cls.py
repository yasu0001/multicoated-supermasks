from mmcls.models.heads import ClsHead
from mmcls.models.builder import HEADS
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from .custom_grad import GetSubnet, percentile

@HEADS.register_module()
class CustomClsHead(ClsHead):
    def __init__(self, 
                num_classes,
                in_channels,
                sparsity,
                grad_type='STE',
                bw_scale=1.0,
                scale_grad=False,
                scale_sparse=False,
                init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                *args,
                **kwargs):
        super().__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sparsity = np.array(sparsity)

        if self.num_classes <= 0.:
            raise ValueError(f'num_classes={num_classes} must be a positive integer')

        self.fc = CustomLinear(self.in_channels, self.num_classes, self.sparsity, grad_type, bw_scale)

    def simple_test(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses

class CustomLinear(nn.Linear):
    def __init__(self, in_channels, num_classes, sparsity, grad_type = 'STE', bw_scale=1.0):
        super().__init__(in_channels, num_classes, bias=False)
        self.score = nn.Parameter(torch.Tensor(self.weight.size()))

        self.sparsity = sparsity

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

        return F.linear(x, w, self.bias)

@HEADS.register_module()
class CustomClsHead2(ClsHead):
    def __init__(self, 
                num_classes,
                in_channels,
                sparsity,
                grad_type='STE',
                bw_scale=1.0,
                scale_grad=False,
                scale_sparse=False,
                init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                *args,
                **kwargs):
        super().__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sparsity = np.array(sparsity)

        if self.num_classes <= 0.:
            raise ValueError(f'num_classes={num_classes} must be a positive integer')

        self.fc = CustomLinear2(self.in_channels, self.num_classes, self.sparsity, grad_type, bw_scale)

    def simple_test(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses

class CustomLinear2(nn.Linear):
    def __init__(self, in_channels, num_classes, sparsity, grad_type = 'STE', bw_scale=1.0):
        super().__init__(in_channels, num_classes, bias=False)
        self.scores = nn.ParameterList([nn.Parameter(torch.Tensor(self.weight.size())) for _ in range(len(sparsity))])

        self.sparsity = sparsity

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

        return F.linear(x, w, self.bias)