from torch.autograd import Function
import torch
from typing import Optional
from torch.distributions.gamma import Gamma
import numpy as np

import math

from torch import Tensor, Size

from abc import ABC, abstractmethod

# def percentile(t, q):
#     k = 1 + round(float(q) * (t.numel() - 1))
#     return t.view(-1).kthvalue(k).values.item()


def percentile(t, q):
    k = 1 + np.floor(q * (t.numel() - 1))
    sorted_t = t.view(-1).sort().values
    values = sorted_t[k]
    return values

class GetSubnet(Function):
    @staticmethod
    def forward(ctx, score, kthvalue, scaling):
        zeros = torch.zeros_like(score)
        ones = torch.ones_like(score)

        ctx._scaling = scaling
        return torch.where(
                score < kthvalue, zeros.to(score.device), ones.to(score.device)
            )

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx._scaling, None, None