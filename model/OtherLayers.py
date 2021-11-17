"""
This script contains some classes used in our models
"""

import numpy as np
import torch
import torch.nn as nn
import math

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LowerBoundClipper(object):
    def __init__(self, threshold):
        self.bound = threshold

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w[w < self.bound] = self.bound


class MaxLogLike(nn.Module):
    """
    The negative log-likelihood loss of events of point processes
    nll = sum_{i in batch}[ -log lambda_ci(ti) + sum_c Lambda_c(ti) ]
    """
    def __init__(self):
        super(MaxLogLike, self).__init__()
        self.eps = float(np.finfo(np.float32).eps)
        self.constant = self.eps+1

    def forward(self, lambda_t, event_num):

        lambda_t_all = 0
        for i in range(len(event_num[0])):
            #print(event_num[i])
            num = event_num[0][i]
            num = torch.tensor(num)
            fa = math.factorial(int(num))
            cons = math.log(fa)
            
            lambda_t_all = lambda_t_all - torch.mul(num, (lambda_t[i]+self.eps).log())+ (lambda_t[i]+self.eps)+cons
        return lambda_t_all







