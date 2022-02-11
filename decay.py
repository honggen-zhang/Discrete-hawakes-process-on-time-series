#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:12:31 2021

@author: user1
"""
from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


class DecayKernel(object):
    """
    The parent class of decay functions, which actually an exponential kernel
    """
    def __init__(self, parameters: torch.Tensor):
        """
        Initialize decay functions
        exponential
                g(t) = w * exp(-w(t-delay)) if t>=delay, = 0 if t<delay
        :param parameters: the parameters related to decay kernel
                (2, 1) array for decay and bandwidth
        """
        self.kernel_type = 'Exponential'
        self.parameters = parameters


    def values(self, dt: torch.Tensor) -> torch.Tensor:
        delay = self.parameters[0, 0]
        bandwidth = self.parameters[1, 0]
        w = bandwidth  # ** 0.5
        #print('Beta:',w)

        dt2 = dt - delay
        # gt = w * np.exp(-w * dt2)
        gt = w * torch.exp(-w * dt2)
        gt[dt2 < 0] = 0
        gt2 = gt
        return gt2

    def integrations(self, t_stop: torch.Tensor, t_start: Optional[torch.Tensor] = None) -> torch.Tensor:

        if t_start is None:
            t_start = 0 * t_stop

        if t_start.size() != t_stop.size():
            t_start = 0 * t_stop

        delay = self.parameters[0, 0]
        bandwidth = self.parameters[1, 0]
        # w = np.sqrt(1 / bandwidth)
        w = torch.sqrt(1 / bandwidth)
        # gt_start = np.exp(-w * (t_start - delay))
        gt_start = (-w * (t_start - delay)).exp()
        gt_start[gt_start > 1] = 1
        # gt_stop = np.exp(-w * (t_stop - delay))
        gt_stop = (-w * (t_stop - delay)).exp()
        gt_stop[gt_stop > 1] = 1

        gt_d = gt_stop - gt_start
        gt = -gt_d.view(gt_d.size(0), gt_d.size(1), 1)
        return gt

    def plot_and_save(self, t_stop: float = 5.0, output_name: str = None):

        dt = np.arange(0.0, t_stop, 0.01)
        dt = np.tile(dt, (1, 1))
        dt = torch.from_numpy(dt)
        dt = dt.type(torch.FloatTensor)
        gt = self.values(dt)
        # t_start = torch.zeros(dt.size())
        igt = self.integrations(dt)
        # print(gt.shape)

        plt.figure(figsize=(5, 5))
        for k in range(gt.shape[2]):
            plt.plot(dt[0, :].cpu().numpy(), gt[0, :, k].cpu().numpy(), label='g_{}(t)'.format(k), c='r')
            plt.plot(dt[0, :].cpu().numpy(), igt[0, :, k].cpu().numpy(), label='G_{}(t)'.format(k), c='b')
        leg = plt.legend(loc='upper left', ncol=1, shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.title('{} decay kernel and its integration'.format(self.kernel_type))
        if output_name is None:
            plt.savefig('{}_decay_kernel.png'.format(self.kernel_type))
        else:
            plt.savefig(output_name,dpi=300)
        plt.close("all")

