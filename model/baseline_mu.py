#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:55:04 2021

@author: user1
"""

"""
This script contains a parent class of exogenous intensity function mu(t).
"""

import torch
import torch.nn as nn
from typing import Dict
from dev.util import logger
import matplotlib.pyplot as plt
from model.OtherLayers import Identity

class Baseline_rate(nn.Module):
    """
    The parent class of exogenous intensity function mu(t), which actually a constant exogenous intensity
    """
    def __init__(self, num_type: int):

        super(Baseline_rate, self).__init__()
        #super(BasicExogenousIntensity, self).__init__(num_type)
        #activation = parameter_set['activation']
        #if activation is None:
        self.exogenous_intensity_type = 'constant'
        self.activation = 'identity'
        #else:
            #self.exogenous_intensity_type = '{}(constant)'.format(activation)
            #self.activation = activation

        self.num_type = num_type
        self.dim_embedding = 1
        self.emb = nn.Embedding(self.num_type, self.dim_embedding)
        #print('emb:',self.emb)
        #self.emb.weight = nn.Parameter(
                       #torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                     #1 / self.dim_embedding))
        self.emb.weight = nn.Parameter(
                       torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                     1))
# =============================================================================
#         self.emb.weight = nn.Parameter(torch.FloatTensor([[0.1423],
#         [0.0860],
#         [0.0806],
#         [0.1427]]))
# =============================================================================

        self.act = Identity()
        #else:
            #logger.warning('The actvation layer is {}, which can not be identified... '.format(self.activation))
            #logger.warning('Identity activation is applied instead.')
            #self.act = Identity()

    def print_info(self):
        """
        Print basic information of the exogenous intensity function.
        """
        logger.info('Exogenous intensity function: mu(t) = {}.'.format(self.exogenous_intensity_type))
        logger.info('The number of event types = {}.'.format(self.num_type))

    def forward(self, sample_dict: Dict):

        mu_c = self.intensity(sample_dict)
        #mU = self.expect_counts(sample_dict)
        return mu_c

    def intensity(self, sample_dict):

        events = sample_dict['ci']  
        #print(events)# (batch_size, 1)
        events.view(-1,1)
        #mu_c = self.act(self.emb[1])
        #print('----------------')
        mu_c_list = []
        for i in range(self.num_type):
            idx = torch.tensor([i])
            #print(self.emb(idx))
            mu_c =self.emb(idx)
            #mu_c = self.act(self.emb(events))  # (batch_size, 1, 1)
            mu_c = mu_c.squeeze(1)   
            mu_c_list.append(mu_c)
            # (batch_size, 1)
            #print(mu_c)
        return mu_c_list

    def expect_counts(self, sample_dict):

        dts = sample_dict['ti'] - sample_dict['tjs'][:, -1].view(-1, 1)
        all_types = sample_dict['Cs']  # (num_type, 1)
        mu_all = self.act(self.emb(all_types))  # (num_type, 1, 1)
        mu_all = mu_all.squeeze(1)  # (num_type, 1)
        mU = torch.matmul(dts, torch.t(mu_all))  # (batch_size, num_type)
        return mU

    def plot_and_save(self, mu_all: torch.Tensor, output_name: str = None):

        mu_all = mu_all.squeeze(1)  # (C,)
        mu_all = mu_all.data.cpu().numpy()
        xx = ['Append','Extend','Mutation']

        plt.figure(figsize=(10, 10))
        #plt.stem(range(mu_all.shape[0]), mu_all, '-')
        plt.stem(xx, mu_all, '-')
        plt.ylabel('Exogenous intensity')
        plt.xlabel('Index of event type')
        plt.xticks(xx,xx, rotation='vertical')
        plt.tight_layout()  
        if output_name is None:
            plt.savefig('exogenous_intensity.png')
        else:
            plt.savefig(output_name,dpi=300)
        plt.close("all")
        logger.info("Done!")
