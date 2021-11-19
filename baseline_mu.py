#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:00:18 2021

@author: user1
"""
import torch
import torch.nn as nn
from typing import Dict
import matplotlib.pyplot as plt
from other_layers import Identity

class Baseline(nn.Module):
    """
    The parent class of exogenous intensity function mu(t), which actually a constant exogenous intensity
    """
    def __init__(self, num_type: int):

        super(Baseline, self).__init__()
        self.num_type = num_type
        self.dim_embedding = 1
        self.emb = nn.Embedding(self.num_type, self.dim_embedding)
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


    def forward(self, sample_dict: Dict):

        mu_c = self.intensity(sample_dict)
        #mU = self.expect_counts(sample_dict)
        return mu_c

    def intensity(self, sample_dict):

        events = sample_dict['ci']  
        events.view(-1,1)
        mu_c_list = []
        for i in range(self.num_type):
            idx = torch.tensor([i])
            mu_c =self.emb(idx)
            mu_c = mu_c.squeeze(1)   
            mu_c_list.append(mu_c)
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