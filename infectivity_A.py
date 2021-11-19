#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:00:46 2021

@author: user1
"""
import torch
import torch.nn as nn
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from other_layers import Identity

class Infectivity(nn.Module):

    def __init__(self, num_type: int, kernel):

        super(Infectivity, self).__init__()

        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.num_type = num_type
        self.dim_embedding = num_type
        for m in range(1):
            emb = nn.Embedding(self.num_type, self.dim_embedding)
            emb.weight = nn.Parameter(
                           torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                         1 / self.dim_embedding))
# =============================================================================
#             emb.weight = nn.Parameter(torch.FloatTensor([[0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8874, 0.6937, 0.1289, 0.0639],
#         [1.5959, 0.0000, 1.5273, 0.2063],
#         [1.5943, 0.0000, 0.5338, 1.4565]]))
# =============================================================================
            if m == 0:
                self.basis = nn.ModuleList([emb])
            else:
                self.basis.append(emb)

        self.act = Identity()


    def intensity(self, sample_dict: Dict):

        event_time = sample_dict['ti']     # (batch_size, 1)
        history_time = sample_dict['tjs']  # (batch_size, memory_size)
        events = sample_dict['ci']         # (batch_size, 4)
        history = sample_dict['cjs']       # (batch_size, memory_size)

        dts = event_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)                               # (batch_size, memory_size, num_base)
        gt = self.decay_kernel.values(dts)

        phi_c_list = []
        for m in range(self.num_type):
            A_cm = self.basis[0](torch.tensor([m]))                        # (batch_size, 1, dim_embedding)
            
            A_cm = A_cm.squeeze(1)
            A_cm_t = A_cm.view(-1,1)
            A_cm_t.type(torch.LongTensor)
            history_t = history.squeeze(0)
            history_t.type(torch.FloatTensor)
            history_t = history_t.float()
            phi_c = torch.mm(history_t, A_cm_t)
            phi = torch.mm(gt, phi_c)
            #print(phi)
            phi_c_list.append(phi)

        return phi_c_list

    def forward(self, sample_dict: Dict):

        phi_c = self.intensity(sample_dict)
        #pHi = self.expect_counts(sample_dict)
        return phi_c
    def expect_counts(self, sample_dict: Dict):

        event_time = sample_dict['ti']     # (batch_size, 1)
        history_time = sample_dict['tjs']  # (batch_size, memory_size)
        history = sample_dict['cjs']       # (batch_size, memory_size)
        all_types = sample_dict['Cs']      # (num_type, 1)

        dts = event_time.repeat(1, history_time.size(1)) - history_time     # (batch_size, memory_size)
        last_time = history_time[:, -1].unsqueeze(1)
        t_start = last_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        t_stop = dts                                                        # (batch_size, memory_size)
        # Gt = self.decay_kernel.integrations(t_stop.numpy(), t_start.numpy())
        # Gt = torch.from_numpy(Gt)
        # Gt = Gt.type(torch.FloatTensor)                                     # (batch_size, memory_size, num_base)
        Gt = self.decay_kernel.integrations(t_stop, t_start)

        pHi = 0
        history2 = history.unsqueeze(1).repeat(1, all_types.size(0), 1)     # (batch_size, num_type, memory_size)
        for m in range(self.num_base):
            A_all = self.basis[m](all_types)                    # (num_type, 1, dim_embedding)
            A_all = A_all.squeeze(1).unsqueeze(0)               # (1, num_type, dim_embedding)
            A_all = A_all.repeat(Gt.size(0), 1, 1)              # (batch_size, num_type, dim_embedding)
            A_all = A_all.gather(2, history2)                   # (batch_size, num_type, memory_size)
            A_all = self.act(A_all)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))   # (batch_size, num_type, 1)
        pHi = pHi[:, :, 0]
        return pHi

    def granger_causality(self, sample_dict: Dict):

        all_types = sample_dict['Cs']  # (num_type, 1)
        A_all = 0
        for m in range(self.num_base):
            A_tmp = self.basis[m](all_types)  # (num_type, 1, num_type)
            A_tmp = self.act(torch.transpose(A_tmp, 1, 2))
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all



    def plot_and_save(self, infect: torch.Tensor, output_name: str = None):

        xx = ['Append','Extend','Mutate']
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        impact = infect.numpy()
        #print(impact)
        #plt.figure(figsize=(10, 10))
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.rcParams['image.cmap'] = 'Oranges'
        im = ax.imshow(impact,alpha = 0.6,vmin=0,vmax=1.)
        ax.set_xticks(np.arange(len(xx)))
        ax.set_yticks(np.arange(len(xx)))
        ax.set_xticklabels(xx,fontsize=20)
        ax.set_yticklabels(xx,fontsize=20)
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(3+1)-.5, minor=True)
        ax.set_yticks(np.arange(3+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        for i in range(len(xx)):
            for j in range(len(xx)):               
                text = ax.text(j, i, np.around(impact[i, j],3),fontsize=20, ha="center", va="center", color="k")
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)

        #plt.colorbar(im, cax=cax)
        #plt.show()
        #plt.colorbar()
        plt.tight_layout()  
        if output_name is None:
            plt.savefig('endogenous_impact.png')
        else:
            plt.savefig(output_name,dpi=200)
        plt.close("all")

