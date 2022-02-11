#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:39:25 2021

@author: user1
"""
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset import load_sequences_csv
from data_operation import EventSampler, enumerate_all_events, samples2dict
import os
from model_DHP import HawkesProcessModel
import pandas as pd
import matplotlib.pyplot as plt
# Custmize the window size and slide steps

def cluster_data(str):
    content_test = open(str, "r").readlines();
    cluster_k = []
    for triple in content_test[1:]:
        a = triple.split(',')
        head = a[0];
        tail = a[1]
        relation = a[2][:-1]
        cluster_k.append(head+'@'+relation+'@'+tail)
    return cluster_k


file_path = 'data/IP/data_label_norm/'
cluster_file = 'output/cluster_file/IP_C3_KG'
saving_model_path_full = 'output/IP_c3/full_'+str(11)+'.pt'
saving_model_path_pare = 'output/IP_c3/para_'+str(11)+'.pt'

#windows = [(2,26),(2,27),(2,28),(2,29),(2,30)]
#windows = [(3,41),(3,42),(3,43),(3,44),(3,45)]
windows = [(2,31)]
mu_all = []
A_all = []
loss_all = []
prediction_result ={}
true_result = {}
for i in range(len(windows)):
    # Read files window by window   
    all_files = []
    all_files.append('file_'+str(1)+'.csv') #selecting files
    all_files.append('file_'+str(2)+'.csv')

    print('window:',windows[i])
    print('Files need to read:', all_files)
    
    window_start = int(windows[i][0])
    window_end = int(windows[i][1])

    if __name__ == '__main__':
        # hyper-parameters
        cluster = cluster_data(cluster_file)
        memory_size = 3
        batch_size = 1
        use_cuda = False
        use_cuda = use_cuda and torch.cuda.is_available()
        seed = 1
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        epochs = 30
        # load event sequences from csv file
        domain_names = {'seq_id': 'id',
                        'time': 'time',
                        'event': 'event'}
        database = load_sequences_csv(file_path,all_files,window_start,window_end,cluster, domain_names=domain_names)   

        
        
        trainloader = DataLoader(EventSampler(database=database, memorysize=memory_size, start = window_start),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 **kwargs)

        
        num_type = 3
        kernel_para = np.zeros((2, 1))
        kernel_para[1, 0] = 0.5
        kernel_para = torch.from_numpy(kernel_para)
        kernel_para = kernel_para.type(torch.FloatTensor)
        kernel_dict = {'model_name': 'ExponentialKernel',
                       'parameter_set': kernel_para}
        #print(kernel_para)
        loss_type = 'mle'        
        hawkes_model = HawkesProcessModel(num_type=num_type,
                                          kernel_dict=kernel_dict,
                                          activation='identity',
                                          loss_type=loss_type,
                                          use_cuda=use_cuda)
        
        optimizer = optim.Adam(hawkes_model.lambda_model.parameters(), lr=0.1)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        
        
        # train model
        
        
        loss = hawkes_model.fit(trainloader, optimizer, epochs, scheduler=scheduler,
                         sparsity=10, nonnegative=0, use_cuda=use_cuda)
        plt.plot(loss)
        mu, A = hawkes_model.parameters_list()
        #print('mu:',mu)
        #print('A:',A)
        mu_all.append(mu)
        A_all.append(A)
        loss_all.append(loss)
        hawkes_model.save_model(saving_model_path_full, mode='entire')
        hawkes_model.save_model(saving_model_path_pare, mode='parameter')
