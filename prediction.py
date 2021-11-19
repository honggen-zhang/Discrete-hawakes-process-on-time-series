#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:00:48 2021

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
cluster_file = 'cluster_file/IP_C3_KG'
loading_model_path = 'meta_full_IP_c3.pt'

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
        kernel_para[1, 0] = 0.25
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
        
        
        # load model
        hawkes_model.load_model(loading_model_path, mode='entire')
        print('--------------------------')
        
        ss = windows[i][0]+3
        ee = windows[i][1]
        y_hat_all = []
        y_true_all = []
        mu, A = hawkes_model.parameters_list()
        #hawkes_model.plot_exogenous_ave(mu,output_name='/home/user1/Desktop/ddhp/paper_figures/mu/IP_c1'+'.pdf')
        #hawkes_model.plot_causality_ave(A,output_name='/home/user1/Desktop/ddhp/paper_figures/A/JS_c3'+'.pdf')
        for tt in range(6,ee+1):
            #new_data, y_true= hawkes_model.simulate1(history=database,memory_size=memory_size,window_s = ss-3, window_e = 47, t_now = tt, t_max = window_end, use_cuda=False)
            new_data , y_true= hawkes_model.simulate_uni(history=database,memory_size=memory_size,window_s = ss-3, window_e = 47, t_now = tt, use_cuda=False)
            app = 0
            ext = 0
            mut = 0
            for d in new_data:
                #print(d)
                app = app + d[0].item()
                ext = ext + d[1].item()
                mut = mut + d[2].item()
            haw = [app/len(new_data),ext/len(new_data),mut/len(new_data)]
            prediction_result[str(tt)] = haw
            
            app1 = 0
            ext1 = 0
            mut1 = 0
            #mut = 0
            for d in y_true:
                #print(d)
                app1 = app1 + d[0].item()
                ext1 = ext1 + d[1].item()
                mut1 = mut1 + d[2].item()
            haw1 = [app1/len(y_true),ext1/len(new_data),mut1/len(new_data)] 
            true_result[str(tt)] = haw1

'''
            if tt == ee:
                new_data1 = new_data
                y_true1 = y_true
                
        error1 = []
        error2 = []
        error3 = []
        e = 0.0001
        for i in range(len(new_data1)):
            #print(new_data1[i])
            pre = np.squeeze(new_data1[i])[0]
            #pre = new_data1[i].squeeze()[0]
            tru = y_true1[i][0].numpy()[0]
            #tru = y_true1[i].squeeze()[0]
            if tru ==0:
                tru = e
            e1 = np.abs(pre-tru)
            e2 = (pre-tru)**2
            e3 = np.abs(pre)+np.abs(tru)
            error1.append(e1)
            error2.append(e2)
            error3.append(e1/e3)
        #print(np.average(error1));print(np.average(error2));
        print(np.average(error3)*2) 

        error1 = []
        error2 = []
        error3 = []
        e = 0.0001
        y_2 = []
        for i in range(len(new_data1)):
            #print(new_data1[i])
            pre = np.squeeze(new_data1[i])[1]
            #pre = new_data1[i].squeeze()[0]
            tru = y_true1[i][1].numpy()[0]
            y_2.append(tru)
            #tru = y_true1[i].squeeze()[0]
            if tru ==0:
                tru = e
            e1 = np.abs(pre-tru)
            e2 = (pre-tru)**2
            e3 = np.abs(pre)+np.abs(tru)
            error1.append(e1)
            error2.append(e2)
            error3.append(e1/e3)
        #print(np.average(error1));print(np.average(error2));
        print(np.average(error3)*2)  
        error1 = []
        error2 = []
        error3 = []
        e = 0.0001
        
        for i in range(len(new_data1)):
            #print(new_data1[i])
            pre = np.squeeze(new_data1[i])[2]
            #pre = new_data1[i].squeeze()[0]
            tru = y_true1[i][2].numpy()[0]
            #tru = y_true1[i].squeeze()[0]
            if tru ==0:
                tru = e
            e1 = np.abs(pre-tru)
            e2 = (pre-tru)**2
            e3 = np.abs(pre)+np.abs(tru)
            error1.append(e1)
            error2.append(e2)
            error3.append(e1/e3)
        #print(np.average(error1));print(np.average(error2));
        print(np.average(error3)*2)  
        print('------------------------')
'''
