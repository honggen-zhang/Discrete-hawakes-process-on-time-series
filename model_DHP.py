#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:59:29 2021

@author: user1
"""


import copy
import torch
import torch.nn as nn
from other_layers import Identity, LowerBoundClipper, MaxLogLike
from baseline_mu import Baseline
from infectivity_A import Infectivity
from decay import DecayKernel
import numpy as np
from data_operation import samples2dict
import time
import math
from random import sample



class HawkesProcessIntensity(nn.Module):
    """
    The class of inhomogeneous Poisson process
    """
    def __init__(self,
                 mu_model,
                 alpha_mode,
                 activation: str = None):
        super(HawkesProcessIntensity, self).__init__()
        self.baseline = mu_model
        self.infectivity = alpha_mode
        self.act = Identity()



    def forward(self, sample_dict):
        events = sample_dict['ci']  
        mu= self.baseline(sample_dict)
        #print('mu:',mu)
        alpha = self.infectivity(sample_dict)

        lambda_list = []
        for i in range(len(mu)):
            lambda_list.append(torch.add(mu[i], alpha[i]))
            
        return lambda_list,events
    def intensity(self, sample_dict):
        mu = self.baseline.intensity(sample_dict)
        alpha = self.infectivity.intensity(sample_dict)

        lambda_list = []
        for i in range(len(mu)):
            lambda_list.append(torch.add(mu[i], alpha[i]))
        return lambda_list


#class HawkesProcessModel(object):
class HawkesProcessModel(nn.Module):
    """
    The class of generalized Hawkes process model
    contains most of necessary function.
    """

    def __init__(self, num_type, kernel_dict, activation, loss_type, use_cuda):

        super(HawkesProcessModel, self).__init__()
        #super(HawkesProcessModel, self).__init__(num_type, mu_dict, alpha_dict, kernel_dict, activation, loss_type, use_cuda)
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.loss_type = loss_type
        #if self.loss_type == 'mle':
        self.loss_function = MaxLogLike()
        # self.num_type = num_type
        self.activation = activation# control mu and alpha if less than 0
        baseline = Baseline
        infectivity = Infectivity
        decayKernel = DecayKernel
        #print('ff----------------------------------------------------------------------------------: ',mu_dict['parameter_set'])

        mu_model = baseline(num_type)
        kernel_para = kernel_dict['parameter_set'].to(self.device)
        kernel_model = decayKernel(kernel_para)
        alpha_model = infectivity(num_type, kernel_model)
        #print('ff----------------------------------------------------------------------------------: ',alpha_model)

        self.lambda_model = HawkesProcessIntensity(mu_model, alpha_model, self.activation)


    def fit(self, dataloader, optimizer, epochs: int, scheduler=None, sparsity: float=None, nonnegative=None,
            use_cuda: bool=False, validation_set=None):

        device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.lambda_model.to(device)
        best_model = None
        self.lambda_model.train()

        if nonnegative is not None:
            clipper = LowerBoundClipper(nonnegative)

        #Cs = torch.LongTensor(list(range(4))) #mutivariate
        Cs = torch.LongTensor(list(range(4)))
        Cs = Cs.view(-1, 1)
        Cs = Cs.to(device)

        FCs = None

        if validation_set is not None:
            validation_loss = self.validation(validation_set, use_cuda)
            best_loss = validation_loss
        else:
            best_loss = np.inf
        loss_list = []

        for epoch in range(epochs):
            print(epoch)
            loss_all = 0
            if scheduler is not None:
                scheduler.step()
            start = time.time()
            for batch_idx, samples in enumerate(dataloader):
                ci, batch_dict = samples2dict(samples, device, Cs)

                optimizer.zero_grad()
                lambda_t, event_num = self.lambda_model(batch_dict)
                #print(event_num)
                loss = self.loss_function(lambda_t, event_num)
                #print(loss)
                loss_all = loss_all +loss
            reg = 0
            if sparsity is not None:
                for parameter in self.lambda_model.parameters():
                    print(parameter)
                    reg += sparsity * torch.sum(torch.abs(parameter))
            loss_total = loss_all + reg
            #print(self.lambda_model.state_dict())
            loss_total.backward()
            optimizer.step()
            if nonnegative is not None:
                self.lambda_model.apply(clipper)
            loss_list.append(loss_total.detach().numpy()[0][0]/len(list(enumerate(dataloader))))


        return loss_list

    def validation(self, dataloader, use_cuda):

        device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.lambda_model.to(device)
        self.lambda_model.eval()

        Cs = torch.LongTensor(list(range(len(dataloader.dataset.database['type2idx']))))
        Cs = Cs.view(-1, 1)
        Cs = Cs.to(device)

        if dataloader.dataset.database['event_features'] is not None:
            all_event_feature = torch.from_numpy(dataloader.dataset.database['event_features'])
            FCs = all_event_feature.type(torch.FloatTensor)
            FCs = torch.t(FCs)    # (num_type, dim_features)
            FCs = FCs.to(device)
        else:
            FCs = None

        #start = time.time()
        loss = 0
        for batch_idx, samples in enumerate(dataloader):
            ci, batch_dict = samples2dict(samples, device, Cs, FCs)
            lambda_t, Lambda_t = self.lambda_model(batch_dict)
            loss += self.loss_function(lambda_t, Lambda_t, ci).item()


        return loss/len(dataloader.dataset)


    def plot_mu(self,intensity,output_name: str = None):
        self.lambda_model.baseline.plot_and_save(intensity, output_name)

    def plot_alpha(self,infectivitys,output_name: str = None):
        
        self.lambda_model.infectivity.plot_and_save(infectivitys, output_name)
        
    def intensity(self, sample_dict):
        mu = self.lambda_model.exogenous_intensity.intensity(sample_dict)
        alpha = self.lambda_model.endogenous_intensity.intensity(sample_dict)
        lambda_t = self.act(mu + alpha)  # (batch_size, 1)
        print('uuuuuuuuuuuuuuuuuuuuuuu',mu)
        print('mu={}'.format(float(mu.sum())))
        print('alpha={}'.format(float(alpha.sum())))
        return lambda_t
    def save_model(self, full_path, mode: str='entire'):
        """
        Save trained model
        :param full_path: the path of directory
        :param mode: 'parameter' for saving only parameters of the model,
                     'entire' for saving entire model
        """
        if mode == 'entire':
            torch.save(self.lambda_model, full_path)
        elif mode == 'parameter':
            torch.save(self.lambda_model.state_dict(), full_path)
            #print(self.lambda_model.state_dict())
            print('The parameters of the model is saved in {}.'.format(full_path))

    def parameters_list(self):
        paras = self.lambda_model.state_dict().values()
        
        i = 0
        for par in paras:
            if i == 0:
                mu = par
            else:
                A = par
            i = i+1
        return mu, A

        
        

    def load_model(self, full_path, mode: str='entire'):

        if mode == 'entire':
            print(full_path)
            self.lambda_model = torch.load(full_path)
            for param_tensor in self.lambda_model.state_dict():
                print(param_tensor, "\t", self.lambda_model.state_dict()[param_tensor].size())
        elif mode == 'parameter':
            self.lambda_model.load_state_dict(torch.load(full_path))
        else:
            print("'{}' is a undefined mode, we use 'entire' mode instead.".format(mode))
            self.lambda_model = torch.load(full_path)
            
    def simulate_uni(self, history, memory_size: int = 3, window_s:int = 30, window_e:int = 39, t_now: int = 41, use_cuda: bool = False):

        device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.lambda_model.to(device)
        self.lambda_model.eval()
        #print(self.lambda_model.state_dict())

        #Cs = torch.LongTensor(list(range(1)))
        Cs = torch.LongTensor(list(range(3)))
        Cs = Cs.view(-1, 1)
        Cs = Cs.to(device)
        FCs = None

        t_start = time.time()
        new_data = copy.deepcopy(history)
        new_datas = []
        y_true = []
        # the number of new synthetic events for each type
        #counts = np.zeros((self.num_type, len(new_data['sequences'])))
        for i in range(len(new_data['sequences'])):
            times_tmp = []
            events_tmp = []
            ci = Cs
            # print(ci)
            ci = ci.to(device)
            ti = torch.FloatTensor([t_now])
            ti = ti.to(device)
            ti = ti.view(1, 1)
            #ti = ti.repeat(ci.size(0), 1)
            #print(ti)

            events = history['sequences'][i]['events']
            times = history['sequences'][i]['times']#[1,2,3,4,5]

            new_times = []
            new_events = []

            event_now = [0,0,0]
            #print(times)
            for tti in range(len(times)):
                if times[tti] < t_now:
                    new_times.append(times[tti])
                    new_events.append(list(events[tti]))
                if times[tti] == t_now:
                    event_now = list(events[tti])
            events = new_events
            times = new_times
            #print(events)
            if memory_size <= len(new_times):
                tjs = torch.from_numpy(np.array([times[-memory_size:]]))
                tjs = tjs.type(torch.FloatTensor)
                tjs = tjs.to(device)
                events_numpy = []
                event_tmp =events[-memory_size:]
                for e in event_tmp:
                    tmp = []
                    for es in e:
                        tmp.append(float(es))
                    events_numpy.append(tmp)

                ci = torch.from_numpy(np.array([[float(event_now[0])],[float(event_now[1])],[float(event_now[2])]]))
                ci = ci.to(device)
                cjs = torch.from_numpy(np.array([events_numpy]))
                cjs = cjs.type(torch.LongTensor)
                cjs = cjs.to(device)    
                tjs = tjs.to(device)
                
                sn = torch.LongTensor([i])
                sn = sn.to(device)
  
                sample_dict = {'ti': ti,
                               'tjs': tjs,
                               'ci': ci,
                               'cjs': cjs,
                               'sn': sn,
                               'Cs': Cs,}
                
               
                lambda_t = self.lambda_model.intensity(sample_dict)

                for lam in lambda_t:
                    expection = lam
                    events_tmp.append(expection.detach().numpy())
                #print(events_tmp)
                times_tmp = [ti] 
    
                times_tmp = np.asarray(times_tmp)
                events_tmp = np.asarray(events_tmp)
                new_datas.append(events_tmp)
                y_true.append(ci)
        
        return new_datas, y_true

    
 