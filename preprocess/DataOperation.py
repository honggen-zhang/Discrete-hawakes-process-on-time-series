"""
This script contains a parent class of data preprocess functions and classes

The data in our toolbox is always formulated as follows:

database = {'event_features': None or (De, C) float array of event's static features,
                              C is the number of event types.
            'seq2idx': a Dict = {'seq_name': seq_index}
            'idx2seq': a Dict = {seq_index: 'seq_name'}
            'sequences': a List  = [seq_1, seq_2, ..., seq_N].
            }

For the i-th sequence:
seq_i = {'times': (N,) float array of timestamps, N is the number of events.
         'events': (N,) int array of event types.
         'seq_feature': None or (Ds,) float array of sequence's static feature.
         't_start': a float number indicating the start timestamp of the sequence.
         't_stop': a float number indicating the stop timestamp of the sequence.
         }
"""

import copy
from dev.util import logger
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict




class EventSampler(Dataset):
    """Load event sequences via minbatch"""
    def __init__(self, database, memorysize,start):

        self.event_cell = []
        #self.event_num = []
        self.time_cell = []
        self.database = database
        self.memory_size = memorysize
        for i in range(len(database['sequences'])):
            seq_i = database['sequences'][i]
            times = seq_i['times']
            events = seq_i['events']
            #print(events)
            t_start = seq_i['t_start']
            for j in range(memorysize,len(events)):
                target = events[j]
                target_t = times[j]
                former = []
                for k in range(memorysize):
                    #former.append([0.,0.,0.,0.])#multvariate
                    former.append([0.])
                former = np.array(former)
                        

                        #former = np.random.choice(len(self.database['type2idx']), memorysize)
                        
                former_t = t_start * np.ones((memorysize,))

                if 0 < j < memorysize:
                    former[-j:] = events[:j]
                    former_t[-j:] = times[:j]
                elif j >= memorysize:
                    former = events[j-memorysize:j]
                    former_t = times[j-memorysize:j]
                #for cur_c in target:
                    #if cur_c

                #self.event_cell.append((target, former, i))
                if int(target_t) >= start:
                    self.event_cell.append((target, former, i))
                    #if i==2:
                    #print((target, former, i))
                    #print('tt:',(target_t, former_t))
                    self.time_cell.append((target_t, former_t))
        #logger.info('In this dataset, the number of events = {}.'.format(len(self.event_cell)))
        logger.info('Each event is influenced by its last {} historical events.'.format(self.memory_size))

    def __len__(self):
        return len(self.event_cell)

    def __getitem__(self, idx):
        current_time = torch.Tensor([self.time_cell[idx][0]])  # torch.from_numpy()
        current_time = current_time.type(torch.FloatTensor)
        history_time = torch.from_numpy(self.time_cell[idx][1])
        history_time = history_time.type(torch.FloatTensor)

        current_event_numpy = self.event_cell[idx][0]
        n_current_event = []
        #print(current_event_numpy)
        for i in current_event_numpy:
            n_current_event.append(float(i))
        current_event = torch.Tensor(n_current_event)
        current_event = current_event.type(torch.LongTensor)
        history_event_numpy = self.event_cell[idx][1]
        n_history_event = []
        for i in history_event_numpy:
            tmp = []
            for j in i:
                tmp.append(float(j))
            #tmp = torch.from_numpy(np.array(tmp))
            #tmp.type(torch.LongTensor)
            n_history_event.append(tmp)
        #print(n_history_event)
        #history_event = torch.tensor(n_history_event)
        history_event = torch.from_numpy(np.array(n_history_event))
        history_event = history_event.type(torch.LongTensor)

        #current_seq_numpy = self.event_cell[idx][2]
        current_seq = torch.Tensor([self.event_cell[idx][2]])
        current_seq = current_seq.type(torch.LongTensor)

        return current_time, history_time, current_event, history_event, current_seq  # 5 outputs

     




def samples2dict(samples, device, Cs):

    ti = samples[0].to(device)
    tjs = samples[1].to(device)
    ci = samples[2].to(device)
    cjs = samples[3].to(device)
    sn = samples[4].to(device)

    batch_dict = {'ti': ti,
                  'tjs': tjs,
                  'ci': ci,
                  'cjs': cjs,
                  'sn': sn,
                  'Cs': Cs,}
    #print(batch_dict)
    return ci, batch_dict


def enumerate_all_events(database, seq_id, use_cuda):

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    Cs = torch.LongTensor(list(range(4)))
    Cs = Cs.view(-1, 1)
    Cs = Cs.to(device)

    FCs = None

    sn = torch.LongTensor([seq_id])
    sn = sn.view(1, 1).repeat(4, 1)
    sn = sn.to(device)

    fsn = None

    event_dict = {'ci': Cs,
                  'sn': sn,
                  'fsn': fsn,
                  'Cs': Cs,
                  'FCs': FCs}
    return event_dict


def data_info(database):

    logger.info('** Statistics of Target Database **')
    logger.info('- The number of event types = {}.'.format(len(database['type2idx'])))
    logger.info('- The number of sequences = {}.'.format(len(database['seq2idx'])))
    if database['event_features'] is not None:
        logger.info('- Each event has a feature vector with dimension {}.'.format(
            database['event_features'].shape[1]))
    else:
        logger.info('- Event feature is None.')

    if database['sequences'][0]['seq_feature'] is not None:
        logger.info('- Each sequence has a feature vector with dimension {}.'.format(
            database['sequences'][0]['seq_feature'].shape[0]))
    else:
        logger.info('- Sequence feature is None.')

    N_max = 0
    N_min = np.inf
    N_mean = 0
    for i in range(len(database['sequences'])):
        num_event = database['sequences'][i]['events'].shape[0]
        N_mean += num_event
        if num_event < N_min:
            N_min = num_event
        if num_event > N_max:
            N_max = num_event
    N_mean /= len(database['sequences'])
    logger.info('- The longest sequence is with {} events.'.format(N_max))
    logger.info('- The shortest sequence is with {} events.'.format(N_min))
    logger.info('- The average number of events per sequence is {:.2f}.'.format(N_mean))
