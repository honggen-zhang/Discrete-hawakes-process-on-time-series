#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:40:00 2021

@author: user1
"""

"""
This script contains the function loading data from csv file
"""

from dev.util import logger
import numpy as np
import pandas as pd
import time
from typing import Dict
from sklearn import preprocessing


def load_sequences_csv(file_name: str,all_files,window_s,window_e,cluster_set, domain_names: Dict):

    print('Data loading --------------------------')
    database = {'event_features': None,
                'seq2idx': None,
                'idx2seq': None,
                'sequences': []}
    if len(all_files) !=1:
        
        #df = pd.read_csv(file_name+str(all_files[0]))
        add_n = 0
        bigdata = []
        
        for file in all_files:
            #content_test = open(file_name+str(file), "r").readlines()
            #print(str(file))
            all_id = []
            nn = 0
            df2 = pd.read_csv(file_name+str(file), usecols=['name','id','time','event'])    
            content_test = df2.values.tolist()
            for row in content_test:
               
                data ={}
                name_triple = row[0]
                idx = row[1]
                timex = row[2]
                eventx = row[3]
                data['name'] = name_triple
                data['id'] = int(idx) + add_n
                
                data['time'] = timex
                data['event'] = eventx
                bigdata.append(data)
                if int(idx) not in all_id:
                    nn = nn+1
                    all_id.append(int(idx))
                    
            add_n = nn + add_n
            #print(int(idx))
            #print(add_n)
        df=pd.DataFrame.from_dict(bigdata)
        df.to_csv('/home/user1/Desktop/ddhp/data_temp.csv',index=False)   
        df = pd.read_csv('/home/user1/Desktop/ddhp/data_temp.csv')

    else:
        df = pd.read_csv(file_name+str(all_files[0]))
        #df = pd.read_csv(file_name)
#-------------------------------------------------
        


#=================================================================================    
    seq2idx = {}
    idx2seq = {}

    #logger.info('Count the number of sequences...')
    start = time.time()
    seq_idx = 0
    for i, row in df.iterrows():
        triple_name = row['name']
        seq_name = str(row['id'])
        timestamp = float(row['time'])
        if seq_name not in seq2idx.keys() and triple_name in cluster_set:
            seq2idx[seq_name] = seq_idx
            seq = {'times': [],
                   'events': [],
                   'seq_feature': None,
                   't_start': 0.0,
                   't_stop': 0.0,
                   'label': None}
            database['sequences'].append(seq)
            seq_idx += 1

    #logger.info('Build proposed database for the sequences...')
    #start2 = time.time()
    for seq_name in seq2idx.keys():
        seq_idx = seq2idx[seq_name]
        idx2seq[seq_idx] = seq_name

    database['seq2idx'] = seq2idx
    database['idx2seq'] = idx2seq
    old_name = str(-1)
    #print(database['seq2idx'])

    for i, row in df.iterrows():
        seq_name = str(row['id'])#0
        tri_name = row['name']
        if tri_name in cluster_set:
            seq_idx = database['seq2idx'][seq_name]        
            timestamp = float(row['time'])
            if int(timestamp)<= window_e:
                event_type = str(row['event'])[1:-1].split(',')
                #print('event_type:', event_type)
                append_event = event_type[0:1]
                copy_event =  event_type[1:2]
                extend_event =  event_type[2:3]
                mutate_event =  event_type[3:4]
                #print(mutate_event[0])
                target_event = [append_event[0],extend_event[0],mutate_event[0]]

                #print('target_event:', target_event)
                #if float(target_event[0]) != 0: 
                seq_idx = database['seq2idx'][seq_name]
                database['sequences'][seq_idx]['times'].append(timestamp)
                database['sequences'][seq_idx]['events'].append(target_event)
                
            
            if seq_name!=old_name and database['sequences'][seq_idx]['times']==[]:
                database['sequences'][seq_idx]['times'].append(float(window_s))
                database['sequences'][seq_idx]['events'].append([0])   
            old_name = seq_name
        
    

    for n in range(len(database['sequences'])):
        #print(database['sequences'][2])
        database['sequences'][n]['t_start'] = database['sequences'][n]['times'][0]
        #database['sequences'][n]['t_stop'] = database['sequences'][n]['times'][-1]+1e-2
        database['sequences'][n]['t_stop'] = database['sequences'][n]['times'][-1]
        database['sequences'][n]['times'] = np.asarray(database['sequences'][n]['times'])
        matrix = (np.asarray(database['sequences'][n]['events']))

        database['sequences'][n]['events'] = np.asarray(matrix)
        if n % 1000 == 0:
            logger.info('{} sequences have been processed... Time={}ms.'.format(n, round(1000*(time.time() - start))))
    logger.info('Done! The database has been built in {}ms'.format(round(1000*(time.time() - start))))
    

    return database