# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:15:39 2019

@author: rochej
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
import tensorflow as tf
from collections import deque, namedtuple
import itertools
import random
import os
import sys

from sklearn import metrics, preprocessing
from progress.bar import Bar
from tqdm import tqdm, tqdm_notebook

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, Add, Concatenate, LSTM 
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling1D, GlobalAveragePooling1D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop,Adam
from functools import partial
import keras.backend as K
from keras import regularizers
from stockstats import StockDataFrame as Sdf


from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

v=np.array([1,1])
w=2*v
z=np.vdot(v,w)
z=w*v

def get_return(Y):
    Y_df=pd.DataFrame(Y)
#    R=Y_df.pct_change()[1:]
    R=Y_df.pct_change()
    return(R)

class Trading_env:
    def __init__(self, data, window_state):
        self.window_state=window_state
        self.data=data
        self.n_action = 3
        self.state_shape = (window_state,)
        self.t_max=len(self.data) - 1
        self.data_indicator=self.get_indicator()
        
    def get_indicator(self,spread=.005):
        df = self.data
        if "Name" in df:
            df.drop('Name',axis=1,inplace=True)
        _stock = Sdf.retype(df.copy())
        _stock.get('cci_14')
        _stock.get('rsi_14')
        _stock.get('dx_14')
        
        _stock = _stock.dropna(how='any')

        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(_stock[['rsi_14', 'cci_14','dx_14','volume']])
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized.columns = ['rsi_14', 'cci_14','dx_14','volume']
        df_normalized['bid'] = _stock['close'].values
        df_normalized['ask'] = df_normalized['bid'] + spread
        df_normalized['mid'] = (df_normalized['bid'] + df_normalized['ask'])/2
        df_normalized['return']=get_return(df_normalized['mid'])
        
        df_normalized=df_normalized.dropna(axis=0)
        
        return(df_normalized)
        
    def reset(self,test=False):
        self.done=False
        self.current_step = np.random.randint(self.window_state, self.t_max)
        self.list_action=[]
        if test:
            self.current_step=self.window_state+1
        return(self.get_state())
    
    def get_state(self):
        return(np.array(self.data_indicator.iloc[self.current_step]))
        
    def get_past_state(self):
        past_values=self.data_indicator.iloc[self.current_step-self.window_state:self.current_step]
        return(np.array(past_values['return']))
        
    def step(self, action):
        
        if self.current_step > len(self.data)-2:
            self.done=True
        self.current_step += 1
        reward = self.get_reward(action)
        obs = self.get_state()
        return(obs, reward, self.done)
        
    def get_reward(self, action):
        self.list_action.append(action)
        n=len(self.list_action)
        if n<self.window_state+1:
            sharpe=0
        else:
            action_L=np.array(self.list_action[n-self.window_state-1:n-1])
            if check_zero(action_L):
                sharpe=0
            else:
                ret_L=self.get_past_state()
                m=np.mean(np.vdot(action_L,ret_L))
                std=np.std(ret_L*action_L)*np.sqrt(252)
#                print(m,std)
                sharpe=m/std
#                sharpe=m
        return(sharpe)
        
def take_random_act():
    return(np.random.randint(0,3)-1)
    

    
def cumreturn(ret):
    return((1+ret).cumprod())
    
    
def check_zero(L):
    return(all(v == 0 for v in L))
    
#prices=pd.read_csv('AAP_data.csv',index_col='date',parse_dates=True)
#window_state=5
#env = Trading_env(prices,window_state)
#data_n=env.get_indicator()
##v=data_n.iloc[2:10]['return']
#
#obs=env.reset(test=True)
#for k in range(1000):
#    if env.done:
#        break
#    action=take_random_act()
##    action=1
#    obs, reward, done=env.step(action)
#    step=env.current_step
#    print("action #: %s reward: %f step: %f " % (action, reward, step))
    
        



    


    

    
          

        
        
    