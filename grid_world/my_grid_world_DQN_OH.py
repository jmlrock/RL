# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:49:06 2019

@author: rochej
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:31:59 2019

@author: rochej
"""

import numpy as np
from random import randint
import random
from collections import deque


from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, Add, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling1D, GlobalAveragePooling1D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop,Adam
from functools import partial
import keras.backend as K
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))





class EnvGrid(object):
    """
        docstring forEnvGrid.
    """
    def __init__(self):
        super(EnvGrid, self).__init__()

        self.grid = [
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 0]
        ]
        # Starting position
        self.y = 2
        self.x = 0

        self.actions = [
            [-1, 0], # Up
            [1, 0], #Down
            [0, -1], # Left
            [0, 1] # Right
        ]

    def reset(self):
        """
            Reset world
        """
        self.y = 2
        self.x = 0
        return (self.y*3+self.x+1)

    def step(self, action):
        """
            Action: 0, 1, 2, 3
        """
        self.y = max(0, min(self.y + self.actions[action][0],2))
        self.x = max(0, min(self.x + self.actions[action][1],2))

        return (self.y*3+self.x+1) , self.grid[self.y][self.x]

    def show(self):
        """
            Show the grid
        """
        print("---------------------")
        y = 0
        for line in self.grid:
            x = 0
            for pt in line:
                print("%s\t" % (pt if y != self.y or x != self.x else "X"), end="")
                x += 1
            y += 1
            print("")

    def is_finished(self):
        return self.grid[self.y][self.x] == 1

def take_action(st, Q, eps):
    # Take an action
    if random.uniform(0, 1) < eps:
        action = randint(0, 3)
    else: # Or greedy action
        action = np.argmax(Q[st])
    return action

def onehot(state):
    v=np.zeros(9)
    v[state-1]=1
    v=v.reshape(1,-1)
    return(v)
    
def decode(v):
    return(np.argmax(v)+1)
    
s=np.array([i for i in range(1,10)]).reshape(-1,1)
L_s=scaler.fit_transform(s)

print(L_s[2])

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=50) #peut stocker au max 2000 transitions: off-policy model
        self.gamma = 0.9 #futur reward depreciation
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.model= self.create_model()
        self.target_model = self.create_model()
#        self.summary_t=self.target_model.summary()
#        self.summary=self.model.summary()
        
    def create_model(self):
        model   = Sequential()
        state_shape  = 1
        model.add(Dense(15, input_dim=state_shape, activation="relu"))
        model.add(Dense(4))  #4 action possible 
        
        a=Input(shape=(1,))
        b=model(a)
        my_model=Model(a,b)
        my_model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.1))
        return(my_model)
    
    #epsilon-greedy function
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return(np.random.randint(4))
        return np.argmax(self.model.predict(np.array(L_s[state-1]))[0]) #ok
    
    def take_action(self,state, eps=0.4):
        # Take an action
        if random.uniform(0, 1) < eps:
            action = randint(0, 3)
        else: # Or greedy action
            action = np.argmax(self.model.predict(np.array(L_s[state-1]))[0])
        return action
    
    #ajout des transitions au buffer de l'experience replay
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
        
    def replay(self):
#        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.1))
#        self.target_model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.1))
        batch_size = 10
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(np.array(L_s[state-1]))
            if done:
                target[0][action] = reward
            else:
                Q_future = np.max(self.target_model.predict(np.array([L_s[state-1]])))
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(np.array(L_s[state-1]), target, epochs=1, verbose=0) #1 seul update 
    
    #update du target_model plus lent que l'update du model
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
        
    def save_model(self, fn):
        self.model.save(fn)

#if __name__ == '__main__':
env = EnvGrid()
dqn_agent = DQN(env)
memo=dqn_agent.memory
#
epoch=2
for e in range(epoch):
    print(e)
#    print('memo',len(memo))
    st = env.reset()
    
    
    for s in range(1,10):
        Q=dqn_agent.model.predict(np.array(L_s[s-1]))
        print(s,Q)
        
    while not env.is_finished():
#        env.show()
        at = dqn_agent.take_action(st)
        stp1, r = env.step(at)
        
        dqn_agent.remember(st, at, r, stp1, env.is_finished())
        dqn_agent.replay()       # internally iterates default (prediction) model
        dqn_agent.target_train()
        
        st = stp1

            
            

#model=Sequential()
#state_shape  = 9
#model.add(Dense(15, input_dim=state_shape, activation="relu"))
#model.add(Dense(4))  #4 action possible 
#model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.1))
#
#a=Input(shape=(9,))
#b=model(a)
#my_model=Model(a,b)
##
#for s in range(1,10):
#    s_oh=onehot(s).reshape(1,-1)
#    q=my_model.predict(s_oh)
#    print(s,q)
