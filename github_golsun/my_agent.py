# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:11:15 2019

@author: rochej
"""

from lib import *

class Agent:
    def __init__(self, model, batch_size=32, discount_factor=0.95):
        self.model = model
        self.batch_size = batch_size
        self.discount_factor = discount_factor #gamma
        self.memory = []
    
    #experience replay: stockage dans la mémoire    
    def remember(self, state, action, reward, next_state, done, next_valid_actions):
        self.memory.append((state, action, reward, next_state, done, next_valid_actions))
        
    
    def replay(self):
        #choix random de n= batch_size tuples elementaires
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done, next_valid_actions in batch:
            q = reward
            if not done:
                q += self.discount_factor * np.nanmax(self.get_q_valid(next_state, next_valid_actions))
            self.model.fit(state, action, q)
    
    #acces à l'output du MLP: action prise en full exploitation    
    def get_q_valid(self, state, valid_actions):
        #output du MLP
        q = self.model.predict(state)
        q_valid = [np.nan] * len(q) #3 action donc vaut 3
        for action in valid_actions:
            #si je ne fais pas parti des valid action=> la sortie reste ds l'etat Nan
            q_valid[action] = q[action] 
        return q_valid
    
    #epsilon greedy
    def act(self, state, exploration, valid_actions):
        if np.random.random() > exploration:
            #exploitation
            q_valid = self.get_q_valid(state, valid_actions)
            #min et max en ignorant le Nan
            if np.nanmin(q_valid) != np.nanmax(q_valid):
                #on prend l'action qui maximize la Q_value
                return np.nanargmax(q_valid)
        #sinon, on prend une action random mais qui est valid
        return random.sample(valid_actions, 1)[0]
    

        
def add_dim(x, shape):
    return np.reshape(x, (1,) + shape)


class QModelKeras:
    # ref: https://keon.io/deep-q-learning/
    def init(self):
        pass
    
    def build_model(self):
        pass
    
    def __init__(self, state_shape, n_action):
        self.state_shape = state_shape
        self.n_action = n_action
        self.init()
        
                    
    def predict(self, state):
        #reshape l'input: state (1,40)
        q = self.model.predict(add_dim(state, self.state_shape))[0]
        if np.isnan(max(q)):
            print('state'+str(state))
            print('q'+str(q))
            raise ValueError
        return q
    
    def fit(self, state, action, q_action):
        #q: output du MLP
        q = self.predict(state)
        q[action] = q_action
        #fait backpropaf pour un seul update
        self.model.fit(add_dim(state, self.state_shape), add_dim(q, (self.n_action,)), 
                       epochs=1, verbose=0)
        
        
class QModelMLP(QModelKeras):
    # multi-layer perception (MLP), i.e., dense only
    def init(self):
        self.qmodel = 'MLP'	
    
    def build_model(self, n_hidden, learning_rate, activation='relu'):
        model = keras.models.Sequential()
        model.add(keras.layers.Reshape(
                (self.state_shape[0]*self.state_shape[1],), 
                input_shape=self.state_shape))
        
        for i in range(len(n_hidden)):
            model.add(keras.layers.Dense(n_hidden[i], activation=activation))
            #model.add(keras.layers.Dropout(drop_rate))
            
        model.add(keras.layers.Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
        self.model = model
        self.model_name = self.qmodel + str(n_hidden)
        
        
        
        