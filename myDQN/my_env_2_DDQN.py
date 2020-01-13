# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:50:20 2019

@author: rochej
"""

from my_env_1 import *

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory  = deque(maxlen=2000) #peut stocker au max 2000 transitions: off-policy model
        self.gamma = 0.85 #futur reward depreciation
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.model= self.create_model()
        self.target_model = self.create_model()
        
    def create_model(self):
        model   = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
#        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model
    
    #epsilon-greedy function
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])
    
    #ajout des transitions au buffer de l'experience replay
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
        
    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0) #1 seul update 
    
    #update du target_model plus lent que l'update du model
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
        
    def save_model(self, fn):
        self.model.save(fn)
        

data=np.random.normal(0,0.1,100)

line=-np.arange(200)+300
line=np.arange(200)+300
line = 50*np.sin(np.arange(500)/20)+100
#plt.plot(line)


prices=pd.read_csv('CAC.csv',index_col='Date',parse_dates=True)
prices_a=prices['Adj Close'].iloc[4500:].values


data = get_return(line)

episodes=150
window_state=20
VALID_ACTIONS=[-1,0,1]
action_size=len(VALID_ACTIONS)


env = Trading_env(data,window_state)
agent = DQN(window_state,action_size)


episodes  = 100
trial_len = 500


steps = []
for e in range(episodes):

    cur_state = env.reset().reshape(1,window_state)
    for step in range(trial_len):
        print(step,agent.epsilon)
        action = agent.act(cur_state)
        new_state, reward, done= env.step(action)
        new_state = new_state.reshape(1,window_state)
        agent.remember(cur_state, action, reward, new_state, done)

        agent.replay()       # internally iterates default (prediction) model
        agent.target_train() # iterates target model

        cur_state = new_state
        if done:
            break