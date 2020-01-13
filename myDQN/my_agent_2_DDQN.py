# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:50:20 2019

@author: rochej
"""

from my_env_1 import *

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory  = deque(maxlen=2000) #peut stocker au max 2000 transitions: off-policy model
        self.gamma = 0.9 #futur reward depreciation

        self.learning_rate = 0.005
        self.tau = .125
        self.model= self.create_model()
        self.target_model = self.create_model()
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(28, input_dim=self.state_size, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return(model)
    
    #epsilon-greedy function
    def act(self, state, epsilon):            
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
#        print(act_values)
        return np.argmax(act_values[0])  # returns action
    
    #ajout des transitions au buffer de l'experience replay
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
        
    def replay(self,batch_size):
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
#line=line[:200]
#plt.plot(line)


prices=pd.read_csv('CAC.csv',index_col='Date',parse_dates=True)
prices_a=prices['Adj Close'].iloc[4800:].values
#plt.plot(prices_a)

data = get_return(line)

episodes=3
window_state=2

VALID_ACTIONS=[-1,0,1]
action_size=len(VALID_ACTIONS)

env = Trading_env(data,window_state)
agent = DQNAgent(window_state,action_size)


epsilon=1
epsilon_min=0.1
epi_reward = []
for e in range(episodes):
    r_sum = 0

    print(e,epsilon)
##    # reset state in the beginning of each game
    state = env.reset()
    state = np.reshape(state, [1, window_state])
    
    for time_t in range(500):
        action_idx = agent.act(state,epsilon)
        action=VALID_ACTIONS[action_idx]

        next_state, reward, done= env.step(action)
        next_state = np.reshape(next_state, [1, window_state])
#        # Remember the previous state, action, reward, and done
        agent.remember(state, action_idx, reward, next_state, done)
#        # make next_state the new current state for the next frame.
#        state = next_state
        # done becomes True when the game ends
        # ex) The agent drops the pole
        agent.replay(32)
        agent.target_train()
        
        state = next_state
        r_sum += reward
        
        if done:
            break
    print(r_sum)
    epi_reward.append(r_sum)
        
    if epsilon > epsilon_min:
        epsilon -= (1.0/episodes)

state = env.reset(test=True)
state = np.reshape(state, [1,window_state])
##
for k in range(1000):
    if env.done:
        break
    action = VALID_ACTIONS[agent.act(state,epsilon=0)]
#    action=VALID_ACTIONS[1]
    next_state, reward, done= env.step(action)
    next_state = np.reshape(next_state, [1, window_state])
    state = next_state
    step=env.current_step
    print("action #: %s reward: %f step: %f  " % (action, reward, step))


l_act=np.array(env.list_action)
ret=data[window_state+1:,].reshape(-1,)
ret_strat=l_act*ret
cum_ret=cumreturn(pd.DataFrame(ret_strat))
cum_ret_real=cumreturn(pd.DataFrame(ret))
#
plt.figure()
plt.plot(cum_ret_real,label='prices')
plt.plot(cum_ret,label='strat')
plt.legend()
#
plt.figure()
plt.plot(l_act,label='action')
#plt.plot(normalize(prices_a[window_state+1:,]),label='prix')
plt.plot(normalize(line),label='prix')
plt.legend()
