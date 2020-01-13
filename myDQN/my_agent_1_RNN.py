# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:27:29 2019

@author: rochej
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:17:59 2019

@author: rochej
"""

from my_env_1 import *

def normalize(data):
    mu=np.mean(data)
    std=np.std(data)
    return((data-mu)/std)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
#    def _build_model(self):
#        # Neural Net for Deep-Q learning Model
#        model = Sequential()
#        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#        model.add(Dense(4, activation='relu'))
#        model.add(Dense(self.action_size, activation='linear'))
#        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
#        return(model)
        
    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64,input_shape=(None,self.state_size),  return_sequences=True,stateful=False,kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return(model)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, epsilon):            
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print(act_values)

        return np.argmax(act_values[0][0])  # returns action
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
    
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
#            target_f_modif=target_f.reshape(1,3)
            target_f[0][0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay
            


data=np.random.normal(0,0.1,100)

line=-np.arange(200)+300
line=np.arange(200)+300
line = 50*np.sin(np.arange(500)/20)+100
#plt.plot(line)


prices=pd.read_csv('CAC.csv',index_col='Date',parse_dates=True)
prices_a=prices['Adj Close'].iloc[4800:].values
#plt.plot(prices_a)

data = get_return(prices_a)

episodes=10
window_state=5

VALID_ACTIONS=[-1,0,1]
action_size=len(VALID_ACTIONS)

env = Trading_env(data,window_state)
agent = DQNAgent(window_state,action_size)


epsilon=1
for e in range(episodes):

    print(e,epsilon)
##    # reset state in the beginning of each game
    state = env.reset()
    state = np.reshape(state, [1, 1,window_state])
    
#    # time_t represents each frame of the game
#    # Our goal is to keep the pole upright as long as possible until score of 500
#    # the more time_t the more score
    for time_t in range(500):
#        # turn this on if you want to render
#        # env.render()
#        # Decide action
        action_idx = agent.act(state,epsilon)
        action=VALID_ACTIONS[action_idx]

#        print(action)
        next_state, reward, done= env.step(action)
        next_state = np.reshape(next_state, [1,1, window_state])
#        # Remember the previous state, action, reward, and done
        agent.remember(state, action_idx, reward, next_state, done)
#        # make next_state the new current state for the next frame.
        state = next_state
        # done becomes True when the game ends
        # ex) The agent drops the pole
        agent.replay(32)
        
        if done:
            break
        
    if epsilon > 0.1:
        epsilon -= (1.0/episodes)

        

    # train the agent with the experience of the episode
#    agent.replay(32)
#    print(agent.epsilon)
##    
##
##
state = env.reset(test=True)
state = np.reshape(state, [1,1,window_state])
##
for k in range(1000):
    if env.done:
        break
    action = VALID_ACTIONS[agent.act(state,epsilon=0)]
#    action=VALID_ACTIONS[1]
    next_state, reward, done= env.step(action)
    next_state = np.reshape(next_state, [1, 1,window_state])
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
plt.plot(normalize(prices_a[window_state+1:,]),label='prix')
plt.legend()

a=np.ones((1,1,3))
a_p=a[0][0]
