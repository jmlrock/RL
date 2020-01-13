# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:44:40 2019

@author: rochej
"""

from lib import *
from my_agent import *
import matplotlib.pyplot as plt
from visualizer import *





df=pd.read_excel('VIX.xlsx',index_col='Date')
df_vix=df['VIX']
df_sp=df['SPX 500']
n_var=1




def get_random_block(df,window_data):
    t_max=len(df)
    t_rand=np.random.randint(t_max-window_data-1)
    return(df[t_rand:t_rand+window_data])
    
df_rand=get_random_block(np.array(df_sp),180)

#plt.plot(df_rand)




def find_ideal(p, just_once):
    if not just_once:
        diff = np.array(p[1:]) - np.array(p[:-1])
        return sum(np.maximum(np.zeros(diff.shape), diff))
    else:
        best = 0.
        i0_best = None
        for i in range(len(p)-1):
            best = max(best, max(p[i+1:]) - p[i])
            
    return(best)
    
def load_data_sin():
    price = 90*np.sin(np.arange(180)/20)+100 #sine prices
    return pd.Series(price)

df_sin= load_data_sin()
#df_sin.plot()

#Trading environement 
class Market_env:
    """
    state
    MA of prices, normalized using values at t
    ndarray of shape (window_state, n_instruments * n_MA), i.e., 2D
    which is self.state_shape
    
    action:three action
    0:empty, don't open/close. 
    1:open a position
    2:keep a position
    """
    def __init__(self, 
                 prices_a, window_state, open_cost,window_data=180,
                 direction=1., risk_averse=0.,):
        self.prices_a = prices_a
        self.window_state = window_state
        self.open_cost = open_cost
        self.direction = direction
        self.risk_averse = risk_averse
        self.window_data=window_data
        
        self.n_action = 3
        self.state_shape = (window_state, n_var)
        self.action_labels = ['empty','open','keep']
        self.t0 = window_state - 1
        self.title='sin'
    
    def reset(self,rand_price=True):
        #empty vrai car intitialiament, pas de postions ouveerte ou fermée
        #PF vide
        self.empty = True
        if rand_price:
            #genere une nouvelle serie
#            prices=get_random_block(self.prices_a,self.window_data)
            prices=self.prices_a
            price = np.reshape(prices[:,0], prices.shape[0])
            self.prices = prices.copy()
            #rescaling 
            self.price = price/price[0]*100
            #t_max=window_data
            self.t_max = len(self.price) - 1
            
        self.max_profit = find_ideal(self.price[self.t0:], False)
        #reinitialisation à l'instant initial to=39 
        self.t = self.t0
        return self.get_state(), self.get_valid_actions()
    
    def get_state(self, t=None):
        if t is None:
            t = self.t
        #fenetre de 40 past days
        state = self.prices[t - self.window_state + 1: t + 1, :].copy()
        #nbre d'asset
        for i in range(n_var):
            #renormalisation de l'input = fenetre des past days
            norm = np.mean(state[:,i])
            state[:,i] = (state[:,i]/norm - 1.)*100
        return state
    
    def get_valid_actions(self):
        #empt =True; je viens juste de prendre l'action 0
        if self.empty:
            #si je ne possède pas de stocks: 
            #2 possibilté: attendre (0) ou prendre une positions acheteuse (1)
            return [0, 1]	# wait, open
        else:
            #si j'ai pris une position dans le passé:
            #2 possibilité: la fermer (cad la vendre donc retour à l'etat 0) ou conserver la position (2)
            return [0, 2]	# close, keep
        
    def get_noncash_reward(self, t=None, empty=None):
        if t is None:
            t = self.t
        if empty is None:
            empty = self.empty
        reward = self.direction * (self.price[t+1] - self.price[t])
        if empty:
            reward -= self.open_cost
        if reward < 0:
            reward *= (1. + self.risk_averse)
        return reward
    
    def step(self, action):
        done = False
        if action == 0:		# wait/close
            reward = 0.
            self.empty = True
        elif action == 1:	# open
            reward = self.get_noncash_reward()
            self.empty = False
        elif action == 2:	# keep
            reward = self.get_noncash_reward()
        else:
            raise ValueError('no such action: '+str(action))
            
        self.t += 1
        return self.get_state(), reward, self.t == self.t_max, self.get_valid_actions()

'test environement'

prices_a=np.array(df_sin)
prices_a=prices_a.reshape(-1,1)
window_state = 40
open_cost = 3.3
env=Market_env(prices_a,window_state, open_cost)
state,valid_action=env.reset()


class Simulator:
    def __init__(self, agent, env, visualizer, fld_save):
        self.agent = agent
        self.env = env
        self.visualizer = visualizer
        self.fld_save = fld_save
    
    def play_one_episode(self, exploration, training=True, rand_price=True, print_t=False):
        #si rand_pruce=True=> generation d'une nouvelle serie
        #on rest: temps remis a zeros,...
        state, valid_actions = self.env.reset(rand_price=rand_price)
        done = False
        env_t = 0
        try:
            env_t = self.env.t
            #normalement, env.t est initailiser à t=40
        except AttributeError:
            pass
        
        cum_rewards = [np.nan] * env_t
        actions = [np.nan] * env_t
        states = [None] * env_t
        prev_cum_rewards = 0.
        
        #done=False: tant que la session de trading n'est pas terminé
        #done=True: fin de la session de trading (if t=t_max)
        while not done:
            if print_t:
                print(self.env.t)
            #prs d'action selon eps-greedy
            #exploration = epsilon
            action = self.agent.act(state, exploration, valid_actions)
            next_state, reward, done, valid_actions = self.env.step(action)
            
            cum_rewards.append(prev_cum_rewards+reward)
            prev_cum_rewards = cum_rewards[-1]
            actions.append(action)
            states.append(next_state)
            
            if training:
                #stockage memoire des transitions
                self.agent.remember(state, action, reward, next_state, done, valid_actions)
                #replay,  train le MLP => optimization pr 1 step
                self.agent.replay()
                
            state = next_state
            
        return cum_rewards, actions, states
    
    def train(self, n_episode, save_per_episode=10, exploration_decay=0.995, exploration_min=0.01, print_t=False, exploration_init=1.):
        exploration = exploration_init
        fld_save = os.path.join(self.fld_save,'training')
        
        makedirs(fld_save)
        MA_window = 100		# MA of performance
        safe_total_rewards = [] #safe reward => pas d'exploration => que de l'exploitation
        explored_total_rewards = []
        explorations = [] #stockage des epsilon
        path_record = os.path.join(fld_save,'record.csv')
        
        with open(path_record,'w') as f:
            f.write('episode,exploration,explored,safe,MA_explored,MA_safe\n')
            
        for n in range(n_episode):
            
            print('\ntraining...')
            print('episode',n)
            #update de l'exploaration=epsilon
            exploration = max(exploration_min, exploration * exploration_decay)
            explorations.append(exploration)
            explored_cum_rewards, explored_actions, _ = self.play_one_episode(exploration, print_t=print_t)
            explored_total_rewards.append(100.*explored_cum_rewards[-1]/self.env.max_profit)
            
            #plus d'exploration (epsilon=0) => que de l'exploitation // pas d'update = pas de training
            safe_cum_rewards, safe_actions, _ = self.play_one_episode(0, training=False, rand_price=False, print_t=False)
            safe_total_rewards.append(100.*safe_cum_rewards[-1]/self.env.max_profit)
            
            MA_total_rewards = np.median(explored_total_rewards[-MA_window:])
            MA_safe_total_rewards = np.median(safe_total_rewards[-MA_window:])
            
            ss =[
                    str(n), '%.1f'%(exploration*100.),
                    '%.1f'%(explored_total_rewards[-1]), '%.1f'%(safe_total_rewards[-1]),
                    '%.1f'%MA_total_rewards, '%.1f'%MA_safe_total_rewards,
                    
                ]
            
            with open(path_record,'a') as f:
                f.write(','.join(ss)+'\n')
                print('\t'.join(ss))
                
                
#            if n%save_per_episode == 0:
#                print('saving results...')
#                self.agent.save(fld_model)
                
                """
                self.visualizer.plot_a_episode(
                        self.env, self.agent.model,
                        explored_cum_rewards, explored_actions,
                        safe_cum_rewards, safe_actions,
                        os.path.join(fld_save, 'episode_%i.png'%(n)))
                
                self.visualizer.plot_episodes(
                        explored_total_rewards, safe_total_rewards, explorations,
                        os.path.join(fld_save, 'total_rewards.png'),
                        MA_window)
                """



    def test(self, n_episode, save_per_episode=10, subfld='testing'):
        
        fld_save = os.path.join(self.fld_save, subfld)
        makedirs(fld_save)
        MA_window = 100		# MA of performance
        #ds la session de test, on ne fait que de l'exploitation
        safe_total_rewards = []
        path_record = os.path.join(fld_save,'record.csv')
        
        with open(path_record,'w') as f:
            f.write('episode,game,pnl,rel,MA\n')
        
        
        for n in range(n_episode):
            print('\ntesting...okok')
            
            #cum_reward, acion, state
            safe_cum_rewards, safe_actions, _ = self.play_one_episode(0, training=False, rand_price=True)
            safe_total_rewards.append(100.*safe_cum_rewards[-1]/self.env.max_profit)
            MA_safe_total_rewards = np.median(safe_total_rewards[-MA_window:])
            ss = [str(n), 
                  '%.1f'%(safe_cum_rewards[-1]),
                  '%.1f'%(safe_total_rewards[-1]), 
                  '%.1f'%MA_safe_total_rewards]
            
            with open(path_record,'a') as f:
                f.write(','.join(ss)+'\n')
                print('\t'.join(ss))
                
#            if n%save_per_episode == 0:
            if n ==n_episode-1 :
                print('saving results...')
                self.visualizer.plot_a_episode(
                    self.env, self.agent.model, 
                    [np.nan]*len(safe_cum_rewards), [np.nan]*len(safe_actions),
                    safe_cum_rewards, safe_actions,
                    os.path.join(fld_save, 'episode_%i.png'%(n)))

                self.visualizer.plot_episodes(
                    None, safe_total_rewards, None, 
                    os.path.join(fld_save, 'total_rewards.png'),
                    MA_window)


#prices_a=np.array(df_sin)
prices_a=np.array(df_rand)                
prices_a=prices_a.reshape(-1,1)
#
n_episode_training = 100
n_episode_testing = 1
open_cost = 3.3
#
batch_size = 8
learning_rate = 0.001 #learning rate de l'update de Q => ds la descente/monte de gradient
discount_factor = 0.8 #gamma
exploration_init = 1.
exploration_decay = 0.99 #decroissance de l'epsilon
exploration_min = 0.01
window_state = 5
#
env=Market_env(prices_a,window_state, open_cost)
visualizer = Visualizer(env.action_labels)
#
m = 16
layers = 5
hidden_size = [m]*layers
model = QModelMLP(env.state_shape, env.n_action)
model.build_model(hidden_size, learning_rate=learning_rate, activation='tanh')
agent = Agent(model, discount_factor=discount_factor, batch_size=batch_size)
#
fld_save = os.path.join(OUTPUT_FLD, 'spx_fix', model.model_name, 
                        str((env.window_state, 180, agent.batch_size, learning_rate,
                             agent.discount_factor, exploration_decay, env.open_cost)))
#
#
simulator = Simulator(agent, env, visualizer=visualizer, fld_save=fld_save)
simulator.train(n_episode_training, save_per_episode=1, exploration_decay=exploration_decay, 
                exploration_min=exploration_min, print_t=False, exploration_init=exploration_init)

simulator.test(n_episode_testing, save_per_episode=10, subfld='in-sample testing')
















