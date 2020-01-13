# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:44:51 2019

@author: rochej
"""

class DQN():

    def __init__(self, scope):
      self.scope = scope
      with tf.variable_scope(self.scope):
        self._build_model()

    def _build_model(self):
        # 4 Last frames of the game
        self.X_pl = tf.placeholder(shape=[None, 1,window_state], dtype=tf.float32, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # Get the batch size
        batch_size = tf.shape(self.X_pl)[0]
        
        d=window_state
        dp=50
        df=len(VALID_ACTIONS)
        
        w1=tf.Variable(tf.random.normal([d,dp ]))
        b1=tf.Variable(tf.zeros(dp))
        w2=tf.Variable(tf.random.normal([dp,df ]))
        b2=tf.Variable(tf.zeros(df))
        
        z1=tf.matmul(self.X_pl,w1)+b1
        z1a=tf.nn.leaky_relu(z1,alpha=0.5,name=None)
        z2=tf.matmul(z1a,w2)+b2
        
        self.predictions=z2
        tf.identity(self.predictions, name="predictions")

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.01, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, s):
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        ops = [self.train_op, self.loss]
        _, loss = sess.run(ops, feed_dict)
        return loss
    
def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)
    

tf.reset_default_graph()
# DQN
dqn = DQN(scope="dqn")
# DQN target
target_dqn = DQN(scope="target_dqn")

num_episodes = 10

replay_memory_size = 1000
replay_memory_init_size = 200

update_target_estimator_every = 5

epsilon_start = 1.0
epsilon_end = 0.1


epsilon_decay_steps = 5000
discount_factor = 0.9
batch_size = 32

    
    
def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



#saver = tf.train.Saver()
start_i_episode = 0
opti_step = -1

# The replay memory
replay_memory = []


with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    
    
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    #  Epsilon decay
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    # Policy
    policy = make_epsilon_greedy_policy(dqn, len(VALID_ACTIONS))
    
    epi_reward = []
    best_epi_reward = 0
    
    for i_episode in range(start_i_episode, num_episodes):
        print(i_episode)
        # Reset the environment
        state = env.reset()
        loss = None
        done = env.done
        r_sum = 0
        mean_epi_reward = np.mean(epi_reward)
        if best_epi_reward < mean_epi_reward:
            best_epi_reward = mean_epi_reward
    #            saver.save(tf.get_default_session(), checkpoint_path)
    
        len_replay_memory = len(replay_memory)
    
        while not done:
            
            # Get the epsilon for this step
            epsilon = epsilons[min(opti_step+1, epsilon_decay_steps-1)]
    
    
            # Update the target network
            if opti_step % update_target_estimator_every == 0:
                copy_model_parameters(sess, dqn, target_dqn)
    
#            print("\r Epsilon ({}) ReplayMemorySize : ({}) rSum: ({}) best_epi_reward: ({}) OptiStep ({}) @ Episode {}/{}, loss: {}".format(epsilon, len_replay_memory, mean_epi_reward, best_epi_reward, opti_step, i_episode + 1, num_episodes, loss), end="")
            print("\r Epsilon ({}) ReplayMemorySize : ({}) rSum: ({}) best_epi_reward: ({}) OptiStep ({}) @ Episode {}/{}, loss: {}"
                  .format(epsilon, len_replay_memory, mean_epi_reward, 
                          best_epi_reward, opti_step, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()
    
    
            #  Select an action with eps-greedy
            state=state.T
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            print('action',action)
    
            # Step in the env with this action
            next_state, reward, done = env.step(VALID_ACTIONS[action])
            r_sum += reward
    
    
            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
    
    
            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))
    
    
            if len_replay_memory > replay_memory_init_size:
                # Sample a minibatch from the replay memory
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
                
                next_states_batch=next_states_batch.reshape(32,1,window_state)
                # We compute the next q value with
                q_values_next_target = target_dqn.predict(sess, next_states_batch)
                q_values_next_target=q_values_next_target.reshape(32,3)
                
                t_best_actions = np.argmax(q_values_next_target, axis=1)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * q_values_next_target[np.arange(batch_size), t_best_actions]
    
                # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = dqn.update(sess, states_batch, action_batch, targets_batch)
    
                opti_step += 1
    
            state = next_state
            if done:
              break
    
        epi_reward.append(r_sum)
        if len(epi_reward) > 100:
            epi_reward = epi_reward[1:]
            
    obs=env.reset(test=True)
    obs=obs.T
    epsilon=0
    for k in range(1000):
        if env.done:
            break
#        action=take_random_act()
        action_probs = policy(sess, obs, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        obs, reward, done=env.step(VALID_ACTIONS[action])
        obs=obs.T
        step=env.current_step
        print("action #: %s reward: %f step: %f done: %d " % (VALID_ACTIONS[action], reward, step,done))

        
        




l_act=np.array(env.list_action)
ret=data[window_state+1:,].reshape(-1,)
ret_strat=l_act*ret
cum_ret=cumreturn(pd.DataFrame(ret_strat))
cum_ret_real=cumreturn(pd.DataFrame(ret))

plt.figure()
plt.plot(cum_ret_real)
plt.plot(cum_ret)


plt.figure()
plt.plot(l_act)   
