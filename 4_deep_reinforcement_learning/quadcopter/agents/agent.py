
import random
from collections import namedtuple, deque
from keras import layers, models, optimizers
from keras import backend as K
import numpy as np
import copy
from agents.critic import Critic
from agents.actor import Actor

class ReplayBuffer:
  #fixed-size byffer to store experience tuples.

  def __init__(self, buffer_size, batch_size):
    #initialise a ReplayBuffer object
    self.memory = deque(maxlen=buffer_size) #internal memory (deque)
    self.batch_size = batch_size
    self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

  def add(self, state, action, reward, next_state, done):
    #add a new experience to memory
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)

  def sample(self, batch_size=64):
    #randomly sample a batch of experiences from memory
    return random.sample(self.memory, k=self.batch_size)

  def __len__(self):
    #return current size of internal memory
    return len(self.memory)

class OUNoise:
  #Ornstein-Uhlenbeck process

  def __init__(self, size, mu, theta, sigma):
    #initialise parameters and noise process
    self.mu = mu * np.ones(size)
    self.theta = theta
    self.sigma = sigma
    self.reset()

  def sample(self):
    #reset internal state (= noise) to mean (mu)
    self.state = copy.copy(self.mu)

  def sample(self):
    #update internal state and return it as a noise sample
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
    self.state = x + dx
    return self.state

class DDPG_Agent():
  #reinforcement agent by DDPG

  def __init__(self, task):
    self.task = task
    self.state_size = task.state_size
    self.action_size = task.action_size
    self.action_low = task.action_low
    self.action_high = task.action_high

    #actor (policy) model
    self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
    self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

    #critic (value) model
    self.critic_local = Critic(self.state_size, self.action_size)

    #initialise target model parameters with local model parameters
    self.critic_target.model.set_weights(self.critic_local.model.get_weights())
    self.actor_target.model.set_weights(self.actor_local.model.get_weights())

    #noise process long-running mean / the speed of mean reversion / the volatility parameter
    self.exploration_mu = 0
    self.exploration_theta = 0.15
    self.exploration_sigma = 0.2
    self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

    #replay memory
    self.buffer_size = 1000000
    self.batch_size = 64
    self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    #algorithm parameters
    self.gamma = 0.99 #discount factor
    self.tau = 0.001 #soft update of target parameters

  def reset_episode(self):
    self.noise.reset()
    state = self.task.reset()
    self.last_state = state
    return state
  
  def step(self, action, reward, next_state, done):
    #save experience / reward
    self.memory.add(self.last_state, action, reward, next_state, done)

    #learn, if enough samples are available in memory
    if len(self.memory) > self.batch_size:
      experiences = self.memory.sample()
      self.learn(experiences)

    #roll over last state and action
    self.last_state = next_state

  def act(self, state):
    #returns actions for given state(s) as per current policy
    state = np.reshape(state, [-1, self.state_size])
    action = self.actor_local.model.predict(state)[0]
    return list(action + self.noise.sample()) #add noise for exploration

  def learn(self, experiences):
    #update polocy and value parameters using given batch of experience tuples

    #convert experience tuples to separate arrays for each element(states, actions, rewards, etc.)
    states = np.vstack([e.state for e in experiences if e is not None])
    actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
    rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
    dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
    next_states = np.vstack([e.next_state for e in experiences if e is not None])

    #get predicted next-state actions and Q values from target models
    actions_next = self.actor_target.model.predict_on_batch(next_states)
    Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

    #compute Q targets for current states and train critic model (local)
    Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
    self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

    #train actor model (local)
    action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
    self.actor_local.train_fn([states, action_gradients, 1]) #custom training function

    #soft-update target models
    self.soft_update(self.critic_local.model, self.critic_target.model)
    self.soft_update(self.actor_local.model, self.actor_target.model)

  def soft_update(self, local_model, target_model):
    #soft update model parameters
    local_weights = np.array(local_model.get_weights())
    target_weights = np.array(target_model.get_weights())

    assert len(local_weights) == len(target_weights), 'local and target model parameters must have same size'

    new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
    target_model.set_weights(new_weights)





