import gym
import gym_futbol
import numpy as np
from IPython import display
import matplotlib.pyplot as plt

from collections import deque

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda
from keras.optimizers import Adam

import random

def train_agent(EPISODES=100):
    
      env = gym.make('Futbol-v0')
      state_size_a, state_size_b = env.observation_space.shape
      state_size = state_size_a * state_size_b
      action_size = env.action_space.n

      agent = DQNAgent(state_size, action_size)

      state = env.reset()
      state = np.reshape(state, [1, state_size])

      batch_size = 32

      for _ in range(EPISODES):

          done = False

          while not done: 
            
              action = agent.act(state)
              next_state, reward, done, _ = env.step(action)
              env.render()
              next_state = np.reshape(next_state, [1, state_size])
              agent.memorize(state, action, reward, next_state, done)
              state = next_state

              if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.name = "DQN Agent"

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

train_agent(EPISODES=1)

def eval_agent(agent, episodes = 10):
    
    total_score = 0
    env = gym.make('Futbol-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    for _ in range(episodes):

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        time = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            time += 1

        total_score += time

    print(f"Results after {episodes} episodes:")
    print("average score: {}, epsilon: {:.5}".format(total_score / episodes, agent.epsilon))