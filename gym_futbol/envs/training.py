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
import copy
import math

class DQNAgent:
    def __init__(self, state_size, action_size, num_players):
        self.state_size = state_size
        self.action_size = action_size
        self.num_players = num_players
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
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
        individual_size = int(math.log(self.action_size, self.num_players)) # 4 right now for 2 players bc log_2(16)
        if np.random.rand() <= self.epsilon:
            return (random.randrange(individual_size), random.randrange(individual_size))
        act_values = self.model.predict(state)
        pre_tuple_int = np.argmax(act_values[0])
        return (pre_tuple_int // individual_size, pre_tuple_int % individual_size)  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            print(action)
            target_f[0][action[0]*4 + action[1]] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=2)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

def train_agents(agent_class=DQNAgent, EPISODES=100, dueling_types = []):
    
    env = gym.make('Futbol-v0')
#    state_size = env.observation_space.shape[0] #TODO: dk what it is
    state_size = 30 # (2 players * 2 teams + 1 ball + 1 ball owner array) * (x, y, x_dir, y_dir, mag)
#    action_size = env.action_space.n #TODO: dk what the n here is
    action_size = 16 # 4^2 because 4 actions ^ 2 players
    num_players = 2

    agent_dic_list = []

    temp_agent = DQNAgent(state_size, action_size, num_players)

    temp_dict = dict(model=temp_agent, name=temp_agent.name, score=[], raw_model=copy.deepcopy(temp_agent))

    agent_dic_list.append(temp_dict)

    done = False
    batch_size = 32

    for e in range(EPISODES):

        for agent_dic in agent_dic_list:

            agent = agent_dic['model']

            state = env.reset()
            state = np.reshape(state, [1, state_size])

            for time in range(500):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.memorize(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)

    return agent_dic_list

agent_list = train_agents(EPISODES=1)