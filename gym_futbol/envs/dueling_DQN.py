import random
import gym
import numpy as np
from collections import deque

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda
from keras.optimizers import Adam

class Dueling_DQNAgent:
    def __init__(self, state_size, action_size, dueling_type = 'avg', has_ball = False, team = "left", agent_index = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.dueling_type = dueling_type

        self.model = self._build_model()

        self.name = "Dueling (" + dueling_type + ") DQN Agent" 

        self.has_ball = has_ball
        self.team = team
        self.agent_index = agent_index

    def _set_has_ball(self, has_ball):
        self.has_ball = has_ball

    def _build_model(self):
        
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))

        model.add(Dense(self.action_size + 1, activation='linear'))

        if self.dueling_type == 'avg':
            model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self.action_size,)))
        elif self.dueling_type == 'max':
            model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(self.action_size,)))
        elif self.dueling_type == 'naive':
            model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(self.action_size,)))
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

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

    def load_model(self, name = "Dueling_DQN.h5"):
        self.model = keras.models.load_model(name)
        print("load success")

    def save_model(self, name = "Dueling_DQN.h5"):
        self.model.save(name)
        print("save success")