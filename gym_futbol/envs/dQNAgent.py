from collections import deque
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda
from keras.optimizers import Adam
import numpy as np
import random
import keras
import math


class DQNAgent:
    def __init__(self, state_size, action_size, num_players=2):
        self.state_size = state_size
        self.action_size = action_size
        self.num_players = num_players
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.name = "DQN Agent"

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 4 right now for 2 players bc log_2(16)
        individual_size = int(math.log(self.action_size, self.num_players))
        if np.random.rand() <= self.epsilon:
            return (random.randrange(individual_size), random.randrange(individual_size))
        act_values = self.model.predict(state)
        pre_tuple_int = np.argmax(act_values[0])
        # returns action
        return (pre_tuple_int // individual_size, pre_tuple_int % individual_size)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            # print(action)
            target_f[0][action[0]*4 + action[1]] = target
            # Filtering out states and targets for training
            states.append(state[0])
            # print(state)
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(
            targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load_model(self, name="DQN_model.h5"):
        self.model = keras.models.load_model(name)
        print("load success")

    def save_model(self, name="DQN_model.h5"):
        self.model.save(name)
        print("save success")
