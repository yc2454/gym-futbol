import gym
from gym import error, spaces, utils
import numpy as np
import math
import time
from action import Action

class FutbolEnv(gym.Env):

      def __init__(self):
        # super(FutbolEnv, self).__init__()
        # Define action and observation space

        # data structure to contain all the actions
        self.action_space = spaces.Tuple((spaces.Discrete(4),
                            spaces.Box(low=0, high=100, shape=1),
                            spaces.Box(low=-180, high=180, shape=1),
                            spaces.Box(low=-180, high=180, shape=1),
                            spaces.Box(low=0, high=100, shape=1),
                            spaces.Box(low=-180, high=180, shape=1)))
        # data structure to contain observations the agent would make in one step
        self.observation_space = spaces.Box(low=np.zeros((5, 6)),
                                            high=np.array([[1000, 600, 1.0, 1.0, 10],
                                                          [1000, 600, 1.0, 1.0, 10],
                                                          [1000, 600, 1.0, 1.0, 10],
                                                          [1000, 600, 1.0, 1.0, 10],
                                                          [1000, 600, 1.0, 1.0, 10],
                                                          [1000, 600, 1.0, 1.0, 10]]))
        # initial space
        # TODO: be more specific
        self.init_space = spaces.Box(low=np.zeros((5, 6)), 
                                    high=np.array([[1000, 600, 1.0, 1.0, 10],
                                                  [1000, 600, 1.0, 1.0, 10],
                                                  [1000, 600, 1.0, 1.0, 10],
                                                  [1000, 600, 1.0, 1.0, 10],
                                                  [1000, 600, 1.0, 1.0, 10],
                                                  [1000, 600, 1.0, 1.0, 10]]))
        # current time in the match
        self.time = 0


      def _take_action(self, action):
            action_type = action[0]
            if action_type == Action.SHOOT:
                  pass
            elif action_type == Action.TACKLE:
                  pass
            elif action_type == Action.PASS:
                  pass
            elif action_type == Action.RUN:
                  pass
            else:
                  print('Unrecognized action %d' % action_type)


      def _next_observation(self):
            return self.init_space.sample()

      def _get_reward(self, action):
            pass

      def set_init_space(self, low, high):
            self.init_space = spaces.Box(low=np.array(low), high=np.array(high))

      # Execute one time step within the environment
      def step(self, action):
            self._take_action(action)
            # calculate reward
            reward = 0
            # figure out whether the process is done
            done = False
            # get next observation
            obs = self._next_observation()
            # one second passes in the game
            self.time += 1
            return obs, reward, done, {}

      # Reset the state of the environment to an initial state
      def reset(self):
            return self.init_space.sample()


      # Render the environment to the screen
      def render(self, mode='human', close=False):
            pass