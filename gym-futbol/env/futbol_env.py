import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import numpy as np
from action import Action
from vector import Vector
from team import Team
from field import Field
from player import Player
from ball import Ball
import math
import time

class FutbolEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # super(FutbolEnv, self).__init__()
    # Define action and observation space

    # data structure to contain all the actions
    self.action_space = spaces.Tuple((spaces.Discrete(3),\
                                          spaces.Box(low=0, high=100, shape=1),\
                                          spaces.Box(low=-180, high=180, shape=1),\
                                          spaces.Box(low=-180, high=180, shape=1),\
                                          spaces.Box(low=0, high=100, shape=1),\
                                          spaces.Box(low=-180, high=180, shape=1)))

    # data structure to contain observations the agent would make in one step
    self.observation_space = spaces.Box(low=np.zeros((5, 6))),\ 
    high=np.array([[1000, 600, 1.0, 1.0, 10],\
    [1000, 600, 1.0, 1.0, 10],[1000, 600, 1.0, 1.0, 10],[1000, 600, 1.0, 1.0, 10],\
    [1000, 600, 1.0, 1.0, 10],[1000, 600, 1.0, 1.0, 10]))

    self.init_space = spaces.Box(low=np.zeros((5, 6))), \
    high=np.array([[1000, 600, 1.0, 1.0, 10],\
    [1000, 600, 1.0, 1.0, 10],[1000, 600, 1.0, 1.0, 10],[1000, 600, 1.0, 1.0, 10],\
    [1000, 600, 1.0, 1.0, 10],[1000, 600, 1.0, 1.0, 10]))

  def _take_action(self):
    ...

  def _next_observation(self):
    ...

  def set_init_space(self, low, high):
    self.init_space = spaces.Box(low=np.array(low), high=np.array(high))

  # Execute one time step within the environment
  def step(self, action):
    self._take_action(action)
    self.current_step += 1
    # calculate reward
    reward = 0
    # figure out whether the process is done
    done = False
    # get next observation
    obs = self._next_observation()
    return obs, reward, done, {}

  # Reset the state of the environment to an initial state
  def reset(self):
    return self._next_observation()


  # Render the environment to the screen
  def render(self, mode='human', close=False):
    ...