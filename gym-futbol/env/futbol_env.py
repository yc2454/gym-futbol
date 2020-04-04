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

  def __init__(self, df):
    # super(FutbolEnv, self).__init__()
    # Define action and observation space

    # df stands for data frame, containing the data that we might
    # pass in during initiation.
    ## From online code. May or may not use.
    self.df = df

    # data structure to contain all the actions
    self.action_space = spaces.Discrete(4)

    # data structure to contain observations the agent would make in one step
    self.observation_space = spaces.Box(
      low=0, high=1, shape=(6, 6), dtype=np.float16)

  # observe the data at this step
  def _next_observation(self):
    ...

  def _take_action(self, action):
    ...

  # Execute one time step within the environment
  def step(self, action):
    self._take_action(action)
    self.current_step += 1
    # calculate reward
    reward = 0
    # figure out whether the process is done
    done = False
    obs = self._next_observation()
    return obs, reward, done, {}

  # Reset the state of the environment to an initial state
  def reset(self):
    return self._next_observation()

  # Render the environment to the screen
  def render(self, mode='human', close=False):
    ...