import gym
from gym import error, spaces, utils
import numpy as np
import math
import time
from action import Action
import random

# constants
GOAL_UPPER = 296
GOAL_LOWER = 304
FIELD_LEN = 1000
FIELD_WID = 600
BALL_SPEED = 20
PLARYER_SPEED = 9
GAME_TIME = 600

class FutbolEnv(gym.Env):

      def __init__(self):
        # super(FutbolEnv, self).__init__()
        # Define action and observation space

        # data structure to contain the 4 actions
        self.action_space = spaces.Discrete(4)
        # data structure to contain observations the agent would make in one step
        # the 5 values in the array represents: x coor, y coor, vector direction
        # sine, vector direction cosine, vector magnitude
        self.observation_space = spaces.Box(low=np.array([[0, 0, 0, 0, 0]] * 2), 
                                          high=np.array([[FIELD_LEN, FIELD_WID, 1.0, 1.0, PLARYER_SPEED],
                                          [FIELD_LEN, FIELD_WID, 1.0, 1.0, BALL_SPEED]]))
        # initial space
        self.init_space = spaces.Box(low=np.array([[FIELD_LEN/2, FIELD_WID/2, 0, 0, 0]] * 2), 
                                    high=np.array([[FIELD_LEN/2, FIELD_WID/2, 1.0, 1.0, 0],
                                    [FIELD_LEN/2, FIELD_WID/2, 1.0, 1.0, 0]]))
        # current time in the match, in seconds
        self.time = 0
        # position and movement of the ball
        self.ball = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])
        # position and movement of AI player
        self.ai = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])
        # position and movement of opponent player
        self.opp = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])


      def _take_action(self, action):

            action_type = action

            # vector from ball to ai player
            b2p = self.ball[:2] - self.ai[:2]
            b2p_mag = math.sqrt(b2p[0]**2 + b2p[1]**2)

            # vector from one of the goal tips to the ball
            prob = random.random()
            if prob > 0.5:
                  b2g = np.array([FIELD_LEN, GOAL_LOWER]) - self.ball[:2]
            else:
                  b2g = np.array([FIELD_LEN, GOAL_UPPER]) - self.ball[:2]
            b2g_mag = math.sqrt(b2g[0]**2 + b2g[1]**2)

            if action_type == Action.SHOOT:
                  if b2p_mag > 0.5:
                        pass
                  else:
                        self.ball[0] += self.ball[4] * (b2g[0] / b2g_mag)
                        self.ball[1] += self.ball[4] * (b2g[1] / b2g_mag)
                        self.ball[2:3] = b2g

            elif action_type == Action.TACKLE:
                  if b2p_mag > 0.5:
                        pass
                  else:
                        succ_p = random.random()
                        if succ_p < 0.3:
                              self.ball = self.ai

            elif action_type == Action.RUN:
                  self.ai[:2] = b2p

            else:
                  print('Unrecognized action %d' % action_type)


      def _next_observation(self):
            return np.concatenate((self.opp, self.ball)).reshape((2, 5))


      def _get_reward(self, ball, ai, opp):

            ball_advance = ball[:2] - self.ball[:2]
            ball_advance_mag = math.sqrt(ball_advance[0]**2 + ball_advance[1]**2)
            player_adv = ai[:2] - self.ai[:2]
            player_adv_mag = math.sqrt(player_adv[0]**2 + player_adv[1]**2)
            player_adv_mag /= math.sqrt(self.ai[0]**2 + self.ai[1]**2)

            return ball_advance_mag + player_adv_mag


      # Execute one time step within the environment
      def step(self, action):
            o_b, o_ai, o_p = self.ball, self.ai, self.opp
            self._take_action(action)
            # calculate reward
            reward = self._get_reward(o_b, o_ai, o_p)
            # figure out whether the game is over
            if self.time == GAME_TIME:
                  done = True
            else:
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