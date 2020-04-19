import gym
from gym import error, spaces, utils
import numpy as np
import math
import time
from action import Action
from ballowner import BallOwner
import random

# constants
GOAL_UPPER = 296
GOAL_LOWER = 304
FIELD_LEN = 1000
FIELD_WID = 600
BALL_SPEED = 20
PLARYER_SPEED = 9
GAME_TIME = 600

def get_vec(coor1, coor2):
      vec = coor1[:2] - coor2[:2]
      vec_mag = math.sqrt(vec[0]**2 + vec[1]**2)
      return vec, vec_mag

class FutbolEnv(gym.Env):

      def __init__(self):
            # super(FutbolEnv, self).__init__()
            # data structure to contain the 3 actions
            self.action_space = spaces.Discrete(3)
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
            # who has the ball
            self.ball_owner = BallOwner.NOONE


      def _take_action(self, action):

            action_type = Action(action)

            # vector from ball to ai player
            b2a, b2a_mag = get_vec(self.ball[:2], self.ai[:2])

            # vector from one of the goal tips to the ball
            prob = random.random()
            if prob > 0.5:
                  g2b, g2b_mag = get_vec(np.array([FIELD_LEN, GOAL_LOWER]), self.ball[:2])
            else:
                  g2b, g2b_mag = get_vec(np.array([FIELD_LEN, GOAL_UPPER]), self.ball[:2])

            if action_type == Action.SHOOT:
                  if self.ball_owner != BallOwner.AI:
                        pass
                  else:
                        self.ball[0] += self.ball[4] * (g2b[0] / g2b_mag)
                        self.ball[1] += self.ball[4] * (g2b[1] / g2b_mag)
                        self.ball[2:4] = g2b
                        o2b, o2b_mag = get_vec(self.opp[:2], self.ball[:2])
                        self.opp[0] += self.opp[4] * (o2b[0] / o2b_mag)
                        self.opp[1] += self.opp[4] * (o2b[1] / o2b_mag)
                        self.opp[2:4] = o2b

            elif action_type == Action.TACKLE:
                  if ((b2a_mag > 0.5) or (self.ball_owner == BallOwner.AI)):
                        pass
                  else:
                        succ_p = random.random()
                        if succ_p < 0.3:
                              self.ball = self.ai
                              self.ball_owner = BallOwner.AI

            elif action_type == Action.RUN:
                  self.ai[4] = PLARYER_SPEED
                  self.ai[2:4] = b2a
                  self.ai[0] += self.ai[4] * (b2a[0] / b2a_mag)
                  self.ai[0] += self.ai[4] * (b2a[1] / b2a_mag)
                  o2a, o2a_mag = get_vec(self.opp[:2], self.ai[:2])
                  self.opp[4] = PLARYER_SPEED
                  self.opp[0] += self.opp[4] * (o2a[0] / o2a_mag)
                  self.opp[1] += self.opp[4] * (o2a[1] / o2a_mag)
                  self.opp[2:4] = o2a

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

            if self.ball_owner == BallOwner.AI:
                  get_ball = 0.5 * (ball_advance_mag + player_adv_mag)
            else:
                  get_ball = 0

            return ball_advance_mag + player_adv_mag + get_ball


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