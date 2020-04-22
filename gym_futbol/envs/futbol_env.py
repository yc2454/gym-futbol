import gym
from gym import error, spaces, utils
import numpy as np
import math
import time
from .action import Action
#import importlib
#moduleName = input('ballowner.py')
#importlib.import_module(moduleName)
from .ballowner import BallOwner
import random
from PIL import Image, ImageDraw

# constants
GOAL_UPPER = 296
GOAL_LOWER = 304
FIELD_LEN = 1000
FIELD_WID = 600
BALL_SPEED = 20
PLARYER_SPEED_W_BALL = 6
PLARYER_SPEED_WO_BALL = 9
GAME_TIME = 600

# get the vector pointing from [coor2] to [coor1] and 
# its magnitude
def get_vec(coor_t, coor_o):
      vec = coor_t[:2] - coor_o[:2]
      vec_mag = math.sqrt(vec[0]**2 + vec[1]**2)
      return vec, vec_mag

# fix the coordinated in the range [0, max]
def lock_in(val, max):
      if val < 0:
            return 0
      elif val > max:
            return max
      else:
            return val

# move the [loc] according to [vec]
def move_by_vec(vec, vec_mag, loc):
      if vec_mag == 0:
            pass
      else:
            loc[0] += loc[4] * (vec[0] * 1.0 / vec_mag)
            loc[1] += loc[4] * (vec[1] * 1.0 / vec_mag)
            loc[2:4] = vec

# get back to field
def get_back(obj):
      back, back_mag = get_vec(np.array([FIELD_LEN/2, FIELD_WID/2]), obj[:2])
      move_by_vec(back, back_mag, obj)

# a normal distribution array
nd = np.random.normal(0, 15, 50)

# twist the direction of vector [vec] a little
# the twist follows normal distribution
def screw_vec(vec, vec_mag):
      i = vec[0] * 1.0 / vec_mag
      j = vec[1] * 1.0 / vec_mag
      seed = random.randint(0, 49)
      twist_angle = (nd[seed] / 180) * math.pi
      twist_sin = math.sin(twist_angle)
      twist_cos = math.cos(twist_angle)
      twisted_i = (j * twist_cos) - (i * twist_sin)
      twisted_j = (i * twist_cos) + (j * twist_sin)
      twisted_vec = np.array([twisted_i * vec_mag, twisted_j * vec_mag])
      return twisted_vec

class FutbolEnv(gym.Env):

      def __init__(self):

            # super(FutbolEnv, self).__init__()

            # data structure to contain the 3 actions
            self.action_space = spaces.Discrete(3)

            # data structure to contain observations the agent would make in one step
            # the 5 values in the array represents: 
            # [0]: x coor, 
            # [1]: y coor, 
            # [2]: target x coor - object x coor
            # [3]: target y coor - object y coor
            # [4]: speed magnitude
            self.observation_space = spaces.Box(low=np.array([[0, 0, -FIELD_LEN, -FIELD_WID, 0]] * 3),
                                                      high=np.array([[FIELD_LEN, FIELD_WID, FIELD_LEN, FIELD_WID, PLARYER_SPEED_W_BALL],
                                                      [FIELD_LEN, FIELD_WID, FIELD_LEN, FIELD_WID, PLARYER_SPEED_W_BALL],
                                                      [FIELD_LEN, FIELD_WID, FIELD_LEN, FIELD_WID, BALL_SPEED]]))
            
            # initial space
            self.init_space = spaces.Box(low=np.array([[FIELD_LEN/2, FIELD_WID/2, 0, 0, 0]] * 3),
                                          high=np.array([[FIELD_LEN/2, FIELD_WID/2, 1.0, 1.0, 0],
                                          [FIELD_LEN/2, FIELD_WID/2, 1.0, 1.0, PLARYER_SPEED_W_BALL],
                                          [FIELD_LEN/2, FIELD_WID/2, 1.0, 1.0, 0]]))
            
            # current time in the match, in seconds
            self.time = 0

            # below are the coordinates and vectors of ball, agent and opponent, 
            # refer to the observation_space comment
            #  
            # position and movement of the ball
            self.ball = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])
            # position and movement of AI player
            self.ai = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])
            # position and movement of opponent player
            self.opp = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])

            # who has the ball
            self.ball_owner = BallOwner.NOONE

            # the scores
            self.ai_score = 0
            self.opp_score = 0

      
      def out(self, obj):
            x = obj[0] < 0 or obj[0] > FIELD_LEN
            y = obj[1] < 0 or obj[1] > FIELD_WID
            return x or y

      
      def score(self):
            ai_in = self.ball[0] <= 0 and (self.ball[1] > GOAL_UPPER and self.ball[1] < GOAL_LOWER)
            opp_in = self.ball[0] >= 1000 and (self.ball[1] > GOAL_UPPER and self.ball[1] < GOAL_LOWER)
            return ai_in or opp_in

      
      def fix(self, player):

            if player == BallOwner.OPP:
                  new_owner = BallOwner.AI
            else:
                  new_owner = BallOwner.OPP

            # relocate the ball to where it went out
            lock_in(self.ball[0], FIELD_LEN)
            lock_in(self.ball[1], FIELD_WID)
            self.ball_owner = new_owner

            # move the other player and the ball together
            self.ball[2:5] = np.array([0,0,0])
            if new_owner == BallOwner.AI:
                  self.ai = self.ball
            else:
                  self.opp = self.ball


      def _take_action(self, action):

            action_type = Action(action)

            # vector from ball to ai player
            b2a, b2a_mag = get_vec(self.ball[:2], self.ai[:2])

            # vector from ball to opp
            b2o, b2o_mag = get_vec(self.ball[:2], self.opp[:2])

            # vector from one of the goal tips to the ball
            # the shooter randomly chooses to aim for the upper or lower goal tip
            prob = random.random()
            if prob > 0.5:
                  rg2b, rg2b_mag = get_vec(np.array([FIELD_LEN, GOAL_LOWER]), self.ball[:2])
                  lg2b, lg2b_mag = get_vec(np.array([0, GOAL_LOWER]), self.ball[:2])
            else:
                  rg2b, rg2b_mag = get_vec(np.array([FIELD_LEN, GOAL_UPPER]), self.ball[:2])
                  lg2b, lg2b_mag = get_vec(np.array([0, GOAL_LOWER]), self.ball[:2])

            # by 0.5 chance, opponent act first, and if the ball is close enough, 
            # kick or take the ball
            opp_act_first = random.random()
            if opp_act_first < 0.5:
                  # first, if the opponent is out af the field, get back
                  if self.out(self.opp):
                        get_back(self.opp)
                  if b2o_mag < 2:
                        tackle_p = random.random()
                        if tackle_p < 0.5 and self.ball_owner != BallOwner.OPP:
                              # tackle for the ball
                              succ_p = random.random()
                              if succ_p <0.3 or self.ball_owner == BallOwner.NOONE:
                                    self.ball = self.opp
                                    self.ball_owner = BallOwner.OPP
                        else:
                              # shoot the ball
                              # blur the shooting direction a little
                              s_lg2b = screw_vec(lg2b, lg2b_mag)
                              self.ball[4] = random.randint(BALL_SPEED - 20, BALL_SPEED + 10)
                              move_by_vec(s_lg2b, lg2b_mag, self.ball)
                              # now the ball belongs to no one
                              self.ball_owner = BallOwner.NOONE
                              # if the ball is out of the field, fix it
                              if self.out(self.ball):
                                    self.fix(BallOwner.OPP)

            # now, the agent acts first
            if action_type == Action.SHOOT:
                  if b2a_mag > 2:
                        pass
                  else:
                        # blur the shooting direction a little
                        s_rg2b = screw_vec(rg2b, rg2b_mag)
                        # set ball speed
                        self.ball[4] = random.randint(BALL_SPEED - 20, BALL_SPEED + 10) * 1.0
                        move_by_vec(s_rg2b, rg2b_mag, self.ball)
                        # now the ball belongs to no one
                        self.ball_owner = BallOwner.NOONE
                        # opponent run towards the ball
                        self.opp[4] = PLARYER_SPEED_WO_BALL
                        move_by_vec(b2o, b2o_mag, self.opp)
                        # if the ball is out of the field, fix it
                        if self.out(self.ball):
                              self.fix(BallOwner.AI)

            # if the ball is close enough to the agent, try taking it
            elif action_type == Action.TACKLE:
                  if ((b2a_mag > 2) or (self.ball_owner == BallOwner.AI)):
                        pass
                  else:
                        succ_p = random.random()
                        if succ_p < 0.3:
                              self.ball = self.ai
                              self.ball_owner = BallOwner.AI
                        else:
                              if self.ball_owner == BallOwner.OPP:
                                    move_by_vec(lg2b, lg2b_mag, self.opp)
                                    self.ball = self.opp
                              else:
                                    move_by_vec(self.ball[2:4], math.sqrt(self.ball[1]**2 + self.ball[2]**2), self.ball)
                                    move_by_vec(b2o, b2o_mag, self.opp)

            elif action_type == Action.RUN:
                  self.ai[4] = 1.0 * random.randint(PLARYER_SPEED_WO_BALL - 4, PLARYER_SPEED_WO_BALL + 4)
                  self.opp[4] = 1.0 * random.randint(PLARYER_SPEED_WO_BALL - 4, PLARYER_SPEED_WO_BALL + 4)
                  o2a, o2a_mag = get_vec(self.opp[:2], self.ai[:2])
                  # if the agent is out, get back
                  if self.out(self.ai):
                        get_back(self.ai)
                  # if agent has the ball, agent run toward the goal, opp chase the 
                  # agent
                  if self.ball_owner == BallOwner.AI:
                        move_by_vec(rg2b, rg2b_mag, self.ai)
                        self.ball = self.ai
                        move_by_vec(-o2a, o2a_mag, self.opp)
                  # if opp has the ball, run towards opp
                  elif self.ball_owner == BallOwner.OPP:
                        move_by_vec(lg2b, lg2b_mag, self.opp)
                        self.ball = self.opp
                        move_by_vec(o2a, o2a_mag, self.ai)
                  # if neither has the ball, run towards the ball
                  else:
                        move_by_vec(self.ball[2:4], math.sqrt(self.ball[2]**2 + self.ball[3]**2), self.ball)
                        move_by_vec(b2o, b2o_mag, self.opp)
                        move_by_vec(b2a, b2a_mag, self.ai)

            else:
                  print('Unrecognized action %d' % action_type)


      def _next_observation(self):
            return np.concatenate((self.ai, self.opp, self.ball)).reshape((3, 5))


      def _get_reward(self, ball, ai, opp):

            ball_advance = ball[:2] - self.ball[:2]
            ball_advance_mag = math.sqrt(ball_advance[0]**2 + ball_advance[1]**2)

            player_adv = ai[:2] - self.ai[:2]
            player_adv_mag = math.sqrt(player_adv[0]**2 + player_adv[1]**2)
            player_adv_mag /= math.sqrt(self.ai[0]**2 + self.ai[1]**2)

            if self.out(self.ai):
                  out_of_field = -3 * (ball_advance_mag + player_adv_mag)
            else:
                  out_of_field = 0

            if self.ball_owner == BallOwner.AI:
                  get_ball = 0.5 * (ball_advance_mag + player_adv_mag)
            else:
                  get_ball = 0

            if self.ball[0] >= FIELD_LEN and (self.ball[1] > GOAL_UPPER and self.ball[1] < GOAL_LOWER):
                  score = 5 * (ball_advance_mag + player_adv_mag)
            else:
                  score = 0

            if self.ball[0] <= 0 and (self.ball[1] > GOAL_UPPER and self.ball[1] < GOAL_LOWER):
                  get_scored = -5 * (ball_advance_mag + player_adv_mag)
            else:
                  get_scored = 0

            return ball_advance_mag + player_adv_mag + get_ball + score + get_scored + out_of_field


      # Execute one time step within the environment
      def step(self, action):
            o_b, o_ai, o_p = self.ball, self.ai, self.opp
            self._take_action(action)
            # calculate reward
            reward = self._get_reward(o_b, o_ai, o_p)
            # if a ball is scored, reset the players and the ball
            if self.score:
                  if self.ball[0] <= 0:
                        self.ai_score += 1
                  else:
                        self.opp_score += 1
                  self.ball = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])
                  self.ai = np.array([FIELD_LEN/2 - 9, FIELD_WID/2, 0, 0, 0])
                  self.opp = np.array([FIELD_LEN/2 + 9, FIELD_WID/2, 0, 0, 0])
                  self.ball_owner = BallOwner.NOONE
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
            self.time = 0
            self.ball = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])
            self.ai = np.array([FIELD_LEN/2 - 90, FIELD_WID/2, 0, 0, 0])
            self.opp = np.array([FIELD_LEN/2 + 90, FIELD_WID/2, 0, 0, 0])
            self.ball_owner = BallOwner.NOONE
            self.ai_score = 0
            self.opp_score = 0
            return self._next_observation()


      # Render the environment to the screen
      def render(self, mode='human', close=False):
            raise NotImplementedError
            
