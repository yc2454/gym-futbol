import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils

from ballowner import BallOwner
from action import Action
from easy_agent import Easy_Agent

import numpy as np
import math
import time
import random
from PIL import Image, ImageDraw

import copy

# constants
FIELD_LEN = 105
FIELD_WID = 68

### goal_size is the size of the goal
GOAL_SIZE = 10
GOAL_UPPER = FIELD_WID / 2 + GOAL_SIZE/2
GOAL_LOWER = FIELD_WID / 2 - GOAL_SIZE/2

BALL_SPEED = 20
PLARYER_SPEED_W_BALL = 6
PLARYER_SPEED_WO_BALL = 9
GAME_TIME = 60
GOAL_REWARD = 2000
BALL_ADV_REWARD_BASE = 7000
PLAYER_ADV_REWARD_BASE = 2000
OUT_OF_FIELD_PENALTY = -600
BALL_CONTROL = 300

# size of each time step 
# step_size=1 means every step is 1s 
STEP_SIZE = 0.3

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

      def __init__(self, length = FIELD_LEN, width = FIELD_WID, goal_size = GOAL_SIZE, game_time = GAME_TIME, player_speed = PLARYER_SPEED_W_BALL, ball_speed = BALL_SPEED, Debug = False):

            self.length = length
            self.width = width
            self.goal_size = goal_size
            self.goal_up = width / 2 + goal_size / 2
            self.goal_down = width / 2 - goal_size / 2
            self.game_time = game_time
            self.player_speed = player_speed
            self.ball_speed = ball_speed

            self.Debug = Debug

            # data structure to contain the 3 actions
            self.action_space = spaces.Discrete(3)

            # data structure to contain observations the agent would make in one step
            # the 5 values in the array represents: 
            # [0]: x coor, 
            # [1]: y coor, 
            # [2]: target x coor - object x coor
            # [3]: target y coor - object y coor
            # [4]: speed magnitude
            self.observation_space = spaces.Box(low=np.array([[0, 0, 0, 0, 0]] * 3),
                                                high=np.array([[length, width, length, width, player_speed],
                                                      [length, width, length, width, player_speed],
                                                      [length, width, length, width, ball_speed]]),
                                                dtype=np.float64)


            ### moved some parameter out of reset()
            
            self.ai_index = 0
            self.opp_index = 1
            self.ball_index = 2
            
            self.obs = self.reset()


             # opp easy agent
            self.opp_agent = Easy_Agent('opp', self.obs, self.opp_index, self.ball_index, 'right', (self.ball_owner == BallOwner.OPP), self.length, self.width, self.goal_size, shoot_range = 10)

            # ai easy agent
            self.ai_agent = Easy_Agent('ai', self.obs, self.ai_index, self.ball_index, 'left', (self.ball_owner == BallOwner.AI), self.length, self.width, self.goal_size, shoot_range = 10)

            
      
      # Reset the state of the environment to an initial state
      def reset(self):

            # below are the coordinates and vectors of ball, agent and opponent, 
            # refer to the observation_space comment
            #  
            # position and movement of the ball
            self.ball = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])
            # position and movement of AI player
            self.ai = np.array([FIELD_LEN/2 - 9, FIELD_WID/2, 0, 0, 0])
            # position and movement of opponent player
            self.opp = np.array([FIELD_LEN/2 + 9, FIELD_WID/2, 0, 0, 0])

            self.obs = np.concatenate((self.ai, self.opp, self.ball)).reshape((3, 5))

            self.ai = self.obs[self.ai_index]
            self.opp = self.obs[self.opp_index]
            self.ball = self.obs[self.ball_index]

            # who has the ball
            self.ball_owner = BallOwner.NOONE
            self.last_ball_owner = BallOwner.NOONE

            # current time in the match, in seconds
            self.time = 0

            # the scores
            self.ai_score = 0
            self.opp_score = 0

            return self.obs
           

      def _next_observation(self):
            return self.obs


      # Render the environment to the screen
      def render(self, mode='human', close=False):

            fig, ax = plt.subplots()
            ax.set_xlim(0, self.length)
            ax.set_ylim(0, self.width)

            # ai
            ai_x, ai_y, _, _, _ = self.obs[self.ai_index]
            ax.plot(ai_x,ai_y, color = 'red', marker='o', markersize=12, label='ai')

            # opp
            opp_x, opp_y, _, _, _ = self.obs[self.opp_index]
            ax.plot(opp_x, opp_y, color = 'blue', marker='o', markersize=12, label='opp')

            # ball
            ball_x, ball_y, _, _, _ = self.obs[self.ball_index]
            ax.plot(ball_x, ball_y, color = 'green', marker='o', markersize=8, label='ball')

            ax.legend()
            plt.show()

      # set agent's vector observation based on action_type, and ball owner
      # data structure to contain observations the agent would make in one step
          # the 5 values in the array represents: 
          # [0]: x coor, 
          # [1]: y coor, 
          # [2]: target x coor - object x coor
          # [3]: target y coor - object y coor
          # [4]: speed magnitude
      def _set_vector_observation(self, agent, action_type): 

            action = Action(action_type)

            agent_observation = self.obs[agent.agent_index]

            ball_observation = self.obs[self.ball_index]
           
            target_padding = 1

            target_y = random.randint(self.goal_down + target_padding, self.goal_up - target_padding)            
            
            if agent.has_ball: 

                  # has ball and intercept, zeros agent's target x, y, mag
                      # agent_vec = x, y, 0, 0, 0
                      # ball_vec = x, y, 0, 0, 0
                      # ball owener not change
                  if action == Action.intercept: 

                        agent_observation[2:5] = np.array([0,0,0])
                        ball_observation[2:5] = np.array([0,0,0])

                        if self.Debug: 
                              print(agent.name + " with ball: intercept")
                    
                  # has ball and run toward goal (with ball)
                      # agent_vec = x, y, tx, ty, m
                      # ball_vec = x, y, tx, ty, m
                      # ball owener not change
                  elif action == Action.run: 

                        agent_observation[4] = 1.0 * random.randint(self.player_speed - 2, self.player_speed + 2)
                        
                        if self.Debug: 
                              print(agent.name + " with ball: run to goal")

                        # print(agent_observation)

                        if agent.team == 'right': 
                              agent_observation[2:4], _ = get_vec(np.array([0, target_y]), agent_observation[:2])
                        else: 
                              agent_observation[2:4], _ = get_vec(np.array([self.length, target_y]), agent_observation[:2])
                        
                        # print(agent_observation)

                        self.obs[self.ball_index] = agent_observation
                  
                  # has ball and shoot toward goal, zeros agent's target x, y, mag
                      # agent_vec = x, y, 0, 0, 0
                      # ball_vec = x, y, tx, ty, m
                      # ball owener change
                  elif action == Action.shoot: 

                        ball_observation[4] = random.randint(self.ball_speed - 16, self.ball_speed) * 1.0

                        if self.Debug: 
                              print(agent.name + " with ball: shoot")

                        ### changed, as screw_vec is not currently working
                        if agent.team == 'right': 
                              # goal_to_ball, goal_to_ball_mag = get_vec(np.array([0, target_y]), ball_observation[:2])
                              # ball_observation[2:4] = screw_vec(goal_to_ball, goal_to_ball_mag)
                              ball_observation[2:4], _ = get_vec(np.array([0, target_y]), ball_observation[:2])

                        else: 
                              # goal_to_ball, goal_to_ball_mag = get_vec(np.array([self.length, target_y]), ball_observation[:2])
                              # ball_observation[2:4] = screw_vec(goal_to_ball, goal_to_ball_mag)
                              ball_observation[2:4], _ = get_vec(np.array([self.length, target_y]), ball_observation[:2])


                        agent.has_ball = False
                        self.ball_owner = BallOwner.NOONE
                        self.last_ball_owner = BallOwner(agent.agent_index)
                        agent_observation[2:5] = np.array([0,0,0])

                  else: 

                        print('Unrecognized action %d' % action_type)

            else: 

                  ball_to_agent, ball_to_agent_magnitude = get_vec(ball_observation[:2], agent_observation[:2])

                  # no ball and intercept
                      # if close, try get ball, stop agent, zeros agent's target x, y, mag
                          # intercept success
                              # agent_vec = x, y, 0, 0, 0
                              # ball_vec = x, y, 0, 0, 0
                              # ball owener change
                          # intercept failed
                              # agent_vec = x, y, 0, 0, 0
                              # ball_vec = x', y', tx', ty', m' (no change)
                              # ball owener not change
                      # if not close, stop agent, zeros agent's target x, y, mag
                          # agent_vec = x, y, 0, 0, 0
                          # ball_vec = x', y', tx', ty', m' (no change)
                          # ball owener not change
                  if action == Action.intercept: 

                        if self.Debug: 
                              print(agent.name + " no ball: intercept")

                        agent_observation[2:5] = np.array([0,0,0])

                        intercept_distance = 2

                        if ball_to_agent_magnitude > 2:

                              if self.Debug:
                                    print(agent.name + " too far, intercept failed")

                        else: 

                              intercept_success = random.random() <= 0.3

                              if intercept_success: 

                                    ball_observation[2:5] = np.array([0, 0, 0])
                                    ball_observation[:2] = agent_observation[:2]
                                    self.ball_owner = BallOwner(agent.agent_index)
                                    self.last_ball_owner = BallOwner(agent.agent_index)

                                    if self.Debug:
                                          print(agent.name + " lucky, intercept success")

                              else: 

                                    if self.Debug:
                                          print(agent.name + " unlucky, intercept failed")

                  # no ball and run toward ball 
                      # agent_vec = x, y, tx'', ty'', m''
                      # ball_vec = x', y', tx', ty', m' (no change)
                      # ball owener not change
                  elif action == Action.run: 

                        if self.Debug: 
                              print(agent.name + " no ball: run to ball")

                        agent_observation[4] = 1.0 * random.randint(self.player_speed - 2, self.player_speed + 2)

                        agent_observation[2:4] = ball_to_agent
              
                  # no ball and shoot, stop agent
                      # agent_vec = x, y, 0, 0, 0
                      # ball_vec = x', y', tx', ty', m' (no change)
                      # ball owener not change
                  elif action == Action.shoot:

                        if self.Debug: 
                              print(agent.name + " no ball: shoot and stop")

                        agent_observation[2:5] = np.array([0, 0, 0])
                  
                  else: 
                        print('Unrecognized action %d' % action_type)
                        

            return agent_observation

      ### changed
      # action_set=True means the action is given in the step function
      # action_set=False means the action need to be deduced from agent
      def _agent_set_vector_observation(self, agent, action_set = False, action_type = 0):

            if self.ball_owner == BallOwner(agent.agent_index): 
                  agent_has_ball = True
            else:
                  agent_has_ball = False

            if not action_set:
                  action_type = agent.get_action_type(self.obs, agent_has_ball)
            else:
                  agent._set_has_ball(agent_has_ball)

            self.obs[agent.agent_index] = self._set_vector_observation(agent, action_type)

      # move the [loc] according to [vec]
      # notice the STEP_SIZE
      def _step_observation(self, observation):

            tx, ty = observation[2:4]
            vec_mag = math.sqrt(tx**2 + ty**2)

            if vec_mag == 0:
                  pass
            else:
                  observation[0] += observation[4] * (tx * STEP_SIZE / vec_mag)
                  observation[1] += observation[4] * (ty * STEP_SIZE / vec_mag)

            
      def out(self, obj):
            x = obj[0] < 0 or obj[0] > FIELD_LEN
            y = obj[1] < 0 or obj[1] > FIELD_WID
            return x or y

      
      def score(self):
            ai_in = self.ball[0] <= 0 and (self.ball[1] > GOAL_LOWER and self.ball[1] < GOAL_UPPER)
            opp_in = self.ball[0] >= FIELD_LEN and (self.ball[1] > GOAL_LOWER and self.ball[1] < GOAL_UPPER)
            return ai_in or opp_in

      #### need further modification
      def fix(self, player):

            if player == BallOwner.OPP:
                  new_owner = BallOwner.AI
            else:
                  new_owner = BallOwner.OPP

            # relocate the ball to where it went out
            self.ball[0] = lock_in(self.ball[0], FIELD_LEN)
            self.ball[1] = lock_in(self.ball[1], FIELD_WID)
            self.ball_owner = new_owner

            # move the other player and the ball together
            self.ball[2:5] = np.array([0,0,0])
            if new_owner == BallOwner.AI:
                  self.obs[self.ai_index] = self.ball
            else:
                  self.obs[self.opp_index] = self.ball


      def _step_vector_observations(self, observations):

            if self.Debug: 
                print("vector before step:")
                print(self.obs)

            for observation in observations: 

                  self._step_observation(observation)

            if self.Debug: 
                print("vector after step:")
                print(self.obs)
            

      def out_of_field(self):
            x = self.ball[0] < 0 or self.ball[0] > self.length
            y = self.ball[1] < 0 or self.ball[1] > self.width
            return x or y
                  
      def step(self, ai_action_type):

            o_b = copy.copy(self.ball)
            o_ai = copy.copy(self.ai)
            o_p = copy.copy(self.opp)

            self._agent_set_vector_observation(self.opp_agent)

            self._agent_set_vector_observation(self.ai_agent, action_set = True, action_type = ai_action_type)

            self._step_vector_observations(self.obs)

            # calculate reward
            reward = self._get_reward(o_b, o_ai, o_p)

            if self.score():
                  if self.ball[0] <= 0:
                        self.opp_score += 1
                  else:
                        self.ai_score += 1

                  if self.Debug: 
                        print("Score!!!")
                        print("ai : opp = " + str(self.ai_score) + " : " + str(self.opp_score))

                  ### changed from simple reset()
                  self.ball = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0])
                  self.ai = np.array([FIELD_LEN/2 - 9, FIELD_WID/2, 0, 0, 0])
                  self.opp = np.array([FIELD_LEN/2 + 9, FIELD_WID/2, 0, 0, 0])
                  self.obs = np.concatenate((self.ai, self.opp, self.ball)).reshape((3, 5))
                  self.ball_owner = BallOwner.NOONE
                  self.last_ball_owner = BallOwner.NOONE

                  self.ai = self.obs[self.ai_index]
                  self.opp = self.obs[self.opp_index]
                  self.ball = self.obs[self.ball_index]

            
            if self.out(self.ball):

                  self.fix(self.last_ball_owner)

                  if self.Debug: 
                        print("fix out of box ball")


            # figure out whether the game is over
            if self.time == GAME_TIME:
                  done = True
            else:
                  done = False
            
            # one second passes in the game
            self.time += 1
            return self.obs, reward, done, {}

    
      def _get_reward(self, ball, ai, opp):

            ball_adv = self.ball[0] - ball[0]

            player_adv = self.ai[0] - ai[0] 

            ball_adv_r = (ball_adv/FIELD_LEN) * BALL_ADV_REWARD_BASE
            player_adv_r = (player_adv/FIELD_LEN) * PLAYER_ADV_REWARD_BASE

            if self.out(self.ai):
                  out_of_field = OUT_OF_FIELD_PENALTY
            else:
                  out_of_field = 0

            if self.ball_owner == BallOwner.AI:
                  get_ball = BALL_CONTROL
            else:
                  get_ball = 0

            if self.score() and self.ball[0] >= FIELD_LEN:
                  score = GOAL_REWARD
                  if self.Debug:
                        print("scored reward")
            else:
                  score = 0

            if self.score() and self.ball[0] <= 0:
                  get_scored = -GOAL_REWARD
                  if self.Debug:
                        print("get scored reward")
            else:
                  get_scored = 0

            return ball_adv_r + player_adv_r + get_ball + score + get_scored + out_of_field
