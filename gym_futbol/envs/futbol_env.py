import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils

from .ballowner import BallOwner
from .action import Action
from .easy_agent import Easy_Agent

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

# range that the defence will cause pressure to the 
# attacker
PRESSURE_RANGE = 2

SHOOT_SPEED = 20
PASS_SPEED = 10
PLARYER_SPEED_W_BALL = 6
PLARYER_SPEED_WO_BALL = 9
GAME_TIME = 40
GOAL_REWARD = 2000
BALL_ADV_REWARD_BASE = 700
PLAYER_ADV_REWARD_BASE = 1500
OUT_OF_FIELD_PENALTY = -600
BALL_CONTROL = 300
DEFENCE_REWARD_BASE = 800

# size of each time step 
# step_size=1 means every step is 1s 
STEP_SIZE = 0.2

# missing from target value of shooting, represented by the 
# standard deviation of shooting angle
NORMAL_MISS = 5
UNDER_DEFENCE_MISS = 10

# maximum intercept success probability
MAX_INTERCEPT_PROB = 0.9
MAX_INTERCEPT_DIST = 2
MIN_INTERCEPT_DIST = 1


def base_convert(i, b):
    if type(i) == tuple:
        return list(i)
    
    result = []
    while i > 0:
        result.insert(0, i % b)
        i = i // b
    for j in range(5-len(result)):
        result.insert(0, 0)
    return result

# get the vector pointing from [coor2] to [coor1] and 
# its magnitude
def get_vec(coor_t, coor_o):
      vec = coor_t[:2] - coor_o[:2]
      vec_mag = math.sqrt(vec[0]**2 + vec[1]**2)
      return vec, vec_mag

# fix coordinates in the range [0, max]
def lock_in(val, max):
      if val < 0:
            return 0
      elif val > max:
            return max
      else:
            return val

def bigger_than(listy, v):
    return len(list(filter(lambda x: v > x, listy)))

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


# twist the direction of vector [vec] a little
# the twist follows normal distribution
def screw_vec(vec, vec_mag, accuracy=NORMAL_MISS):
      # a Gaussian distribution with std=accuracy
      nd = np.random.normal(0, accuracy, 10)
      # swing the vector by an angle randomly chosen from [nd]
      cos = vec[0] * 1.0 / vec_mag
      sin = vec[1] * 1.0 / vec_mag
      seed = random.randint(0, 9)
      swing_angle = (nd[seed] / 180) * math.pi
      swing_sin = math.sin(swing_angle)
      swing_cos = math.cos(swing_angle)
      # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
      # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
      twisted_cos = (cos * swing_cos) - (sin * swing_sin)
      twisted_sin = (sin * swing_cos) + (cos * swing_sin)
      twisted_vec = np.array([twisted_cos * vec_mag, twisted_sin * vec_mag])
      return twisted_vec

# calculate the probabiity of successfully taking the ball
# if distance [d] < [d1], probability = MAX_INTERCEPT_PROB
# if [d] > d2, probability = 0
# probability decreases linearly between [d1] and [d2]
def intercept_chance(d, d1, d2):
      if d < d1:
            return MAX_INTERCEPT_PROB
      elif d >= d1 and d <= d2:
            k = MAX_INTERCEPT_PROB / (d1 - d2)
            return k * (d - d2)
      else:
            return 0


class FutbolEnv(gym.Env):

      def __init__(self, length = FIELD_LEN, width = FIELD_WID, goal_size = GOAL_SIZE, 
                   game_time = GAME_TIME, player_speed = PLARYER_SPEED_W_BALL, 
                   shoot_speed = SHOOT_SPEED, Debug = False, pressure_range = PRESSURE_RANGE,
                   one_goal_end = False, action_as_int = True, only_reward_goal = False, random_opp = False):

            # constants 
            self.length = length
            self.width = width
            self.goal_size = goal_size
            self.goal_up = width / 2 + goal_size / 2
            self.goal_down = width / 2 - goal_size / 2
            self.game_time = game_time
            self.player_speed = player_speed
            self.shoot_speed = shoot_speed
            self.one_goal_end = one_goal_end

            self.Debug = Debug
            self.action_as_int = action_as_int
            self.only_reward_goal = only_reward_goal
            self.random_opp = random_opp
            
            if self.action_as_int:
                self.action_space = spaces.Discrete(4**5)

            else:
                # data structure to contain the 3 actions
                self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4), spaces.Discrete(4), spaces.Discrete(4), spaces.Discrete(4)))
  

            # data structure to contain the 3 actions
            self.action_space = spaces.Tuple( (spaces.Discrete(4), spaces.Discrete(4), spaces.Discrete(4), spaces.Discrete(4), spaces.Discrete(4)) )

            # data structure to contain observations the agent would make in one step
            # the 5 values in the array represents: 
            # [0]: x coor, 
            # [1]: y coor, 
            # [2]: target x coor - object x coor
            # [3]: target y coor - object y coor
            # [4]: speed magnitude
            self.observation_space = spaces.Box(low=np.array([[0, 0, -length, -width, 0, 0]] * 11),
                                                high=np.array([
                                                      [length, width, length, width, player_speed, 10], # p1t1
                                                      [length, width, length, width, player_speed, 10], # p2t1
                                                      [length, width, length, width, player_speed, 10], # p3t1
                                                      [length, width, length, width, player_speed, 10], # p4t1
                                                      [length, width, length, width, player_speed, 10], # p5t1
                                                      [length, width, length, width, player_speed, 10], # p1t2
                                                      [length, width, length, width, player_speed, 10], # p2t2
                                                      [length, width, length, width, player_speed, 10], # p3t2
                                                      [length, width, length, width, player_speed, 10], # p4t2
                                                      [length, width, length, width, player_speed, 10], # p5t2
                                                      [length, width, length, width, shoot_speed,  666] # ball
                                                      ]),
                                                dtype=np.float64)


            ### moved some parameter out of reset()
            self.ai_1_index = 0
            self.ai_2_index = 1
            self.ai_3_index = 2
            self.ai_4_index = 3
            self.ai_5_index = 4
            
            self.opp_1_index = 5
            self.opp_2_index = 6
            self.opp_3_index = 7
            self.opp_4_index = 8
            self.opp_5_index = 9
            self.ball_index = 10
            self.ball_owner_array_index = 11
            
            self.obs = self.reset()

            # current time in the match, in seconds
            self.time = 0

             # opp easy agent 1
            self.opp_1_agent = Easy_Agent('opp_1', self.obs, self.opp_1_index, [6,7,8,9], self.ball_index, 'right', (self.ball_owner == BallOwner.OPP_1), self.length, self.width, self.goal_size, shoot_range = 20)
            
            self.opp_2_agent = Easy_Agent('opp_2', self.obs, self.opp_2_index, [5,7,8,9], self.ball_index, 'right', (self.ball_owner == BallOwner.OPP_2), self.length, self.width, self.goal_size, shoot_range = 20)
            
            self.opp_3_agent = Easy_Agent('opp_3', self.obs, self.opp_3_index, [5,6,8,9], self.ball_index, 'right', (self.ball_owner == BallOwner.OPP_3), self.length, self.width, self.goal_size, shoot_range = 20)
            
            self.opp_4_agent = Easy_Agent('opp_4', self.obs, self.opp_4_index, [5,6,7,9], self.ball_index, 'right', (self.ball_owner == BallOwner.OPP_4), self.length, self.width, self.goal_size, shoot_range = 20)
            
            self.opp_5_agent = Easy_Agent('opp_5', self.obs, self.opp_5_index, [5,6,7,8], self.ball_index, 'right', (self.ball_owner == BallOwner.OPP_5), self.length, self.width, self.goal_size, shoot_range = 20)

            # ai easy agent
            self.ai_1_agent = Easy_Agent('ai_1', self.obs, self.ai_1_index, [1,2,3,4], self.ball_index, 'left', (self.ball_owner == BallOwner.AI_1), self.length, self.width, self.goal_size, shoot_range = 20)
            self.ai_2_agent = Easy_Agent('ai_2', self.obs, self.ai_2_index, [0,2,3,4], self.ball_index, 'left', (self.ball_owner == BallOwner.AI_2), self.length, self.width, self.goal_size, shoot_range = 20)
            self.ai_3_agent = Easy_Agent('ai_3', self.obs, self.ai_3_index, [0,1,3,4], self.ball_index, 'left', (self.ball_owner == BallOwner.AI_3), self.length, self.width, self.goal_size, shoot_range = 20)
            self.ai_4_agent = Easy_Agent('ai_4', self.obs, self.ai_4_index, [0,1,2,4], self.ball_index, 'left', (self.ball_owner == BallOwner.AI_4), self.length, self.width, self.goal_size, shoot_range = 20)
            self.ai_5_agent = Easy_Agent('ai_5', self.obs, self.ai_5_index, [0,1,2,3], self.ball_index, 'left', (self.ball_owner == BallOwner.AI_5), self.length, self.width, self.goal_size, shoot_range = 20)
            
            
      # Reset the state of the environment to an initial state
      def reset(self):

            # below are the coordinates and vectors of ball, agent and opp_1onent, 
            # refer to the observation_space comment
            #  
            # position and movement of the ball
            self.ball = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0, 0])
            # position and movement of the first AI player
            self.ai_1 = np.array([FIELD_LEN/2 - 9, FIELD_WID/2, 0, 0, 0, 0])
            # position and movement of the rest of AI players
            self.ai_2 = np.array([FIELD_LEN/6, FIELD_WID/2 - FIELD_WID/6, 0, 0, 0, 0])
            self.ai_3 = np.array([FIELD_LEN/6, FIELD_WID/2 + FIELD_WID/6, 0, 0, 0, 0])
            self.ai_4 = np.array([FIELD_LEN/3, FIELD_WID/2 - FIELD_WID/6, 0, 0, 0, 0])
            self.ai_5 = np.array([FIELD_LEN/3, FIELD_WID/2 + FIELD_WID/6, 0, 0, 0, 0])
            
            # position and movement of the first opponent player
            self.opp_1 = np.array([FIELD_LEN/2 + 9, FIELD_WID/2 + 5, 0, 0, 0, 0])
            # position and movement of the rest of the opponent players
            self.opp_2 = np.array([5 * FIELD_LEN/6, FIELD_WID/2 - FIELD_WID/6, 0, 0, 0, 0])
            self.opp_3 = np.array([5 * FIELD_LEN/6, FIELD_WID/2 + FIELD_WID/6, 0, 0, 0, 0])
            self.opp_4 = np.array([4 * FIELD_LEN/6, FIELD_WID/2 - FIELD_WID/6, 0, 0, 0, 0])
            self.opp_5 = np.array([4 * FIELD_LEN/6, FIELD_WID/2 + FIELD_WID/6, 0, 0, 0, 0])
            
            # array representing who has the ball
            # index 0-10 represents players, no one, respectively
            # value 0 means doesn't have ball, 10 means have ball
            self.ball_owner_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            self.obs = np.concatenate((self.ai_1, self.ai_2, self.ai_3, self.ai_4, self.ai_5, self.opp_1, self.opp_2, self.opp_3, self.opp_4, self.opp_5, self.ball)).reshape((11, 6)) # change the reshape when theres diff #s of players
            self.ai_1 = self.obs[self.ai_1_index]
            self.ai_2 = self.obs[self.ai_2_index]
            self.ai_3 = self.obs[self.ai_3_index]
            self.ai_4 = self.obs[self.ai_4_index]
            self.ai_5 = self.obs[self.ai_5_index]
            
            self.opp_1 = self.obs[self.opp_1_index]
            self.opp_2 = self.obs[self.opp_2_index]
            self.opp_3 = self.obs[self.opp_3_index]
            self.opp_4 = self.obs[self.opp_4_index]
            self.opp_5 = self.obs[self.opp_5_index]
            
            self.ball = self.obs[self.ball_index]
            # self.ball_owner_array = self.obs[self.ball_owner_array_index]

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

            _, ax = plt.subplots()
            ax.set_xlim(0, self.length)
            ax.set_ylim(0, self.width)

            # ai
            ai_1_x, ai_1_y, _, _, _, _ = self.obs[self.ai_1_index]
            ai_2_x, ai_2_y, _, _, _, _ = self.obs[self.ai_2_index]
            ai_3_x, ai_3_y, _, _, _, _ = self.obs[self.ai_3_index]
            ai_4_x, ai_4_y, _, _, _, _ = self.obs[self.ai_4_index]
            ai_5_x, ai_5_y, _, _, _, _ = self.obs[self.ai_5_index]
            ax.plot(ai_1_x,ai_1_y, color = 'red', marker='o', markersize=12, label='ai')
            ax.plot(ai_2_x,ai_2_y, color = 'red', marker='o', markersize=12, label='ai')
            ax.plot(ai_3_x,ai_3_y, color = 'red', marker='o', markersize=12, label='ai')
            ax.plot(ai_4_x,ai_4_y, color = 'red', marker='o', markersize=12, label='ai')
            ax.plot(ai_5_x,ai_5_y, color = 'red', marker='o', markersize=12, label='ai')

            # opp
            opp_1_x, opp_1_y, _, _, _, _ = self.obs[self.opp_1_index]
            opp_2_x, opp_2_y, _, _, _, _ = self.obs[self.opp_2_index]
            opp_3_x, opp_3_y, _, _, _, _ = self.obs[self.opp_3_index]
            opp_4_x, opp_4_y, _, _, _, _ = self.obs[self.opp_4_index]
            opp_5_x, opp_5_y, _, _, _, _ = self.obs[self.opp_5_index]
            ax.plot(opp_1_x, opp_1_y, color = 'blue', marker='o', markersize=12, label='opp')
            ax.plot(opp_2_x, opp_2_y, color = 'blue', marker='o', markersize=12, label='opp')
            ax.plot(opp_3_x, opp_3_y, color = 'blue', marker='o', markersize=12, label='opp')
            ax.plot(opp_4_x, opp_4_y, color = 'blue', marker='o', markersize=12, label='opp')
            ax.plot(opp_5_x, opp_5_y, color = 'blue', marker='o', markersize=12, label='opp')
            # ball
            ball_x, ball_y, _, _, _, _ = self.obs[self.ball_index]
            ax.plot(ball_x, ball_y, color = 'green', marker='o', markersize=8, label='ball')

            ax.legend()
            plt.show()


      def defence_near(self, agent):
            #### changed agent.observation to agent.agent_observation
            if agent.team == 'left':
                  _, o1_dis = get_vec(self.opp_1[:2], agent.agent_observation[:2])
                  _, o2_dis = get_vec(self.opp_2[:2], agent.agent_observation[:2])
                  _, o3_dis = get_vec(self.opp_3[:2], agent.agent_observation[:2])
                  _, o4_dis = get_vec(self.opp_4[:2], agent.agent_observation[:2])
                  _, o5_dis = get_vec(self.opp_5[:2], agent.agent_observation[:2])
                  opp_distances = [o1_dis, o2_dis, o3_dis, o4_dis, o5_dis]
                  return bigger_than(opp_distances, 2)
            else:
                  _, a1_dis = get_vec(self.ai_1[:2], agent.agent_observation[:2])
                  _, a2_dis = get_vec(self.ai_2[:2], agent.agent_observation[:2])
                  _, a3_dis = get_vec(self.ai_3[:2], agent.agent_observation[:2])
                  _, a4_dis = get_vec(self.ai_4[:2], agent.agent_observation[:2])
                  _, a5_dis = get_vec(self.ai_5[:2], agent.agent_observation[:2])
                  ag_distances = [a1_dis, a2_dis, a3_dis, a4_dis, a5_dis]
                  return bigger_than(ag_distances, 2)


      # set agent's vector observation based on action_type, and ball owner
      # data structure to contain observations the agent would make in one step
          # the 5 values in the array represents: 
          # [0]: x coor, 
          # [1]: y coor, 
          # [2]: target x coor - object x coor
          # [3]: target y coor - object y coor
          # [4]: speed magnitude
      def _set_vector_observation(self, agent, action_type, set_target = False, target = [0, 0]): 

            action = Action(action_type)
            agent_observation = self.obs[agent.agent_index]
            ball_observation = self.obs[self.ball_index]
            target_padding = 3
            target_y = random.randint(self.goal_down + target_padding, self.goal_up - target_padding) 

            # if self.Debug: 
                  
            #       print(agent.name + " has ball: " + str(agent.has_ball))           
            
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

                        agent_observation[4] = 1.0 * random.randint(self.player_speed - 4, self.player_speed)

                        if set_target:

                              if self.Debug: 
                                    print(agent.name + " with ball: run to target")

                              agent_observation[2:4] = target

                        else:
                        
                              if self.Debug: 
                                    print(agent.name + " with ball: run to goal")

                              if agent.team == 'right': 
                                    agent_observation[2:4], _ = get_vec(np.array([0, target_y]), agent_observation[:2])
                              else: 
                                    agent_observation[2:4], _ = get_vec(np.array([self.length, target_y]), agent_observation[:2])
                        
                        # with 10% chance, the player drops the ball while running
                        if random.random() < 0.1:
                              self.ball_owner = BallOwner.NOONE
                        else:
                              self.obs[self.ball_index] = agent_observation
                  
                  # has ball and shoot toward goal, zeros agent's target x, y, mag
                      # agent_vec = x, y, 0, 0, 0
                      # ball_vec = x, y, tx, ty, m
                      # ball owener change
                  elif action == Action.shoot: 

                        accuracy_under_defence = NORMAL_MISS + self.defence_near(agent) * UNDER_DEFENCE_MISS 

                        ball_observation[4] = random.randint(self.shoot_speed - 16, self.shoot_speed) * 1.0

                        if self.Debug: 
                              print(agent.name + " with ball: shoot")

                        if agent.team == 'right': 
                              goal_to_ball, goal_to_ball_mag = get_vec(np.array([0, target_y]), ball_observation[:2])
                              ball_observation[2:4] = screw_vec(goal_to_ball, goal_to_ball_mag, accuracy_under_defence)

                        else: 
                              goal_to_ball, goal_to_ball_mag = get_vec(np.array([self.length, target_y]), ball_observation[:2])
                              ball_observation[2:4] = screw_vec(goal_to_ball, goal_to_ball_mag, accuracy_under_defence)

                        agent.has_ball = False
                        self.last_ball_owner = copy.copy(self.ball_owner)
                        self.ball_owner = BallOwner.NOONE
                        agent_observation[2:5] = np.array([0,0,0])

                  elif action == Action.assist:

                        if self.Debug: 
                              print(agent.name + " with ball: pass")

                        # figure out who the teammate is: pick the teammate closest to you #TODO: this is probably bad
                        if agent.name == 'opp_1': # 0
                            _, mate2_dis = get_vec(self.opp_2[:2], agent_observation[:2])
                            _, mate3_dis = get_vec(self.opp_3[:2], agent_observation[:2])
                            _, mate4_dis = get_vec(self.opp_4[:2], agent_observation[:2])
                            _, mate5_dis = get_vec(self.opp_5[:2], agent_observation[:2])
                            distance = [mate2_dis, mate3_dis, mate4_dis, mate5_dis]
                            mates = [self.opp_2, self.opp_3, self.opp_4, self.opp_5]
                            mate = mates[distance.index(min(distance))]
                            
                        elif agent.name == 'opp_2': # 1
                            _, mate1_dis = get_vec(self.opp_1[:2], agent_observation[:2])
                            _, mate3_dis = get_vec(self.opp_3[:2], agent_observation[:2])
                            _, mate4_dis = get_vec(self.opp_4[:2], agent_observation[:2])
                            _, mate5_dis = get_vec(self.opp_5[:2], agent_observation[:2])
                            distance = [mate1_dis, mate3_dis, mate4_dis, mate5_dis]
                            mates = [self.opp_1, self.opp_3, self.opp_4, self.opp_5]
                            mate = mates[distance.index(min(distance))]
                        
                        elif agent.name == 'opp_3': # 2
                            _, mate1_dis = get_vec(self.opp_1[:2], agent_observation[:2])
                            _, mate2_dis = get_vec(self.opp_2[:2], agent_observation[:2])
                            _, mate4_dis = get_vec(self.opp_4[:2], agent_observation[:2])
                            _, mate5_dis = get_vec(self.opp_5[:2], agent_observation[:2])
                            distance = [mate1_dis, mate2_dis, mate4_dis, mate5_dis]
                            mates = [self.opp_1, self.opp_2, self.opp_4, self.opp_5]
                            mate = mates[distance.index(min(distance))]
                        
                        elif agent.name == 'opp_4': # 3
                            _, mate1_dis = get_vec(self.opp_1[:2], agent_observation[:2])
                            _, mate2_dis = get_vec(self.opp_2[:2], agent_observation[:2])
                            _, mate3_dis = get_vec(self.opp_3[:2], agent_observation[:2])
                            _, mate5_dis = get_vec(self.opp_5[:2], agent_observation[:2])
                            distance = [mate1_dis, mate2_dis, mate3_dis, mate5_dis]
                            mates = [self.opp_1, self.opp_2, self.opp_3, self.opp_5]
                            mate = mates[distance.index(min(distance))]
                        
                        elif agent.name == 'opp_5': # 3
                            _, mate1_dis = get_vec(self.opp_1[:2], agent_observation[:2])
                            _, mate2_dis = get_vec(self.opp_2[:2], agent_observation[:2])
                            _, mate3_dis = get_vec(self.opp_3[:2], agent_observation[:2])
                            _, mate4_dis = get_vec(self.opp_4[:2], agent_observation[:2])
                            distance = [mate1_dis, mate2_dis, mate3_dis, mate4_dis]
                            mates = [self.opp_1, self.opp_2, self.opp_3, self.opp_4]
                            mate = mates[distance.index(min(distance))]
                        
                        ######### agents ###########
                        # passing to the closest teammate closer to the goal (right side) than I am (AI passing)
                        elif agent.name == 'ai_1':
                            _, mate2_dis = get_vec(self.ai_2[:2], agent_observation[:2])
                            _, mate3_dis = get_vec(self.ai_3[:2], agent_observation[:2])
                            _, mate4_dis = get_vec(self.ai_4[:2], agent_observation[:2])
                            _, mate5_dis = get_vec(self.ai_5[:2], agent_observation[:2])
                            distances = [mate2_dis, mate3_dis, mate4_dis, mate5_dis]
                            mates = [self.ai_2, self.ai_3, self.ai_4, self.ai_5]
                            
                            distances, mates = (list(t) for t in zip(*sorted(zip(distances, mates))))
                            mate = next((person for person in mates if person[0] > agent_observation[0]), mates[0])

                        elif agent.name == 'ai_2':
                            _, mate1_dis = get_vec(self.ai_1[:2], agent_observation[:2])
                            _, mate3_dis = get_vec(self.ai_3[:2], agent_observation[:2])
                            _, mate4_dis = get_vec(self.ai_4[:2], agent_observation[:2])
                            _, mate5_dis = get_vec(self.ai_5[:2], agent_observation[:2])
                            distances = [mate1_dis, mate3_dis, mate4_dis, mate5_dis]
                            mates = [self.ai_1, self.ai_3, self.ai_4, self.ai_5]
                            
                            distances, mates = (list(t) for t in zip(*sorted(zip(distances, mates))))
                            mate = next((person for person in mates if person[0] > agent_observation[0]), mates[0])
                            
                        elif agent.name == 'ai_3':
                            _, mate1_dis = get_vec(self.ai_1[:2], agent_observation[:2])
                            _, mate2_dis = get_vec(self.ai_2[:2], agent_observation[:2])
                            _, mate4_dis = get_vec(self.ai_4[:2], agent_observation[:2])
                            _, mate5_dis = get_vec(self.ai_5[:2], agent_observation[:2])
                            distances = [mate1_dis, mate2_dis, mate4_dis, mate5_dis]
                            mates = [self.ai_1, self.ai_2, self.ai_4, self.ai_5]
                            
                            distances, mates = (list(t) for t in zip(*sorted(zip(distances, mates))))
                            mate = next((person for person in mates if person[0] > agent_observation[0]), mates[0])
                            
                        elif agent.name == 'ai_4':
                            _, mate1_dis = get_vec(self.ai_1[:2], agent_observation[:2])
                            _, mate2_dis = get_vec(self.ai_2[:2], agent_observation[:2])
                            _, mate3_dis = get_vec(self.ai_3[:2], agent_observation[:2])
                            _, mate5_dis = get_vec(self.ai_5[:2], agent_observation[:2])
                            distances = [mate1_dis, mate2_dis, mate3_dis, mate5_dis]
                            mates = [self.ai_1, self.ai_2, self.ai_3, self.ai_5]
                            
                            distances, mates = (list(t) for t in zip(*sorted(zip(distances, mates))))
                            mate = next((person for person in mates if person[0] > agent_observation[0]), mates[0])
                            
                            
                        elif agent.name == 'ai_5':
                            _, mate1_dis = get_vec(self.ai_1[:2], agent_observation[:2])
                            _, mate2_dis = get_vec(self.ai_2[:2], agent_observation[:2])
                            _, mate3_dis = get_vec(self.ai_3[:2], agent_observation[:2])
                            _, mate4_dis = get_vec(self.ai_4[:2], agent_observation[:2])
                            distances = [mate1_dis, mate2_dis, mate3_dis, mate4_dis]
                            mates = [self.ai_1, self.ai_2, self.ai_3, self.ai_4]
                            
                            distances, mates = (list(t) for t in zip(*sorted(zip(distances, mates))))
                            mate = next((person for person in mates if person[0] > agent_observation[0]), mates[0])
                        
                        else:
                            print("paranormal activities")
            
                        # mate_vec_mag = math.sqrt(mate[2]**2 + mate[3]**2)

                        # # anticipate the teammate's location based on the current movement
                        # if mate_vec_mag == 0:
                        #       mate_next_pos_x = mate[0]
                        #       mate_next_pos_y = mate[1]
                        # else:
                        #       mate_next_pos_x = mate[0] + (mate[2]/mate_vec_mag)*mate[4] * STEP_SIZE * 2
                        #       mate_next_pos_y = mate[1] + (mate[3]/mate_vec_mag)*mate[4] * STEP_SIZE * 2

                        # mate_to_ball, _ = get_vec(np.array([mate_next_pos_x, mate_next_pos_y]), ball_observation[:2])

                        mate_to_ball, mate_to_ball_mag = get_vec(mate[:2], ball_observation[:2])
                        custom_pass_speed = mate_to_ball_mag / STEP_SIZE
                        if custom_pass_speed > SHOOT_SPEED:
                              custom_pass_speed = SHOOT_SPEED
                        ball_observation[4] = random.uniform(custom_pass_speed - 1, custom_pass_speed + 1)

                        ball_observation[2:4] = mate_to_ball

                        agent.has_ball = False
                        self.last_ball_owner = copy.copy(self.ball_owner)
                        self.ball_owner = BallOwner.NOONE
                        agent_observation[2:5] = np.array([0,0,0])

                  else: 

                        print('Unrecognized action %d' % action_type)

            # when the agent doesn't have ball
            else: 

                  ball_to_agent, ball_to_agent_magnitude = get_vec(ball_observation[:2], agent_observation[:2])
                  if agent.team == 'right':
                        goal_to_agent, _ = get_vec(np.array([0, self.width/2]), agent_observation[:2])
                  else:
                        goal_to_agent, _ = get_vec(np.array([self.length, self.width/2]), agent_observation[:2])

                  ### MADE CHANGE
                  # no ball and intercept
                      # if close, try get ball, stop agent, zeros agent's target x, y, mag
                          # intercept success
                              # agent_vec = ball_vec
                              # ball owener change
                          # intercept failed
                              # agent_vec no change
                              # ball_vec = x', y', tx', ty', m' (no change)
                              # ball owener not change
                      # if not close, stop agent, zeros agent's target x, y, mag
                          # agent_vec = x, y, 0, 0, 0
                          # ball_vec = x', y', tx', ty', m' (no change)
                          # ball owener not change
                  if action == Action.intercept: 

                        if self.Debug: 
                              print(agent.name + " no ball: intercept")

                        # agent_observation[2:5] = np.array([0,0,0])

                        intercept_success = random.random() < \
                              intercept_chance(ball_to_agent_magnitude, MIN_INTERCEPT_DIST, MAX_INTERCEPT_DIST)

                        if intercept_success or \
                              (self.ball_owner == BallOwner.NOONE and ball_to_agent_magnitude < MAX_INTERCEPT_DIST + 3): 

                              ball_observation[2:5] = agent_observation[2:5]
                              ball_observation[:2] = agent_observation[:2]
                              self.last_ball_owner = copy.copy(self.ball_owner)
                              self.ball_owner = BallOwner(agent.agent_index)
                              
                              if self.Debug:
                                    print(agent.name + " lucky, intercept success")

                        else: 

                              if self.Debug:
                                    print(agent.name + " unlucky, intercept failed")

                  # no ball, if teammate also doesn't have ball, run toward ball 
                  # otherwise, run towards the goal
                      # agent_vec = x, y, tx'', ty'', m''
                      # ball_vec = x', y', tx', ty', m' (no change)
                      # ball owener not change
                  elif action == Action.run: 

                        # agent_observation[4] = 1.0 * random.randint(self.player_speed, self.player_speed + 4)
                        agent_observation[4] = self.player_speed

                        if set_target:
    
                              if self.Debug: 
                                    print(agent.name + " no ball: run to target")
                                    
                              agent_observation[2:4] = target

                        else:

                              if self.Debug: 
                                    print(agent.name + " no ball: run to ball")

                              if self.ball_owner != BallOwner(agent.agent_index):
                                    agent_observation[2:4] = ball_to_agent
                              else:
                                    agent_observation[2:4] = goal_to_agent
              
                  # no ball and shoot, stop agent
                      # agent_vec = x, y, 0, 0, 0
                      # ball_vec = x', y', tx', ty', m' (no change)
                      # ball owener not change
                  elif action == Action.shoot:

                        if self.Debug: 
                              print(agent.name + " no ball: shoot and stop")

                        agent_observation[2:5] = np.array([0, 0, 0])

                  # no ball and assist, stop agent
                      # agent_vec = x, y, 0, 0, 0
                      # ball_vec = x', y', tx', ty', m' (no change)
                      # ball owener not change
                  elif action == Action.assist:

                        if self.Debug: 
                              print(agent.name + " no ball: assist and stop")

                        agent_observation[2:5] = np.array([0, 0, 0])
                  
                  else: 
                        print('Unrecognized action %d' % action_type)
                        
            return agent_observation


      ### changed
      # action_set=True means the action is given in the step function
      # action_set=False means the action need to be deduced from agent
      def _agent_set_vector_observation(self, agent, action_set = False, action_type = 0):
        
        mate_indices = agent.mate_indices
        
        if self.ball_owner == BallOwner(agent.agent_index):
              agent_has_ball = True
              team_has_ball = True
        
        elif self.ball_owner == BallOwner(agent.mate_indices[0]) or self.ball_owner == BallOwner(agent.mate_indices[1]) or self.ball_owner == BallOwner(agent.mate_indices[2]) or self.ball_owner == BallOwner(agent.mate_indices[3]):
              agent_has_ball = False
              team_has_ball = True
        else:
              agent_has_ball = False
              team_has_ball = False

        if not action_set:

              action_type = agent.get_action_type(self.obs, agent_has_ball, team_has_ball)

        else:
              agent._set_has_ball(agent_has_ball)

        self.obs[agent.agent_index] = self._set_vector_observation(agent, action_type)


      # move the [loc] according to [vec]
      # notice the STEP_SIZE
      def _step_by_observation(self, observation, is_ball=False):

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
            opp_1_in = self.ball[0] >= FIELD_LEN and (self.ball[1] > GOAL_LOWER and self.ball[1] < GOAL_UPPER)
            return ai_in or opp_1_in


      #### need further modification
      def fix(self, player):

            if player == BallOwner.OPP_1 or player == BallOwner.OPP_2:
                  new_owner = BallOwner.AI_1
            else:
                  new_owner = BallOwner.OPP_1

            # relocate the ball to where it went out
            self.ball[0] = lock_in(self.ball[0], FIELD_LEN)
            self.ball[1] = lock_in(self.ball[1], FIELD_WID)
            self.ball_owner = new_owner

            # move the other player and the ball together
            self.ball[2:5] = np.array([0,0,0])
            if new_owner == BallOwner.AI_1:
                  self.obs[self.ai_1_index] = copy.copy(self.ball)
            else:
                  self.obs[self.opp_1_index] = copy.copy(self.ball)


      def _step_vector_observations(self, observations):

            if self.Debug: 
                print("vector before step:")
                print(self.obs)

            for observation in observations: 
                  self._step_by_observation(observation)

            if self.Debug: 
                print("vector after step:")
                print(self.obs)
            

      def out_of_field(self):
            x_out = self.ball[0] < 0 or self.ball[0] > self.length
            y_out = self.ball[1] < 0 or self.ball[1] > self.width
            y_score = self.ball[1] > self.goal_down - 2 and self.ball[1] < self.goal_up + 2
            return (x_out and not y_score) or y_out

                  
      def step(self, ai_action_type):
        print(ai_action_type)
      
        o_b = copy.copy(self.ball)
        o_ai_1 = copy.copy(self.ai_1)
        o_ai_2 = copy.copy(self.ai_2)
        o_ai_3 = copy.copy(self.ai_3)
        o_ai_4 = copy.copy(self.ai_4)
        o_ai_5 = copy.copy(self.ai_5)
        
        o_p_1 = copy.copy(self.opp_1)
        o_p_2 = copy.copy(self.opp_2)
        o_p_3 = copy.copy(self.opp_3)
        o_p_4 = copy.copy(self.opp_4)
        o_p_5 = copy.copy(self.opp_5)
        o_b_o_a = copy.copy(self.ball_owner_array)
        
        individual_size = 4

        if self.random_opp:
        
            opp_action_type = random.randint(0, 15)
            opp_action_type = tuple(base_convert(opp_action_type, 4))

            self._agent_set_vector_observation(self.opp_1_agent, action_set = True, action_type = opp_action_type[0])
            self._agent_set_vector_observation(self.opp_2_agent, action_set = True, action_type = opp_action_type[1])
            self._agent_set_vector_observation(self.opp_3_agent, action_set = True, action_type = opp_action_type[2])
            self._agent_set_vector_observation(self.opp_4_agent, action_set = True, action_type = opp_action_type[3])
            self._agent_set_vector_observation(self.opp_5_agent, action_set = True, action_type = opp_action_type[4])

        else:
            self._opp_team_set_vector_observation()

        if self.action_as_int:
            ai_action_type = tuple(base_convert(ai_action_type, 4))

        self._agent_set_vector_observation(self.ai_1_agent, action_set = True, action_type = ai_action_type[0])
        self._agent_set_vector_observation(self.ai_2_agent, action_set = True, action_type = ai_action_type[1])
        self._agent_set_vector_observation(self.ai_3_agent, action_set = True, action_type = ai_action_type[2])
        self._agent_set_vector_observation(self.ai_4_agent, action_set = True, action_type = ai_action_type[3])
        self._agent_set_vector_observation(self.ai_5_agent, action_set = True, action_type = ai_action_type[4])

        ### changed
        # use the old step vector function for the players who
        # doesn't decelerate
        self._step_vector_observations(self.obs[0:4])
        # use deceleration step function for ball
        self._step_by_observation(self.obs[4], is_ball=True)


        # calculate reward
        reward = self._get_reward(o_b, o_ai_1, o_ai_2, o_ai_3, o_ai_4, o_ai_5, o_p_1, o_p_2, o_p_3, o_p_4, o_p_5,  o_b_o_a)

        done = False

        if self.score():
              if self.ball[0] <= 0:
                    self.opp_score += 1
              else:
                    self.ai_score += 1

              if self.Debug:
                    print("Score!!!")
                    print("ai : opp = " + str(self.ai_score) + " : " + str(self.opp_score))

              if self.one_goal_end:
                    done = True

              ### changed from simple reset() TODO: why is everything set twice_
              self.ball = np.array([FIELD_LEN/2, FIELD_WID/2, 0, 0, 0, 0])
              
              self.ai_1 = np.array([FIELD_LEN/2 - 9, FIELD_WID/2 + 5, 0, 0, 0, 0])
              self.ai_2 = np.array([FIELD_LEN/6, FIELD_WID/2 - FIELD_WID/6, 0, 0, 0, 0])
              self.ai_3 = np.array([FIELD_LEN/6, FIELD_WID/2 + FIELD_WID/6, 0, 0, 0, 0])
              self.ai_4 = np.array([FIELD_LEN/3, FIELD_WID/2 - FIELD_WID/6, 0, 0, 0, 0])
              self.ai_5 = np.array([FIELD_LEN/3, FIELD_WID/2 + FIELD_WID/6, 0, 0, 0, 0])
              
              self.opp_1 = np.array([FIELD_LEN/2 + 9, FIELD_WID/2 + 5, 0, 0, 0, 0])
              self.opp_2 = np.array([5 * FIELD_LEN/6, FIELD_WID/2 - FIELD_WID/6, 0, 0, 0, 0])
              self.opp_3 = np.array([5 * FIELD_LEN/6, FIELD_WID/2 + FIELD_WID/6, 0, 0, 0, 0])
              self.opp_4 = np.array([4 * FIELD_LEN/6, FIELD_WID/2 - FIELD_WID/6, 0, 0, 0, 0])
              self.opp_5 = np.array([4 * FIELD_LEN/6, FIELD_WID/2 + FIELD_WID/6, 0, 0, 0, 0])
              
              self.ball_owner_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
              
              self.obs = np.concatenate((self.ai_1, self.ai_2, self.ai_3, self.ai_4, self.ai_5, self.opp_1, self.opp_2, self.opp_3, self.opp_4, self.opp_5, self.ball)).reshape((11, 6))
              
              self.ball_owner = BallOwner.NOONE
              self.last_ball_owner = BallOwner.NOONE

              self.ai_1 = self.obs[self.ai_1_index]
              self.ai_2 = self.obs[self.ai_2_index]
              self.ai_3 = self.obs[self.ai_3_index]
              self.ai_4 = self.obs[self.ai_4_index]
              self.ai_5 = self.obs[self.ai_5_index]
              
              self.opp_1 = self.obs[self.opp_1_index]
              self.opp_2 = self.obs[self.opp_2_index]
              self.opp_3 = self.obs[self.opp_3_index]
              self.opp_4 = self.obs[self.opp_4_index]
              self.opp_5 = self.obs[self.opp_5_index]
              
              self.ball = self.obs[self.ball_index]
#                  self.ball_owner_array = self.obs[self.ball_owner_array_index]

        if self.out_of_field():
              self.fix(self.last_ball_owner)
              if self.Debug:
                    print("fix out of box ball")

              if self.one_goal_end:
                    done = True

        self.ball_owner_array_update()

        # figure out whether the game is over
        if self.time >= self.game_time:
              done = True
        
        # one second passes in the game
        self.time += STEP_SIZE
        return self.obs, reward, done, {}

      
      def ball_owner_array_update(self):
            if self.ball_owner == BallOwner.AI_1:
                owner_idx = self.ai_1_index
            elif self.ball_owner == BallOwner.AI_2:
                owner_idx = self.ai_2_index
            elif self.ball_owner == BallOwner.AI_3:
                owner_idx = self.ai_3_index
            elif self.ball_owner == BallOwner.AI_4:
                owner_idx = self.ai_4_index
            elif self.ball_owner == BallOwner.AI_5:
                owner_idx = self.ai_5_index
                
            elif self.ball_owner == BallOwner.OPP_1:
                  owner_idx = self.opp_1_index
            elif self.ball_owner == BallOwner.OPP_2:
                  owner_idx = self.opp_2_index
            elif self.ball_owner == BallOwner.OPP_3:
                  owner_idx = self.opp_3_index
            elif self.ball_owner == BallOwner.OPP_4:
                  owner_idx = self.opp_4_index
            elif self.ball_owner == BallOwner.OPP_5:
                owner_idx = self.opp_5_index
            else:
                  owner_idx = self.ball_index

#            for idx in range(10):
#                  if idx == owner_idx:
#                        self.obs[self.ball_owner_array_index][idx] = 10
#                  else:
#                        self.obs[self.ball_owner_array_index][idx] = 0

    
      def _get_reward(self, ball, ai_1, ai_2, ai_3, ai_4, ai_5, opp_1, opp_2, opp_3, opp_4, opp_5, ball_owner):

            ball_adv = self.ball[0] - ball[0]
            ball_adv_r = ball_adv * BALL_ADV_REWARD_BASE

            # player_adv = self.ai_1[0] - ai_1[0] + self.ai_2[0] - ai_2[0]

            defence = self.defence_near(self.opp_1_agent) + self.defence_near(self.opp_2_agent) + \
                      self.defence_near(self.opp_3_agent) + self.defence_near(self.opp_4_agent) + \
                      self.defence_near(self.opp_5_agent)
            defence_r = defence * DEFENCE_REWARD_BASE

            if self.out(self.ai_1) or self.out(self.ai_2) or self.out(self.ai_3) or self.out(self.ai_4) or self.out(self.ai_5):
                  out_of_field = OUT_OF_FIELD_PENALTY
            else:
                  out_of_field = 0

            if (self.ball_owner == BallOwner.AI_1 or self.ball_owner == BallOwner.AI_2 or self.ball_owner == BallOwner.AI_3 or self.ball_owner == BallOwner.AI_4 or self.ball_owner == BallOwner.AI_5) and (ball_owner[self.ai_1_index] == 0 and ball_owner[self.ai_2_index] == 0 and ball_owner[self.ai_3_index] == 0 and ball_owner[self.ai_4_index] == 0 and ball_owner[self.ai_5_index] == 0):
                  get_ball = 10 * BALL_CONTROL
            elif self.ball_owner == BallOwner.AI_1 or self.ball_owner == BallOwner.AI_2 or self.ball_owner == BallOwner.AI_3 or self.ball_owner == BallOwner.AI_4 or self.ball_owner == BallOwner.AI_5:
                  get_ball = BALL_CONTROL
            else:
                  get_ball = -BALL_CONTROL

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

            return get_ball + score + get_scored + out_of_field + ball_adv_r + defence_r


      def _opp_team_set_vector_observation(self):

            if self.ball_owner == BallOwner(self.opp_1_agent.agent_index):
                opp_1_has_ball = True
                opp_2_has_ball = False
                opp_3_has_ball = False
                opp_4_has_ball = False
                opp_5_has_ball = False
                team_has_ball = True
            elif self.ball_owner == BallOwner(self.opp_2_agent.agent_index):
                opp_1_has_ball = False
                opp_2_has_ball = True
                opp_3_has_ball = False
                opp_4_has_ball = False
                opp_5_has_ball = False
                team_has_ball = True
            elif self.ball_owner == BallOwner(self.opp_3_agent.agent_index):
                opp_1_has_ball = False
                opp_2_has_ball = False
                opp_3_has_ball = True
                opp_4_has_ball = False
                opp_5_has_ball = False
                team_has_ball = True
            elif self.ball_owner == BallOwner(self.opp_4_agent.agent_index):
                opp_1_has_ball = False
                opp_2_has_ball = False
                opp_3_has_ball = False
                opp_4_has_ball = True
                opp_5_has_ball = False
                team_has_ball = True
            elif self.ball_owner == BallOwner(self.opp_5_agent.agent_index):
                opp_1_has_ball = False
                opp_2_has_ball = False
                opp_3_has_ball = False
                opp_4_has_ball = False
                opp_5_has_ball = True
                team_has_ball = True
            else:
                opp_1_has_ball = False
                opp_2_has_ball = False
                opp_3_has_ball = False
                opp_4_has_ball = False
                opp_5_has_ball = False
                team_has_ball = False

            opp1_action_type = self.opp_1_agent.get_action_type(self.obs, opp_1_has_ball, team_has_ball)
            opp2_action_type = self.opp_2_agent.get_action_type(self.obs, opp_2_has_ball, team_has_ball)
            opp3_action_type = self.opp_3_agent.get_action_type(self.obs, opp_3_has_ball, team_has_ball)
            opp4_action_type = self.opp_4_agent.get_action_type(self.obs, opp_4_has_ball, team_has_ball)
            opp5_action_type = self.opp_4_agent.get_action_type(self.obs, opp_5_has_ball, team_has_ball)

            opp1_action = Action(opp1_action_type)
            opp2_action = Action(opp2_action_type)
            opp3_action = Action(opp3_action_type)
            opp4_action = Action(opp4_action_type)
            opp5_action = Action(opp5_action_type)

            opp1_set_target = False
            opp2_set_target = False
            opp3_set_target = False
            opp4_set_target = False
            opp5_set_target = False

            opp1_target = [0, 0]
            opp2_target = [0, 0]
            opp3_target = [0, 0]
            opp4_target = [0, 0]
            opp5_target = [0, 0]

            # randomize the action of opp2 opp1

            if opp_1_has_ball:

                  if opp1_action == Action.run:

                        # run_to_goal_p = 0.5
                        # if random.random() > run_to_goal_p:
                        if self.opp_1[1] > self.width * 0.2:

                                opp1_set_target = True
                                opp1_target = [-1, -1]

                        if opp2_action == Action.run and self.opp_2[0] > self.length * 0.1 :
                            if self.opp_2[1] < self.width * 0.8:
                                opp2_set_target = True
                                opp2_target = [-1, 1]
                                        
                        if opp3_action == Action.run and self.opp_3[0] > self.length * 0.1 :
                            if self.opp_3[1] < self.width * 0.8:
                                opp3_set_target = True
                                opp3_target = [-1, 1]
                        
                        if opp4_action == Action.run and self.opp_4[0] > self.length * 0.1 :
                            if self.opp_4[1] < self.width * 0.8:
                                opp4_set_target = True
                                opp4_target = [-1, 1]
                                
                        if opp5_action == Action.run and self.opp_5[0] > self.length * 0.1 :
                            if self.opp_5[1] < self.width * 0.8:
                                opp5_set_target = True
                                opp5_target = [-1, 1]


            if opp_2_has_ball:
                if opp2_action == Action.run:
                # run_to_goal_p = 0.5
                # if random.random() > run_to_goal_p:

                    if self.opp_2[1] < self.width * 0.8:
                            
                            opp2_set_target = True
                            opp2_target = [-1, 1]

                    if opp1_action == Action.run and self.opp_1[0] > self.length * 0.1 :

                            if self.opp_1[1] > self.width * 0.2:

                                    opp1_set_target = True
                                    opp1_target = [-1, -1]
                                    
                    if opp3_action == Action.run and self.opp_3[0] > self.length * 0.1 :
                        if self.opp_3[1] < self.width * 0.8:
                            opp3_set_target = True
                            opp3_target = [-1, 1]
                    
                    if opp4_action == Action.run and self.opp_4[0] > self.length * 0.1 :
                        if self.opp_4[1] < self.width * 0.8:
                            opp4_set_target = True
                            opp4_target = [-1, 1]
                            
                    if opp5_action == Action.run and self.opp_5[0] > self.length * 0.1 :
                        if self.opp_5[1] < self.width * 0.8:
                            opp5_set_target = True
                            opp5_target = [-1, 1]


            if self.ball_owner == BallOwner.AI_1 or self.ball_owner == BallOwner.AI_2 or self.ball_owner == BallOwner.AI_3 or self.ball_owner == BallOwner.AI_4 or self.ball_owner == BallOwner.AI_5:

                    if self.ball[0] < self.length * 0.6:
                            # person furthest to the goal runs to defend the goal?? TODO: is that right?
                            defence_point = [self.length * 0.75, self.width * 0.5]
                            
                            dict = {self.opp_1[0]: 0, self.opp_2[0]: 1, self.opp_3[0]: 2, self.opp_4[0]: 3, self.opp_5[0]: 4}
                            
                            x_coors = [self.opp_1[0], self.opp_2[0], self.opp_3[0], self.opp_4[0], self.opp_5[0]]
                            
                            opponents = [self.opp_1, self.opp_2, self.opp_3, self.opp_4, self.opp_5]
                            action_types = [opp1_action_type, opp2_action_type, opp3_action_type, opp4_action_type, opp5_action_type]
                            set_targets = [opp1_set_target, opp2_set_target, opp3_set_target, opp4_set_target, opp5_set_target]
                            targets = [opp1_target, opp2_target, opp3_target, opp4_target, opp5_target]

                            x_coors.sort(reverse = True)
                            number = dict[x_coors[0]]
                            
                            
                            action_types[number] = 0 # run
                            set_targets[number] = True
                            targets[number], _ = get_vec(defence_point, opponents[number])


            self.obs[self.opp_1_agent.agent_index] = self._set_vector_observation(self.opp_1_agent, 
                                                                                  opp1_action_type,
                                                                                  set_target = opp1_set_target, 
                                                                                  target = opp1_target)
            
            self.obs[self.opp_2_agent.agent_index] = self._set_vector_observation(self.opp_2_agent, 
                                                                                  opp2_action_type,
                                                                                  set_target = opp2_set_target, 
                                                                                  target = opp2_target)
                                                                                  
            self.obs[self.opp_3_agent.agent_index] = self._set_vector_observation(self.opp_3_agent,
                                                                                  opp3_action_type,
                                                                                  set_target = opp3_set_target,
                                                                                  target = opp3_target)
                                                                                  
            self.obs[self.opp_4_agent.agent_index] = self._set_vector_observation(self.opp_4_agent,
                                                                                  opp4_action_type,
                                                                                  set_target = opp4_set_target,
                                                                                  target = opp4_target)
                                                                            
            self.obs[self.opp_5_agent.agent_index] = self._set_vector_observation(self.opp_5_agent,
                                                                                  opp5_action_type,
                                                                                  set_target = opp5_set_target,
                                                                                  target = opp5_target)

            

            if self.ball_owner == BallOwner.NOONE and opp1_action == Action.run and opp2_action == Action.run \
                                                  and opp3_action == Action.run and opp4_action == Action.run \
                                                  and opp5_action == Action.run :

                    # anticipate the ball movement
                    ball_next_obs = copy.copy(self.ball)

                    self._step_by_observation(ball_next_obs)

                    opp1_to_ball_next, opp1_to_ball_next_mag = get_vec(ball_next_obs, self.opp_1)
                    opp2_to_ball_next, opp2_to_ball_next_mag = get_vec(ball_next_obs, self.opp_2)
                    opp3_to_ball_next, opp3_to_ball_next_mag = get_vec(ball_next_obs, self.opp_3)
                    opp4_to_ball_next, opp4_to_ball_next_mag = get_vec(ball_next_obs, self.opp_4)
                    opp5_to_ball_next, opp5_to_ball_next_mag = get_vec(ball_next_obs, self.opp_5)

                    if opp1_to_ball_next_mag < STEP_SIZE * self.player_speed:
                        target_m = opp1_to_ball_next_mag / STEP_SIZE
                        self.opp_1[2:4] = opp1_to_ball_next
                        self.opp_1[4] = target_m

                    elif opp2_to_ball_next_mag < STEP_SIZE * self.player_speed:
                        target_m = opp2_to_ball_next_mag / STEP_SIZE
                        self.opp_2[2:4] = opp2_to_ball_next
                        self.opp_2[4] = target_m
                            
                    elif opp3_to_ball_next_mag < STEP_SIZE * self.player_speed:

                        target_m = opp3_to_ball_next_mag / STEP_SIZE
                        self.opp_3[2:4] = opp3_to_ball_next
                        self.opp_3[4] = target_m
                        
                    elif opp4_to_ball_next_mag < STEP_SIZE * self.player_speed:

                        target_m = opp4_to_ball_next_mag / STEP_SIZE
                        self.opp_4[2:4] = opp4_to_ball_next
                        self.opp_4[4] = target_m
                        
                    elif opp5_to_ball_next_mag < STEP_SIZE * self.player_speed:

                        target_m = opp5_to_ball_next_mag / STEP_SIZE
                        self.opp_5[2:4] = opp5_to_ball_next
                        self.opp_5[4] = target_m
