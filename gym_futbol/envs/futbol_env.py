import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils

from ballowner import BallOwner
from action import Action

import numpy as np
import math
import time
import random

# get the vector pointing from [coor2] to [coor1] and 
# its magnitude
def get_vec(coor1, coor2):
      vec = coor1[:2] - coor2[:2]
      vec_mag = math.sqrt(vec[0]**2 + vec[1]**2)
      return vec, vec_mag

class Easy_Agent():

      def __init__(self, name, observations, agent_index, ball_index, team, has_ball, length, width, goal_size, shoot_range = 100):

            self.name = name

            self.observations = observations
            self.agent_index = agent_index
            self.ball_index = ball_index

            self.team = team 

            if team == 'right':
                  self.target_x = 0
                  self.shoot_x = self.target_x + shoot_range
            else:
                  self.target_x = length
                  self.shoot_x = self.target_x - shoot_range

            self.agent_observation = observations[agent_index]
            self.ball = observations[ball_index]
            self.has_ball = has_ball

            self.length = length
            self.width = width
            self.goal_up = width / 2 + goal_size / 2
            self.goal_down = width / 2 - goal_size / 2

            self.shoot_range = shoot_range

      def get_action_type(self, observations, has_ball):

            self.agent_observation = observations[self.agent_index]
            self.ball = observations[self.ball_index]
            self.has_ball = has_ball

            # data structure to contain observations the agent would make in one step
            # the 5 values in the array represents: 
            # [0]: x coor, 
            # [1]: y coor, 
            # [2]: target x coor - object x coor
            # [3]: target y coor - object y coor
            # [4]: speed magnitude

            ball_to_agent, ball_to_agent_magnitude = get_vec(self.ball[:2], self.agent_observation[:2])

            action_type = 0

            if has_ball:

                  if (self.team == 'right' and self.agent_observation[0] <= self.shoot_x) or (self.team != 'right' and self.agent_observation[0] >= self.shoot_x): 
                        # shoot
                        action_type = 2
                        # print('agent shoot')
                  else:
                        # run
                        action_type = 0
            else:
                  if ball_to_agent_magnitude <= 5: 
                        # intercept
                        action_type = 1
                  else: 
                        # run
                        action_type = 0
            return action_type


class FutbolEnv(gym.Env):

      def __init__(self, length = 600, width = 300, goal_size = 100, game_time = 200, player_speed = 9, ball_speed = 20, Debug = False):

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
            
            self.obs = self.reset()

      
      # Reset the state of the environment to an initial state
      def reset(self):
            # current time in the match, in seconds
            self.time = 0

            # below are the coordinates and vectors of ball, agent and opponent, 
            # refer to the observation_space comment
            #  
            # position and movement of the ball
            self.ball = np.array([self.length/2 + 100, self.width/2, 0, 0, 0])
            # position and movement of AI player
            self.ai = np.array([self.length/2 - 100, self.width/2, 0, 0, 0])
            # position and movement of opponent player
            self.opp = np.array([self.length/2 + 100, self.width/2, 0, 0, 0])

            self.obs = np.concatenate((self.ai, self.opp, self.ball)).reshape((3, 5))

            self.ai_index = 0
            self.opp_index = 1
            self.ball_index = 2

            self.ai = self.obs[self.ai_index]
            self.opp = self.obs[self.opp_index]
            self.ball = self.obs[self.ball_index]

            # who has the ball
            self.ball_owner = BallOwner.OPP

            # opp easy agent
            self.opp_agent = Easy_Agent('opp', self.obs, self.opp_index, self.ball_index, 'right', (self.ball_owner == BallOwner.OPP), self.length, self.width, self.goal_size, shoot_range = 100)

            # ai easy agent
            self.ai_agent = Easy_Agent('ai', self.obs, self.ai_index, self.ball_index, 'left', (self.ball_owner == BallOwner.AI), self.length, self.width, self.goal_size, shoot_range = 100)

            return self.obs

      def _next_observation(self):
            return self.obs


      # Render the environment to the screen
      def render(self, mode='human', close=False):

            fig, ax = plt.subplots()
            ax.set_xlim(0, self.length)
            ax.set_ylim(0, self.width)

            # ai
            ai_x, ai_y, _, _, _ = self.ai
            ax.plot(ai_x,ai_y, color = 'red', marker='o', markersize=12, label='ai')

            # opp
            opp_x, opp_y, _, _, _ = self.opp
            ax.plot(opp_x, opp_y, color = 'blue', marker='o', markersize=12, label='opp')

            # ball
            ball_x, ball_y, _, _, _ = self.ball
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

            agent_observation = agent.agent_observation

            ball_observation = self.obs[self.ball_index]
           
            target_padding = 3

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

                        if agent.team == 'right': 
                              ball_observation[2:4], _ = get_vec(np.array([0, target_y]), ball_observation[:2])

                        else: 
                              ball_observation[2:4], _ = get_vec(np.array([self.length, target_y]), ball_observation[:2])

                        agent.has_ball = False
                        self.ball_owner = BallOwner.NOONE
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

                        intercept_distance = 5

                        if ball_to_agent_magnitude > 5:

                              if self.Debug:
                                    print(agent.name + " too far, intercept failed")

                        else: 

                              intercept_success = random.random() <= 0.3

                              if intercept_success: 

                                    ball_observation[2:5] = np.array([0, 0, 0])
                                    ball_observation[:2] = agent_observation[:2]
                                    self.ball_owner = BallOwner(agent.agent_index)

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

      def _agent_set_vector_observation(self, agent):

            if self.ball_owner == BallOwner(agent.agent_index): 
                  agent_has_ball = True
            else:
                  agent_has_ball = False

            agent_action_type = agent.get_action_type(self.obs, agent_has_ball)

            self.obs[agent.agent_index] = self._set_vector_observation(agent, agent_action_type)

      # move the [loc] according to [vec]
      def _step_observation(self, observation):

            tx, ty = observation[2:4]
            vec_mag = math.sqrt(tx**2 + ty**2)

            if vec_mag == 0:
                  pass
            else:
                  observation[0] += observation[4] * (tx * 1.0 / vec_mag)
                  observation[1] += observation[4] * (ty * 1.0 / vec_mag)


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
    
      # Execute one time step within the environment
      def step(self, action_type):

            reward = 0

            self._agent_set_vector_observation(self.opp_agent)
            self._agent_set_vector_observation(self.ai_agent)

            self._step_vector_observations(self.obs)

            # figure out whether the game is over
            if (self.time == self.game_time) or (self.out_of_field()):
                  done = True
            else:
                  done = False
            
            # one second passes in the game
            self.time += 1
            return self.obs, reward, done, {}