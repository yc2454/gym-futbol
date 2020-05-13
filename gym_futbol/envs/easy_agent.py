import math
from .ballowner import BallOwner
from .action import Action
import random

num_mates = 4

# get the vector pointing from [coor2] to [coor1] and 
# its magnitude
def get_vec(coor_t, coor_o):
      vec = coor_t[:2] - coor_o[:2]
      vec_mag = math.sqrt(vec[0]**2 + vec[1]**2)
      return vec, vec_mag

class Easy_Agent():

      def __init__(self, name, observations, agent_index, mates_indices, ball_index, team, has_ball, length, width, goal_size, shoot_range = 10):

            self.name = name

            self.observations = observations
            self.agent_index = agent_index
            self.mate_indices = mate_indices
            self.ball_index = ball_index

            self.team = team 

            if team == 'right':
                  self.target_x = 0
                  self.shoot_x = self.target_x + shoot_range
            else:
                  self.target_x = length
                  self.shoot_x = self.target_x - shoot_range

            self.agent_observation = observations[agent_index]
            
            self.mate_observations = random.sample(range(10), num_mates) # initializing it randomly so I can put things in it
            
            for i in range(num_mates):
                self.mate_observations[i] = observations[self.mate_indices[i]]
                
            self.ball = observations[ball_index]
            self.has_ball = has_ball

            self.length = length
            self.width = width
            self.goal_up = width / 2 + goal_size / 2
            self.goal_down = width / 2 - goal_size / 2

            self.shoot_range = shoot_range
      
      ### added for DQN agent to step
      def _set_has_ball(self, has_ball):
            self.has_ball = has_ball

      def get_action_type(self, observations, has_ball, team_has_ball):

            self.agent_observation = observations[self.agent_index]
            
            for i in range (num_mates):
                self.mate_observations[i] = observations[self.mate_indices[i]]
            
            self.ball = observations[self.ball_index]
            self.has_ball = has_ball

            # data structure to contain observations the agent would make in one step
            # the 5 values in the array represents: 
            # [0]: x coor, 
            # [1]: y coor, 
            # [2]: target x coor - object x coor
            # [3]: target y coor - object y coor
            # [4]: speed magnitude

            _, ball_to_agent_magnitude = get_vec(self.ball[:2], self.agent_observation[:2])
            
            for i in range (num_mates):
                _, mate_to_agent_magnitude[i] = get_vec(self.mate_observations[i][:2], self.agent_observation[:2])

            action_type = 0

            if has_ball:
                if (self.team == 'right' and self.agent_observation[0] <= self.shoot_x) or (self.team != 'right' and self.agent_observation[0] >= self.shoot_x):
                        # shoot
                        action_type = 2
                        
                elif (self.mate_observations[0][0] < self.agent_observation[0] - 10 or \
                    self.mate_observations[1][0] < self.agent_observation[0] - 10 or \
                    self.mate_observations[2][0] < self.agent_observation[0] - 10 or \
                    self.mate_observations[3][0] < self.agent_observation[0] - 10) \
                    and random.random() > 0.8 and (mate_to_agent_magnitude[0] > 12 or \
                                                   mate_to_agent_magnitude[1] > 12 or \
                                                   mate_to_agent_magnitude[2] > 12 or \
                                                   mate_to_agent_magnitude[3] > 12):
                                                   
                        # assist
                        action_type = 3

                else:
                    # run
                    action_type = 0
                    
            else:
                  if ball_to_agent_magnitude <= 0.5 and random.random() < 0.1: 
                        # intercept
                        action_type = 1
                  else: 
                        # run
                        action_type = 0

            return action_type
