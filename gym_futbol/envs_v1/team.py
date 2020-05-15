import pymunk
from pymunk.vec2d import Vec2d
from .player import Player
import numpy as np


class Team():

    def __init__(self, space, width, height, player_weight,
                 player_max_velocity, color=(1, 0, 0, 1), side="left",
                 player_number=2, elasiticity=0.2):
        self.space = space
        self.width = width
        self.height = height
        self.side = side

        self.player_number = player_number
        # implement for 3 players and fewer now
        if player_number <= 3:
            # get x position for each player
            if side == "left":
                self.x_pos_array = [width * 0.25] * player_number
            elif side == "right":
                self.x_pos_array = [width * 0.75] * player_number
            else:
                print("invalid side")
            # get y position for each player
            y_increment = height / (player_number + 1)
            self.y_pos_array = []
            for i in range(player_number):
                self.y_pos_array.append(y_increment * (i+1))

        else:
            print("unimplemented")

        self.player_array = []
        for x, y in zip(self.x_pos_array, self.y_pos_array):
            self.player_array.append(
                Player(self.space, x, y,
                       mass=player_weight,
                       color=color,
                       max_velocity=player_max_velocity,
                       elasticity=elasiticity,
                       side=side))

    def set_position_to_initial(self):
        for player, x, y in zip(self.player_array, self.x_pos_array, self.y_pos_array):
            player.set_position(x, y)
            # zero velocity
            player.body.velocity = 0, 0

    # return reshpaed numpy array observation
    def get_observation(self):
        obs_array = []
        for player in self.player_array:
            obs_array.append(player.get_observation())
        obs_array = np.reshape(np.array(obs_array), -1)
        return obs_array
