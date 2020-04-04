import enum
from vector import Vector

class Player:

    def __init__(self, name, coordinate, vector, team, balled):
        self.name = name
        self.coordinate = coordinate
        self.vector = vector
        self.team = team
        self.balled = balled

    def get_coordinate(self):
        return self.coordinate

    def get_vector(self):
        return self.coordinate

    # def get_action(self):
    #     return self.action

    def get_balled(self):
        return self.balled

    def set_coordinate(self, x, y):
        self.coordinate = (x, y)

    def set_vector(self, x, y, mag):
        self.vector = Vector(x, y, mag)

    # def set_action(self, act):
    #     self.action = act

    def set_balled(self):
        self.balled = not self.balled