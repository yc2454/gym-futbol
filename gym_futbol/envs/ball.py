from vector import Vector

class Ball:
    def __init__(self, coordinate, vector, owner):
        self.coordinate = coordinate
        self.vector = vector
        self.owner = owner

    def set_ball_vec(self, x, y, mag):
        self.vector = Vector(x, y, mag)

    def set_ball_owner(self, owner):
        self.owner = owner

    def set_ball_coord(self, x, y):
        self.coordinate = (x, y)

    def get_coordinate(self):
        return self.coordinate
    
    def set_coordinate(self, x, y):
        self.coordinate = (x, y)