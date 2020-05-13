import enum

class BallOwner(enum.Enum):
    AI_1 = 0
    AI_2 = 1
    AI_3 = 2
    AI_4 = 3
    AI_5 = 4
    
    OPP_1 = 5
    OPP_2 = 6
    OPP_3 = 7
    OPP_4 = 8
    OPP_5 = 9
    NOONE = 10 # has to equal ball_index in futbol_env
