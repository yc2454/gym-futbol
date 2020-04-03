from gym.envs.registration import register

register(
    id='futbol-v0',
    entry_point='gym_futbol.envs:FutbolEnv',
)
register(
    id='futbol-extrahard-v0',
    entry_point='gym_futbol.envs:FutbolExtraHardEnv',
)

def my_cool_test_method():
    print('It works!')