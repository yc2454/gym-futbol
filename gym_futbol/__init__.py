from gym.envs.registration import register

register(
    id='Futbol-v0',
    entry_point='gym_futbol.envs:FutbolEnv',
)
register(
    id='Futbol-extrahard-v0',
    entry_point='gym_futbol.envs:FutbolExtraHardEnv',
)

register(
    id='Futbol-v1',
    entry_point='gym_futbol.envs_v1:Futbol',
)
