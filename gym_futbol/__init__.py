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
    kwargs={'number_of_player': 10},
)

register(
    id='Futbol2v2-v1',
    entry_point='gym_futbol.envs_v1:Futbol',
    kwargs={'number_of_player': 2},
)

register(
    id='Futbol5v5-v1',
    entry_point='gym_futbol.envs_v1:Futbol',
    kwargs={'number_of_player': 5},
)
