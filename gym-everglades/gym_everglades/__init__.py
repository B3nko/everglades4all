from gym.envs.registration import register

register(
        id='everglades-v0',
        entry_point='gym_everglades.envs:EvergladesEnv'
)
