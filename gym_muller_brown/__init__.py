from gym.envs.registration import register

register(
    id='MullerBrownDiscrete-v0',
    entry_point='gym_muller_brown.envs:MullerBrownDiscreteEnv',
    kwargs={'m':5,'n':5},
)

register(
    id='MullerBrownContinuous-v0',
    entry_point='gym_muller_brown.envs:MullerBrownContinuousEnv',
)
