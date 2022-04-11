import math

import gym.envs.registration as gym_registration

gym_registration.register(
    id="MiniWorld-LookAtObjs-v0",
    entry_point="envs.lookatobjs:LookAtObjs",
    kwargs=dict(max_angle=2 * math.pi, num_objs=5),
)

gym_registration.register(
    id="MiniWorld-LookAtObjs-v1",
    entry_point="envs.lookatobjs:LookAtObjs",
    kwargs=dict(max_angle=math.pi, num_objs=5),
)
