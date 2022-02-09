import gym.envs.registration as gym_registration

# trunk-ignore-all(flake8/F401)
from .envs import make_vec_env

gym_registration.register(
    id="MiniWorld-LookAtObjs-v0",
    entry_point="envs.lookatobjs:LookAtObjs",
)
