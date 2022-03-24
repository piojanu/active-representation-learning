import gym.envs.registration as gym_registration

gym_registration.register(
    id="MiniWorld-LookAtObjs-v0",
    entry_point="envs.lookatobjs:LookAtObjs",
)
