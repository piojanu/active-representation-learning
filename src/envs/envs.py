import gym
import gym_miniworld
import torch
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper

from a2c_ppo_acktr.envs import TimeLimitMask
from a2c_ppo_acktr.envs import TransposeImage

def make_env(env_name, **kwargs):
    def _thunk():
        env = gym.make(env_name, **kwargs)
        env = RecordEpisodeStatistics(env)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_env(env_name, num_processes, device, **kwargs):
    if num_processes > 1:
        env = SubprocVecEnv([
            make_env(env_name, **kwargs) for _ in range(num_processes)])
    else:
        env = DummyVecEnv([make_env(env_name, **kwargs)])
    env = PyTorchToDevice(env, device)

    return env

class PyTorchToDevice(VecEnvWrapper):
    def __init__(self, venv, device):
        super().__init__(venv)

        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        rew = torch.from_numpy(rew).unsqueeze(dim=1).float().to(self.device)
        return obs, rew, done, info