import gym

# trunk-ignore(flake8/F401)
import gym_miniworld
import numpy as np
import torch
from a2c_ppo_acktr.envs import TimeLimitMask, TransposeImage
from gym.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from torchvision.transforms import Resize

from utils.logx import InfoLogger
from wrappers import TrainSimCLR


def make_env(env_name, proc_idx, encoder_kwargs, gym_kwargs):
    def _get_device_name(idx):
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                return "cuda:0"

            # The first GPU serves OGL and the agent training, skip it once
            if idx == 0:
                return "cuda:1"

            # Distribute remaining environments evenly across GPUs
            return "cuda:" + str(idx % torch.cuda.device_count())
        else:
            return "cpu"

    def _thunk():
        env = gym.make(env_name, **gym_kwargs)
        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        if len(encoder_kwargs) > 0:
            env = TrainSimCLR(env, _get_device_name(proc_idx), **encoder_kwargs)
        else:
            env = RecordEpisodeStatistics(env)

        return env

    return _thunk


def make_vec_env(
    env_name, num_processes, device, agent_obs_size, encoder_kwargs, gym_kwargs
):
    if num_processes > 1:
        env = SubprocVecEnv(
            [
                make_env(env_name, idx, encoder_kwargs, gym_kwargs)
                for idx in range(num_processes)
            ]
        )
    else:
        env = DummyVecEnv([make_env(env_name, 0, encoder_kwargs, gym_kwargs)])
    env = PyTorchToDevice(env, device)
    env = PyTorchResizeObs(env, agent_obs_size)

    return env


class EpisodeInfoLogger(InfoLogger):
    @staticmethod
    def log_info(logger, info):
        if "episode" in info.keys():
            logger.store(
                RolloutReturn=info["episode"]["r"],
                RolloutLength=info["episode"]["l"],
            )

    @staticmethod
    def compute_stats(logger):
        logger.log_tabular("RolloutReturn", with_min_and_max=True)
        logger.log_tabular("RolloutLength", with_min_and_max=True)
        logger.log_tabular(
            "RolloutNumber", len(logger.histogram_dict["RolloutReturn/Hist"])
        )


class PyTorchResizeObs(VecEnvWrapper):
    def __init__(self, venv, obs_size):
        super().__init__(venv)

        self.transformation = Resize(obs_size)

        new_shape = self.observation_space.shape[:1] + obs_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def reset(self):
        obs = self.venv.reset()
        return self.transformation(obs)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return self.transformation(obs), rew, done, info


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
