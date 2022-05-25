import gym
import numpy as np
import torch
from a2c_ppo_acktr.envs import TimeLimitMask, TransposeImage
from gym.wrappers import TimeLimit
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from torchvision.transforms import Resize

from algos.simclr import TrainSimCLR


def make_env(
    env_name, local_num_steps, rank, num_processes, seed, encoder_cfg, gym_kwargs
):
    def _get_device_name(rank):
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                return "cuda:0"

            # Distribute environments evenly across GPUs
            return "cuda:" + str(rank % torch.cuda.device_count())
        else:
            return "cpu"

    def _thunk():
        # Lazy import MiniWorld to avoid OGL initialization errors in the access node
        import gym_miniworld  # trunk-ignore(flake8/F401)

        # Register custom gym environments
        import envs  # trunk-ignore(flake8/F401)

        env = gym.make(env_name, **gym_kwargs)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)

        if is_wrapped(env, TimeLimit):
            env = TimeLimitMask(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        if encoder_cfg.algo.lower() == "simclr":
            env = TrainSimCLR(
                env,
                local_num_steps,
                rank,
                num_processes,
                _get_device_name(rank),
                seed,
                projection_dim=encoder_cfg.model.projection_dim,
                temperature=encoder_cfg.model.temperature,
                buffer_size=encoder_cfg.training.buffer_size,
                learning_rate=encoder_cfg.training.learning_rate,
                mini_batch_size=encoder_cfg.training.mini_batch_size,
                num_updates=encoder_cfg.training.num_updates,
                preproc_ratio=encoder_cfg.training.preproc_ratio,
                log_interval=encoder_cfg.logging.log_interval,
                save_interval=encoder_cfg.logging.save_interval,
            )
        else:
            if encoder_cfg.algo.lower() != "dummy":
                raise KeyError(f"Encoder {encoder_cfg.algo} not supported")

        return env

    return _thunk


def make_vec_env(
    env_name,
    local_num_steps,
    num_processes,
    device,
    seed,
    agent_obs_size,
    encoder_cfg,
    gym_kwargs,
):
    if num_processes > 1:
        env = SubprocVecEnv(
            [
                make_env(
                    env_name,
                    local_num_steps,
                    idx,
                    num_processes,
                    seed,
                    encoder_cfg,
                    gym_kwargs,
                )
                for idx in range(num_processes)
            ]
        )
    else:
        env = DummyVecEnv(
            [
                make_env(
                    env_name,
                    local_num_steps,
                    0,
                    num_processes,
                    seed,
                    encoder_cfg,
                    gym_kwargs,
                )
            ]
        )
    env = PyTorchToDevice(env, device)
    env = PyTorchResizeObs(env, agent_obs_size)

    return env


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
        obs = torch.from_numpy(obs).to(
            self.device, dtype=torch.float, non_blocking=True
        )
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).to(
            self.device, dtype=torch.float, non_blocking=True
        )
        rew = (
            torch.from_numpy(rew)
            .unsqueeze(dim=1)
            .to(self.device, dtype=torch.float, non_blocking=True)
        )
        return obs, rew, done, info
