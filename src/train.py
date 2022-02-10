import os
import os.path as osp
import sys
import time

import hydra
import torch
from gym.wrappers import RecordEpisodeStatistics
from omegaconf import OmegaConf

from algos.dummy import DummyAgent
from algos.ppo import PPO
from envs import EpisodeInfoLogger, make_vec_env
from models.actor_critic import ActorCritic
from models.baselines import (
    ConstantActorCritic,
    DummyActorCritic,
    RandomActorCritic,
    RandomRepeatActorCritic,
)
from utils.logx import EpochLogger
from utils.namesgenerator import get_random_name
from utils.storage import RolloutStorage
from wrappers import TrainSimCLR

OBS_WIDTH = 64
OBS_HEIGHT = 64


@hydra.main(config_path="spec", config_name="config")
def main(cfg):
    # WA: CUDA out of memory when starting next experiment in the sweep
    time.sleep(5)

    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(1)

    device = torch.device("cuda:0" if cfg.training.cuda else "cpu")

    env = make_vec_env(
        cfg.env,
        cfg.training.num_processes,
        device=device,
        seed=cfg.seed,
        agent_obs_size=(26, 26),
        encoder_kwargs=cfg.encoder,
        gym_kwargs=dict(obs_width=OBS_WIDTH, obs_height=OBS_HEIGHT),
    )

    env_info_loggers = []
    if all(env.env_is_wrapped(RecordEpisodeStatistics)):
        env_info_loggers.append(EpisodeInfoLogger)
    if all(env.env_is_wrapped(TrainSimCLR)):
        env_info_loggers.append(TrainSimCLR)

    if cfg.agent.algo.lower() == "ppo":
        actor_critic = ActorCritic(
            env.observation_space.shape,
            env.action_space,
            hidden_size=cfg.agent.model.hidden_size,
            recurrent=cfg.agent.model.recurrent,
        )
        actor_critic.to(device)

        agent = PPO(
            actor_critic,
            cfg.agent.loss.clip_ratio_pi,
            cfg.agent.loss.clip_ratio_vf,
            cfg.agent.loss.entropy_coef,
            cfg.agent.loss.value_coef,
            cfg.agent.training.learning_rate,
            cfg.agent.training.num_epochs,
            cfg.agent.training.max_grad_norm,
            cfg.agent.training.max_kl,
            cfg.agent.training.mini_batch_size,
        )
    elif cfg.agent.algo.lower() == "constant":
        actor_critic = ConstantActorCritic(env.action_space, cfg.agent.action)
        agent = DummyAgent()
    elif cfg.agent.algo.lower() == "random":
        actor_critic = RandomActorCritic(env.action_space)
        agent = DummyAgent()
    elif cfg.agent.algo.lower() == "randomrepeat":
        actor_critic = RandomRepeatActorCritic(
            env.action_space, cfg.agent.min_repeat, cfg.agent.max_repeat
        )
        agent = DummyAgent()
    else:
        raise KeyError(f"Algorithm {cfg.agent.algo} not supported")

    local_num_steps = cfg.training.num_steps // cfg.training.num_processes
    rollouts = RolloutStorage(
        local_num_steps,
        cfg.training.num_processes,
        env.observation_space.shape,
        env.action_space,
        actor_critic.recurrent_hidden_state_size,
    )
    obs = env.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    logger = EpochLogger()

    num_iterations = int(cfg.training.total_steps // cfg.training.num_steps)
    start_time = time.time()
    for itr in range(num_iterations):
        for step in range(local_num_steps):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.non_terminal_masks[step],
                )

            # Observe reward and next obs
            obs, reward, done, infos = env.step(action)
            for info in infos:
                for info_logger in env_info_loggers:
                    info_logger.log_info(logger, info)

            # If done then clean the history of observations.
            non_terminal_masks = torch.FloatTensor(
                [[1.0] if not done_ else [0.0] for done_ in done]
            )
            bad_masks = torch.FloatTensor(
                [[1.0] if "bad_transition" in info.keys() else [0.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                non_terminal_masks,
                bad_masks,
            )

        with torch.no_grad():
            last_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.non_terminal_masks[-1],
            ).detach()

        rollouts.compute_returns(
            last_value,
            cfg.rollout.gae_lambda,
            cfg.rollout.gamma,
            cfg.rollout.bootstrap_value_at_time_limit,
            cfg.rollout.force_non_episodic,
        )

        info = agent.update(rollouts)
        agent.log_info(logger, info)

        if not isinstance(actor_critic, DummyActorCritic) and (
            (itr + 1) % cfg.logging.save_interval == 0 or itr == num_iterations - 1
        ):
            ckpt_dir = "./checkpoints"
            weights_dir = osp.join(ckpt_dir, "weights")
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(weights_dir, exist_ok=True)
            torch.save(
                [
                    actor_critic,
                    agent.optimizer.state_dict(),
                ],
                osp.join(ckpt_dir, "model.pkl"),
            )
            torch.save(
                actor_critic.state_dict(), osp.join(weights_dir, f"{itr + 1}.pt")
            )

        if (itr + 1) % cfg.logging.log_interval == 0:
            last_num_steps = cfg.logging.log_interval * cfg.training.num_steps

            for info_logger in env_info_loggers:
                info_logger.compute_stats(logger)
            agent.compute_stats(logger)
            logger.log_tabular(
                "StepsPerSecond", last_num_steps / (time.time() - start_time)
            )
            logger.log_tabular(
                "ETAinMins",
                (
                    (time.time() - start_time)
                    / cfg.logging.log_interval
                    * (num_iterations - itr - 1)
                    // 60
                ),
            )

            start_time = time.time()

        if (itr + 1) % cfg.logging.log_interval == 0 or itr == num_iterations - 1:
            # Record the last iteration rollouts
            logger.writer.add_video(
                "RolloutsBuffer",
                torch.transpose(rollouts.obs, 0, 1).type(torch.uint8),
                itr + 1,
                fps=15,
            )

            logger.log_tabular("TotalEnvInteracts", (itr + 1) * cfg.training.num_steps)
            logger.dump_tabular(itr + 1)

        rollouts.after_update()


if __name__ == "__main__":
    try:
        sys.argv.remove("--mock")
        is_mock = True
    except ValueError:
        is_mock = False

    # Get and register this run name
    run_name = get_random_name() if not is_mock else "mock"
    OmegaConf.register_new_resolver("run_name", lambda: run_name)
    print(f'This run name is "{run_name}", good luck!')

    main()
