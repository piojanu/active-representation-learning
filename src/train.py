import os
import os.path as osp
import sys
import time

import hydra
import torch
from omegaconf import OmegaConf

from algos.dummy import DummyAgent
from algos.ppo import PPO
from envs import make_vec_env
from models.actor_critic import ActorCritic
from models.baselines import (
    ConstantActorCritic,
    RandomActorCritic,
    RandomRepeatActorCritic,
)
from utils.logx import EpochLogger
from utils.namesgenerator import get_random_name
from utils.storage import RolloutStorage

OBS_WIDTH = 64
OBS_HEIGHT = 64


@hydra.main(config_path="spec", config_name="config")
def main(cfg):
    # WA: CUDA out of memory when starting next experiment in the sweep
    time.sleep(5)

    print(OmegaConf.to_yaml(cfg))
    assert cfg.training.num_steps % cfg.training.num_processes == 0

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(1)

    device = torch.device("cuda:0" if cfg.training.cuda else "cpu")

    ckpt_dir = "./checkpoints"
    weights_dir = osp.join(ckpt_dir, "weights")
    os.makedirs(ckpt_dir)
    os.makedirs(weights_dir)

    env = make_vec_env(
        cfg.env,
        cfg.training.num_processes,
        device=device,
        seed=cfg.seed,
        agent_obs_size=(26, 26),
        encoder_cfg=cfg.encoder,
        gym_kwargs=dict(obs_width=OBS_WIDTH, obs_height=OBS_HEIGHT),
    )

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
        raise KeyError(f"Agent {cfg.agent.algo} not supported")

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

    # Prepare logger
    logger = EpochLogger()

    if isinstance(agent, DummyAgent):
        local_agent_log_interval = None
        local_agent_save_interval = None
    else:
        # Change intervals unit into local steps
        local_agent_log_interval = cfg.agent.logging.log_interval * local_num_steps
        local_agent_save_interval = cfg.agent.logging.save_interval * local_num_steps

    if cfg.encoder.algo.lower() == "dummy":
        local_encoder_log_interval = None
    else:
        # Change intervals unit into local steps
        local_encoder_log_interval = (
            cfg.encoder.logging.log_interval // cfg.encoder.num_updates
        )

    # Change intervals unit into local steps
    local_rollout_log_interval = cfg.agent.logging.log_interval * local_num_steps

    # Training loop
    local_total_steps = int(cfg.training.total_steps // cfg.training.num_processes)
    start_time = time.time()
    for local_step in range(local_total_steps):
        global_step_plus_one = (local_step + 1) * cfg.training.num_processes
        epoch_step = local_step % local_num_steps

        # Sample actions
        with torch.no_grad():
            (
                value,
                action,
                action_log_prob,
                recurrent_hidden_states,
            ) = actor_critic.act(
                rollouts.obs[epoch_step],
                rollouts.recurrent_hidden_states[epoch_step],
                rollouts.non_terminal_masks[epoch_step],
            )

        # Observe reward and next obs
        obs, reward, done, infos = env.step(action)
        for info in infos:
            if "episode" in info.keys():
                logger.store(
                    RolloutReturn=info["episode"]["r"],
                    RolloutLength=info["episode"]["l"],
                )
            if "encoder" in info.keys():
                logger.store(
                    LossEncoder=info["encoder"]["loss"],
                    EncoderUpdates=info["encoder"]["total_updates"],
                )

        # If done then clean the history of observations
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

        # If the buffer filled, then train
        if epoch_step + 1 == local_num_steps:
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
            logger.store(**info)

        # Bookkeeping from here til the end of the training loop

        dump_logs = False

        # If it's time to checkpoint agent...
        if local_agent_save_interval is not None and (
            (local_step + 1) % local_agent_save_interval == 0
            or (local_step + 1) == local_total_steps
        ):
            torch.save(
                [
                    actor_critic,
                    agent.optimizer.state_dict(),
                ],
                osp.join(ckpt_dir, "model.pkl"),
            )
            torch.save(
                actor_critic.state_dict(),
                osp.join(
                    weights_dir,
                    f"{(local_step + 1) * cfg.training.num_processes // 1000}k.pt",
                ),
            )

        # If it's time to log encoder...
        if (
            local_encoder_log_interval is not None
            and local_step >= cfg.encoder.buffer_size
            and (local_step - cfg.encoder.buffer_size + 1) % local_encoder_log_interval
            == 0
        ):
            dump_logs = True

            logger.log_tabular("LossEncoder")
            logger.log_tabular(
                "EncoderUpdates", average_only=True, with_min_and_max=True
            )

        # If it's time to log agent...
        if (
            local_agent_log_interval is not None
            and (local_step + 1) % local_agent_log_interval == 0
        ):
            dump_logs = True

            logger.log_tabular("LossValue")
            logger.log_tabular("LossPolicy")
            logger.log_tabular("DistEntropy", with_min_and_max=True)
            logger.log_tabular("ApproxKL", with_min_and_max=True)
            logger.log_tabular("PPOUpdates", average_only=True)
            logger.log_tabular("PolicyInterations", (local_step + 1) // local_num_steps)

        # If it's time to log rollout...
        if (local_step + 1) % local_rollout_log_interval == 0:
            dump_logs = True

            # Record the last iteration rollouts
            logger.writer.add_video(
                "RolloutsBuffer",
                torch.transpose(rollouts.obs, 0, 1).type(torch.uint8),
                global_step_plus_one,
                fps=15,
            )

            # TODO: Uncomment this when working with episodic envs again
            # logger.log_tabular("RolloutReturn", with_min_and_max=True)
            # logger.log_tabular("RolloutLength", with_min_and_max=True)
            # logger.log_tabular(
            #     "RolloutNumber", len(logger.histogram_dict["RolloutReturn/Hist"])
            # )

            elapsed_time = time.time() - start_time
            start_time = time.time()

            logger.log_tabular(
                "StepsPerSecond",
                local_rollout_log_interval * cfg.training.num_processes / elapsed_time,
            )
            logger.log_tabular(
                "ETAinMins",
                (
                    elapsed_time
                    / local_rollout_log_interval
                    * (local_total_steps - local_step - 1)
                    // 60
                ),
            )

        if dump_logs:
            logger.dump_tabular(global_step_plus_one)

        if (local_step + 1) % local_num_steps == 0:
            rollouts.after_update()


if __name__ == "__main__":
    try:
        sys.argv.remove("--mock")
        is_mock = True
    except ValueError:
        is_mock = False

    try:
        idx = sys.argv.index("--resume")
        resume_name = sys.argv[idx + 1]

        # Remove the argument
        del sys.argv[idx]
        # Remove its value
        del sys.argv[idx]
    except ValueError:
        resume_name = None

    # Get the run name
    if is_mock:
        run_name = "mock"
    elif resume_name is not None:
        run_name = resume_name
    else:
        run_name = get_random_name()

    # Register the name
    OmegaConf.register_new_resolver("run_name", lambda: run_name)
    print(f'This run name is "{run_name}", good luck!')

    # Run!
    main()
