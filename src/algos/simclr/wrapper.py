import os
import os.path as osp
import queue
import threading
from collections import namedtuple

import gym
import torch
import torchvision
from simclr.simclr import SimCLR

from algos.simclr import NT_Xent
from nets.convnet import ConvNetEncoder

_Command = namedtuple("Command", ["name", "data"], defaults=[None])


class _Daemon(threading.Thread):
    def __init__(
        self,
        observation_space,
        projection_dim,
        temperature,
        buffer_size,
        learning_rate,
        mini_batch_size,
        num_updates,
        log_interval,
        save_interval,
        # World params
        my_rank,
        num_processes,
        device_name,
        # Parent comm
        commands_queue,
        loss_ready,
        loss_buffer,
    ):
        super().__init__(daemon=True)
        del log_interval

        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.num_updates = num_updates
        self.save_interval = save_interval

        self.num_processes = num_processes
        self.device = torch.device(device_name)

        self.commands_queue = commands_queue
        self.loss_ready = loss_ready
        self.loss_buffer = loss_buffer
        self.local_num_steps = loss_buffer.shape[0]

        self.buffer_size_ones = torch.ones(self.buffer_size)
        self.update_every = 1 / self.num_updates if self.num_updates < 1 else None

        self.step_count = 0
        self.total_updates = 0

        self.ckpt_dir = osp.join("./checkpoints", f"encoder_{my_rank}")
        os.makedirs(self.ckpt_dir)

        # Allocate replay buffer
        self.buffer = torch.zeros(
            self.buffer_size, *observation_space.shape, device=self.device
        )

        # Create SimCLR transformation
        transforms = torch.nn.Sequential(
            # Normalize observations into [0, 1] range as required for floats
            torchvision.transforms.Normalize([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]),
            torchvision.transforms.RandomResizedCrop(size=observation_space.shape[1:]),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply(
                torch.nn.ModuleList(
                    [torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)]
                ),
                p=0.8,
            ),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ).to(self.device, non_blocking=True)
        self.scripted_transforms = torch.jit.script(transforms)

        # Create SimCLR encoder
        n_features = 128
        self.encoder = ConvNetEncoder(n_features)
        self.encoder.to(self.device, non_blocking=True)

        # Create SimCLR projector
        self.model = SimCLR(self.encoder, projection_dim, n_features)
        self.model.to(self.device, non_blocking=True)

        # Create SimCLR criterion
        self.criterion = NT_Xent(self.mini_batch_size, temperature, world_size=1)

        # Create Adam optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Save checkpoint at step zero
        self.save_checkpoint(0)

    def run(self):
        while True:
            cmd = self.commands_queue.get()
            if cmd.name == "NEW_OBS":
                obs = cmd.data
                if self.step_count < self.buffer_size:
                    self.buffer[self.step_count].copy_(
                        torch.from_numpy(obs), non_blocking=True
                    )
                else:
                    # Replace a random buffer's element with the observation
                    ptr = torch.randint(self.buffer_size, size=(1,))
                    self.buffer[ptr].copy_(torch.from_numpy(obs), non_blocking=True)

                    # Run fractional, only one, or multiple SimCLR updates
                    num_updates = (
                        self.num_updates
                        if self.num_updates >= 1
                        else int(
                            (self.step_count - self.buffer_size) % self.update_every
                            == 0
                        )
                    )
                    for _ in range(0, num_updates):
                        # Sample mini batch (without replacement)
                        idxs = self.buffer_size_ones.multinomial(self.mini_batch_size)
                        mini_batch = self.buffer[idxs]

                        # TODO: Should we use the loss after the update as the reward signal?
                        x_i, x_j = self.transform_batch(mini_batch)
                        loss, _ = self.compute_loss(x_i, x_j)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    self.total_updates += num_updates

                    if num_updates > 0:
                        with torch.no_grad():
                            self.loss_buffer[
                                self.step_count % self.local_num_steps
                            ].copy_(loss, non_blocking=True)

                            # TODO: Log the last conf. matrix (only last as it's quite big).

                        # Checkpoint
                        if self.total_updates % self.save_interval == 0:
                            self.save_checkpoint(self.step_count + 1)
                self.step_count += 1
            elif cmd.name == "SIGNAL_READY":
                self.loss_ready.wait()
            elif cmd.name == "CLOSE":
                break
            else:
                raise ValueError("Unknown command")

    def transform_batch(self, batch):
        x_i = torch.empty_like(batch)
        x_j = torch.empty_like(batch)
        for idx in range(0, batch.shape[0]):
            x_i[idx] = self.scripted_transforms(batch[idx])
            x_j[idx] = self.scripted_transforms(batch[idx])

        return x_i, x_j

    def compute_loss(self, x_i, x_j):
        # TODO: You might want to concat. the positive pair into a single batch
        #       for better performance. Same with the transformations.
        _, _, z_i, z_j = self.model(x_i, x_j)

        return self.criterion(z_i, z_j)

    def save_checkpoint(self, local_step):
        torch.save(
            [
                self.encoder,
                self.model.projector,
                self.optimizer.state_dict(),
            ],
            osp.join(self.ckpt_dir, "checkpoint.pkl"),
        )
        torch.save(
            [self.encoder.state_dict(), self.model.projector.state_dict()],
            # Align the file name to the log step
            osp.join(self.ckpt_dir, f"{local_step * self.num_processes}.pt"),
        )


class TrainSimCLR(gym.Wrapper):
    """Collects data and trains SimCLR encoder, where loss drop is reward."""

    def __init__(
        self,
        env,
        local_num_steps,
        my_rank,
        num_processes,
        device_name,
        seed,
        # Encoder params
        projection_dim,
        temperature,
        buffer_size,
        learning_rate,
        mini_batch_size,
        num_updates,
        log_interval,
        save_interval,
    ):
        super().__init__(env)

        torch.manual_seed(seed + my_rank)
        torch.cuda.manual_seed_all(seed + my_rank)

        # NOTE: Without it the CPU is the bottleneck and we don't utilise the GPUs fully
        torch.set_num_threads(2)

        self.commands_queue = queue.SimpleQueue()
        self.loss_ready = threading.Barrier(parties=2)
        self.loss_buffer = torch.zeros(
            local_num_steps, device=torch.device(device_name)
        )

        self.daemon = _Daemon(
            observation_space=self.observation_space,
            projection_dim=projection_dim,
            temperature=temperature,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            num_updates=num_updates,
            log_interval=log_interval,
            save_interval=save_interval,
            my_rank=my_rank,
            num_processes=num_processes,
            device_name=device_name,
            commands_queue=self.commands_queue,
            loss_ready=self.loss_ready,
            loss_buffer=self.loss_buffer,
        )
        self.daemon.start()

        self.local_num_steps = local_num_steps
        self.buffer_size = buffer_size
        self.num_updates = num_updates

        self.step_count = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.commands_queue.put_nowait(_Command("NEW_OBS", obs))
        self.step_count += 1

        if self.step_count % self.local_num_steps == 0:
            self.commands_queue.put_nowait(_Command("SIGNAL_READY"))
            self.loss_ready.wait()

            info["encoder"] = dict(
                losses=self.loss_buffer.tolist(),
                total_updates=max(0, self.step_count - self.buffer_size)
                * self.num_updates,
            )

        return obs, rew, done, info

    def close(self):
        self.commands_queue.put_nowait(_Command("CLOSE"))
        self.env.close()
