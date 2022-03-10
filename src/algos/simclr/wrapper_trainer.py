import os
import os.path as osp

import gym
import torch
import torchvision
from simclr.simclr import SimCLR

from algos.simclr import NT_Xent
from models.resnet import ResNetEncoder


class TrainSimCLR(gym.Wrapper):
    """Collects data and trains SimCLR encoder, where loss drop is reward."""

    def __init__(
        self,
        env,
        rank,
        device_name,
        buffer_size,
        learning_rate,
        mini_batch_size,
        mixing_coef,
        num_processes,
        num_updates,
        projection_dim,
        temperature,
        log_interval,
        save_interval,
    ):
        super().__init__(env)

        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.mixing_coef = mixing_coef
        self.num_processes = num_processes
        self.num_updates = num_updates
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.device = torch.device(device_name)
        self.buffer = torch.zeros(self.buffer_size, *self.observation_space.shape).to(
            self.device
        )

        self.counter = 0
        self.total_updates = 0
        self.update_every = 1 / self.num_updates if self.num_updates < 1 else None

        self.ckpt_dir = osp.join("./checkpoints", f"encoder_{rank}")
        os.makedirs(self.ckpt_dir)

        # Create SimCLR transformation
        transforms = torch.nn.Sequential(
            # Normalize observations into [0, 1] range as required for floats
            torchvision.transforms.Normalize([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]),
            torchvision.transforms.RandomResizedCrop(
                size=self.observation_space.shape[1:]
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply(
                torch.nn.ModuleList(
                    [torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)]
                ),
                p=0.8,
            ),
            torchvision.transforms.RandomGrayscale(p=0.2),
        )
        self.scripted_transforms = torch.jit.script(transforms)

        # Create SimCLR encoder
        self.encoder = ResNetEncoder(layers=[2, 2])
        n_features = self.encoder(torch.randn(1, *env.observation_space.shape)).shape[1]
        self.encoder.to(self.device)

        # Create SimCLR projector
        self.model = SimCLR(self.encoder, projection_dim, n_features)
        self.model.to(self.device)

        # Create SimCLR criterion
        self.criterion = NT_Xent(self.mini_batch_size, temperature, world_size=1)

        # Create Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Save checkpoint at step zero
        self.save_checkpoint(0)

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

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self.counter < self.buffer_size:
            self.buffer[self.counter].copy_(torch.from_numpy(obs))
            mix_rew = 0.0
        else:
            # Replace a random buffer's element with the observation
            ptr = torch.randint(self.buffer_size, size=(1,))
            self.buffer[ptr].copy_(torch.from_numpy(obs))

            # Run fractional, only one, or multiple SimCLR updates
            num_updates = (
                self.num_updates
                if self.num_updates >= 1
                else int((self.counter - self.buffer_size) % self.update_every == 0)
            )
            for _ in range(0, num_updates):
                # Sample mini batch
                idxs = torch.randint(self.buffer_size, size=(self.mini_batch_size,))
                mini_batch = self.buffer[idxs]

                # TODO: Should we use the loss after the update as the reward signal?
                x_i, x_j = self.transform_batch(mini_batch)
                loss, confusion_matrix = self.compute_loss(x_i, x_j)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.total_updates += num_updates

            if num_updates > 0:
                with torch.no_grad():
                    mix_rew = self.mixing_coef * (5.0 - loss.item())

                info["encoder"] = dict(
                    loss=loss.item(),
                    total_updates=self.total_updates,
                )

                # Send (quite big) confusion matrix only when it's time for logging
                if self.total_updates % self.log_interval == 0:
                    info["encoder"]["confusion_matrix"] = confusion_matrix

                # Checkpoint
                if self.total_updates % self.save_interval == 0:
                    self.save_checkpoint(self.counter + 1)
            else:
                mix_rew = 0.0

        self.counter += 1
        mix_rew += (1 - self.mixing_coef) * rew

        return obs, mix_rew, done, info
