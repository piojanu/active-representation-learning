import os
import os.path as osp
import queue
import threading

import gym
import torch
import torchvision
from simclr.simclr import SimCLR

from algos.simclr import NT_Xent
from nets.convnet import ConvNetEncoder


def collate_batch_of_pairs(batch):
    # Transpose to place the batch dim. as second
    x_i, x_j = list(map(list, zip(*batch)))
    return torch.concat(x_i), torch.concat(x_j)


class _Dataset(torch.utils.data.IterableDataset):
    def __init__(self, buffer_shape, transforms):
        """Dataset sampling from the shared memory buffer which alters its content

        Args:
          buffer_shape (tuple): observations buffer shape (<buff. size>, *<obs. shape>).
          transforms (callable): transforms to apply to observations.
        """
        self.buffer_size = buffer_shape[0]
        self.transforms = transforms

        self.scripted_transforms = None
        # Allocate shared memory buffer for observations
        self.buffer = torch.empty(buffer_shape).share_memory_()

    def __iter__(self):
        if self.scripted_transforms is None:
            # NOTE: This boosts the performance by ~1.5x steps/sec
            self.scripted_transforms = torch.jit.script(self.transforms)

        return self

    def __next__(self):
        sample = self.buffer[torch.randint(self.buffer_size, size=(1,))]

        x_i = self.scripted_transforms(sample)
        x_j = self.scripted_transforms(sample)

        return (x_i, x_j)

    def insert(self, obs, ptr=None):
        if ptr is None:
            # Replace a random buffer's element with the new observation
            ptr = torch.randint(self.buffer_size, size=(1,))
        # NOTE: Possible race condition here: prefetching reading while you write here
        self.buffer[ptr].copy_(torch.from_numpy(obs))


class _Worker(threading.Thread):
    def __init__(
        self,
        observation_shape,
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
        obs_queue,
        loss_buffer,
        loss_ready,
    ):
        super().__init__()
        del log_interval
        assert num_updates >= 1

        self.buffer_size = buffer_size
        self.num_updates = num_updates
        self.save_interval = save_interval

        self.num_processes = num_processes
        self.device = torch.device(device_name)

        self.obs_queue = obs_queue
        self.loss_buffer = loss_buffer
        self.loss_ready = loss_ready

        self.local_num_steps = loss_buffer.shape[0]
        self.total_updates = 0
        self.total_steps = 0

        self.ckpt_dir = osp.join("./checkpoints", f"encoder_{my_rank}")
        os.makedirs(self.ckpt_dir)

        # Create SimCLR transformation
        # NOTE: These are non-parametric transforms, so there is nothing to move to GPU
        transforms = torch.nn.Sequential(
            # Normalize observations into [0, 1] range as required for floats
            torchvision.transforms.Normalize([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]),
            torchvision.transforms.RandomResizedCrop(size=observation_shape[1:]),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply(
                torch.nn.ModuleList(
                    [torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)]
                ),
                p=0.8,
            ),
            torchvision.transforms.RandomGrayscale(p=0.2),
        )

        # Create the data loader
        self.dataset = _Dataset(
            (buffer_size,) + observation_shape,
            transforms,
        )
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=mini_batch_size,
            num_workers=4,
            collate_fn=collate_batch_of_pairs,
            pin_memory=True,
            # NOTE: Preprocessing is a bottleneck so prefetching more doesn't make any difference
            #       Moreover, this can't be too big as prefetch is done on the old data
            prefetch_factor=2,
            persistent_workers=True,
        )
        self.data_iter = None

        # Create SimCLR encoder
        n_features = 128
        self.encoder = ConvNetEncoder(n_features)
        self.encoder.to(self.device, non_blocking=True)

        # Create SimCLR projector
        self.model = SimCLR(self.encoder, projection_dim, n_features)
        self.model.to(self.device, non_blocking=True)

        # Create SimCLR criterion
        self.criterion = NT_Xent(mini_batch_size, temperature, world_size=1)

        # Create Adam optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Save checkpoint at step zero
        self.save_checkpoint(0)

    def run(self):
        while True:
            # NOTE: This queue fills quickly, so data collection is not a bottleneck
            new_obs = self.obs_queue.get()
            if new_obs is None:
                break

            if self.total_steps < self.buffer_size:
                self.dataset.insert(new_obs, self.total_steps)

                self.total_steps += 1
            else:
                self.dataset.insert(new_obs)

                if self.data_iter is None:
                    # NOTE: Prefetching starts after you call `iter` on the data loader
                    self.data_iter = iter(self.data_loader)

                for _ in range(self.num_updates):
                    x_i, x_j = next(self.data_iter)
                    loss, _ = self.compute_loss(
                        # NOTE: If you won't block here, then CUDA goes out of memory
                        x_i.to(self.device, non_blocking=False),
                        x_j.to(self.device, non_blocking=False),
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.total_steps += 1
                local_step = self.total_steps % self.local_num_steps
                total_updates = (self.total_steps - self.buffer_size) * self.num_updates

                with torch.no_grad():
                    self.loss_buffer[local_step].copy_(loss, non_blocking=True)
                    # TODO: Log the last conf. matrix (only last as it's quite big).

                if total_updates % self.save_interval == 0:
                    self.save_checkpoint(self.total_steps + 1)

                if self.total_steps % self.local_num_steps == 0:
                    self.loss_ready.wait()

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

        # NOTE: Without limiting threads number the CPU is the bottleneck
        torch.set_num_threads(2)

        self.obs_queue = queue.SimpleQueue()
        self.loss_buffer = torch.zeros(
            local_num_steps, device=torch.device(device_name)
        )
        self.loss_ready = threading.Barrier(parties=2)

        self.worker = _Worker(
            observation_shape=self.observation_space.shape,
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
            obs_queue=self.obs_queue,
            loss_buffer=self.loss_buffer,
            loss_ready=self.loss_ready,
        )
        self.worker.start()

        self.local_num_steps = local_num_steps
        self.buffer_size = buffer_size
        self.num_updates = num_updates

        self.total_steps = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.total_steps += 1

        self.obs_queue.put_nowait(obs)

        if (
            self.total_steps >= self.buffer_size
            and self.total_steps % self.local_num_steps == 0
        ):
            self.loss_ready.wait()

            info["encoder"] = dict(
                losses=self.loss_buffer.tolist(),
                total_updates=(self.total_steps - self.buffer_size) * self.num_updates,
            )

        return obs, rew, done, info

    def close(self):
        try:
            while True:
                self.obs_queue.get_nowait()
        except queue.Empty:
            self.obs_queue.put(None)

        self.env.close()
        self.worker.join()
