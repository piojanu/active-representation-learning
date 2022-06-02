import os
import os.path as osp
import queue
import threading
from copy import deepcopy
from random import randint

import gym
import torch
import torchvision
from simclr.simclr import SimCLR

from algos.simclr import NT_Xent
from nets import ShallowResNet5Encoder


def collate_batch_of_pairs(batch):
    # Place the batch dim. as second
    return torch.stack(batch, dim=1)


class _Dataset(torch.utils.data.IterableDataset):
    def __init__(self, buffer_shape, transforms, preproc_ratio):
        """Dataset sampling from the shared memory buffer which alters its content

        Args:
          buffer_shape (tuple): observations buffer shape (<buff. size>, *<obs. shape>).
          transforms (callable): transforms to apply to observations.
          preproc_ratio (float): probability of applying preprocessing to sampled
            observation. Otherwise, the pair is returned from the cache.
        """
        self.buffer_size = buffer_shape[0]
        self.transforms = transforms
        self.preproc_ratio = preproc_ratio

        # Allocate the buffer for observations
        self.buffer = torch.empty(buffer_shape)
        self.buffer_checksum = torch.zeros(self.buffer_size, dtype=torch.int)

        # Share the observations buffer with all workers
        self.buffer.share_memory_()
        self.buffer_checksum.share_memory_()

        # Cache for transformed pairs, private to each worker
        self.cache = torch.empty(self.buffer_size, 2, *buffer_shape[1:])
        self.cache_checksum = torch.zeros(self.buffer_size, dtype=torch.int)

        self.indices = []
        self.scripted_transforms = None

    def __iter__(self):
        if self.scripted_transforms is None:
            # NOTE: This boosts the performance by ~1.5x steps/sec
            self.scripted_transforms = torch.jit.script(self.transforms)

        return self

    def __next__(self):
        if len(self.indices) == 0:
            # NOTE: It's okay for ptr-s to repeat across workers, because each one holds
            #       a private cache with differently transformed pairs.
            self.indices = torch.randperm(self.buffer_size).tolist()
        ptr = self.indices.pop()

        if (
            torch.rand(1) < self.preproc_ratio
            or self.cache_checksum[ptr] != self.buffer_checksum[ptr]
        ):
            self.fill_cache(ptr)

        return self.cache[ptr]

    def fill_cache(self, ptr):
        """Fills cache with the transformed pair at the index `ptr`."""
        x_i = self.scripted_transforms(self.buffer[ptr])
        x_j = self.scripted_transforms(self.buffer[ptr])

        self.cache[ptr] = torch.stack([x_i, x_j])
        self.cache_checksum[ptr] = self.buffer_checksum[ptr]

    def insert(self, obs, ptr=None):
        if ptr is None:
            # Replace a random buffer's element with the new observation
            ptr = torch.randint(self.buffer_size, size=(1,)).item()
        # NOTE: Possible race condition here: prefetching partially written or currently
        #       being written observation. Shouldn't be common because of random write
        #       and read indices.
        self.buffer[ptr].copy_(torch.from_numpy(obs))
        self.buffer_checksum[ptr] += 1


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
        preproc_ratio,
        log_interval,
        # World params
        local_num_steps,
        my_rank,
        num_processes,
        device_name,
        # Parent comm
        data_queue,
        info_queue,
    ):
        super().__init__()
        assert num_updates >= 1

        self.buffer_size = buffer_size
        self.num_updates = num_updates
        self.local_log_interval = log_interval * local_num_steps

        self.local_num_steps = local_num_steps
        self.num_processes = num_processes
        self.device = torch.device(device_name)

        self.data_queue = data_queue
        self.info_queue = info_queue

        self.all_losses = torch.zeros(local_num_steps, device=self.device)
        self.cumulative_losses = [0.0]
        self.last_losses = list()

        self.episode_steps = 0
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
            preproc_ratio,
        )
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=mini_batch_size,
            num_workers=3,
            collate_fn=collate_batch_of_pairs,
            pin_memory=True,
            # NOTE: Preprocessing is a bottleneck so prefetching more doesn't make much
            #       difference. Moreover, this can't be too big as prefetch is done on
            #       the old data and the new data is being written to the buffer.
            prefetch_factor=mini_batch_size // 2,
        )

        # Create SimCLR encoder
        encoder = ShallowResNet5Encoder()
        n_features = encoder(torch.randn(1, *observation_shape)).shape[1]
        encoder.to(self.device, non_blocking=True)

        # Create SimCLR projector
        self.model = SimCLR(encoder, projection_dim, n_features)
        self.model.to(self.device, non_blocking=True)

        # Create SimCLR criterion
        self.criterion = NT_Xent(mini_batch_size, temperature, world_size=1)

        # Create Adam optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Copy the initial model and optimizer state dicts for resetting
        self.model_init_state = deepcopy(self.model.state_dict())
        self.optim_init_state = deepcopy(self.optimizer.state_dict())

        # Checkpoint at step zero
        self.checkpoint(0)

    def increment_counters(self):
        self.episode_steps += 1
        self.total_steps += 1

    def reset(self):
        self.episode_steps = 0
        self.model.load_state_dict(self.model_init_state)
        self.optimizer.load_state_dict(self.optim_init_state)

    def run(self):
        # Fetch warm-up steps to fill the dataset
        for idx in range(self.buffer_size):
            obs, _ = self.data_queue.get()
            self.dataset.insert(obs, idx)

        # NOTE: Prefetching starts after you call `iter` on the data loader
        data_iter = iter(self.data_loader)

        while True:
            obs_done = self.data_queue.get()
            if obs_done is None:
                break
            new_obs, done = obs_done

            if done:
                self.cumulative_losses.append(0.0)
                self.last_losses.append(loss.item())  # trunk-ignore(flake8/F821)

                # Checkpoint at end of episode (before reset)
                self.checkpoint(self.total_steps)
                self.reset()

            if self.episode_steps < self.buffer_size:
                self.dataset.insert(new_obs, self.episode_steps)
            else:
                self.dataset.insert(new_obs)
            self.increment_counters()

            for _ in range(self.num_updates):
                batch = next(data_iter)
                loss, conf_matrix = self.compute_loss(
                    # NOTE: If you won't block here, then CUDA goes out of memory
                    batch.to(self.device, non_blocking=False)
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.all_losses[self.total_steps % self.local_num_steps].copy_(
                    loss, non_blocking=True
                )
                self.cumulative_losses[-1] += loss.item()

            # Return info when it's needed for an agent update
            if self.total_steps % self.local_num_steps == 0:
                info = dict(
                    losses=self.all_losses.tolist(),
                    total_updates=self.episode_steps * self.num_updates,
                )

                # Send diagnostics only when it's time for logging
                # NOTE: Check if at least one episode has ended since the last logging
                if (
                    self.total_steps % self.local_log_interval == 0
                    and len(self.last_losses) > 0
                ):
                    info["cumulative_losses"] = self.cumulative_losses[:-1].copy()
                    info["last_losses"] = self.last_losses.copy()

                    self.cumulative_losses.clear()
                    self.cumulative_losses.append(0.0)
                    self.last_losses.clear()

                # NOTE: Logging images is very heavy and limits steps per seconds
                #       even by 15%! That's why we log them less often.
                if self.total_steps % (self.local_log_interval * 10) == 0:
                    info["confusion_matrix"] = conf_matrix.cpu().numpy()
                    info["last_batch"] = batch.numpy()

                self.info_queue.put_nowait(info)

    def compute_loss(self, batch):
        x_i, x_j = batch[0], batch[1]

        # TODO: You might want to concat. the positive pair into a single batch
        #       for better performance.
        _, _, z_i, z_j = self.model(x_i, x_j)

        return self.criterion(z_i, z_j)

    def checkpoint(self, local_step):
        torch.save(
            [
                self.model.encoder,
                self.model.projector,
                self.optimizer.state_dict(),
            ],
            osp.join(self.ckpt_dir, "checkpoint.pkl"),
        )
        torch.save(
            [self.model.encoder.state_dict(), self.model.projector.state_dict()],
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
        preproc_ratio,
        log_interval,
    ):
        super().__init__(env)

        torch.manual_seed(seed + my_rank)
        torch.cuda.manual_seed_all(seed + my_rank)

        # NOTE: Without limiting threads number the CPU is the bottleneck
        torch.set_num_threads(3)

        self.buffer_size = buffer_size
        self.local_num_steps = local_num_steps
        self.total_steps = 0

        self.data_queue = queue.SimpleQueue()
        self.info_queue = queue.SimpleQueue()

        self.worker = _Worker(
            observation_shape=self.observation_space.shape,
            projection_dim=projection_dim,
            temperature=temperature,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            num_updates=num_updates,
            preproc_ratio=preproc_ratio,
            log_interval=log_interval,
            local_num_steps=local_num_steps,
            my_rank=my_rank,
            num_processes=num_processes,
            device_name=device_name,
            data_queue=self.data_queue,
            info_queue=self.info_queue,
        )
        self.worker.start()

        # Fill the data queue with warm-up steps
        action_repeat = 0
        obs = self.env.reset()
        for _ in range(self.buffer_size):
            self.data_queue.put_nowait((obs, None))

            if action_repeat == 0:
                action = self.env.action_space.sample()
                action_repeat = randint(1, 4)

            obs, _, done, _ = self.env.step(action)
            action_repeat -= 1

            if done:
                self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.total_steps += 1

        self.data_queue.put_nowait((obs, done))

        if (
            self.total_steps >= self.buffer_size
            and self.total_steps % self.local_num_steps == 0
        ):
            info["encoder"] = self.info_queue.get()

        return obs, rew, done, info

    def close(self):
        try:
            while True:
                self.data_queue.get_nowait()
        except queue.Empty:
            self.data_queue.put(None)

        self.env.close()
        self.worker.join()
