import gym
import torch
import torchvision
from simclr.modules import NT_Xent
from simclr.simclr import SimCLR

from models.resnet import ResNetEncoder
from utils.logx import InfoLogger


class TrainSimCLR(gym.Wrapper, InfoLogger):
    """Collects data and trains SimCLR encoder, where loss drop is reward."""

    def __init__(
        self,
        env,
        device_name,
        batch_size,
        learning_rate,
        mixing_coef,
        num_updates,
        projection_dim,
        temperature,
    ):
        super().__init__(env)

        self.batch_size = batch_size
        self.mixing_coef = mixing_coef
        self.num_updates = num_updates

        self.device = torch.device(device_name)
        self.buffer = torch.zeros(self.batch_size, *self.observation_space.shape).to(
            self.device
        )
        self.ptr = 0

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
        self.criterion = NT_Xent(batch_size, temperature, world_size=1)

        # Create Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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

        if self.ptr < self.batch_size:
            self.buffer[self.ptr].copy_(torch.from_numpy(obs))
            self.ptr += 1

            mix_rew = 0.0
        else:
            # Collect
            ptr = torch.randint(self.batch_size, size=(1,))
            self.buffer[ptr].copy_(torch.from_numpy(obs))

            # Train
            num_updates = (
                self.num_updates
                if self.num_updates >= 1
                else int(torch.rand(1) < self.num_updates)
            )
            for _ in range(0, num_updates):
                # TODO: Should we use the loss after the update as the reward signal?
                x_i, x_j = self.transform_batch(self.buffer)
                loss = self.compute_loss(x_i, x_j)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if num_updates > 0:
                with torch.no_grad():
                    mix_rew = self.mixing_coef * (5.0 - loss.item())

                info["LossEncoder"] = loss.item()
            else:
                mix_rew = 0.0

        mix_rew += (1 - self.mixing_coef) * rew

        return obs, mix_rew, done, info

    @staticmethod
    def log_info(logger, info):
        if "LossEncoder" in info.keys():
            logger.store(LossEncoder=info["LossEncoder"])

    @staticmethod
    def compute_stats(logger):
        logger.log_tabular("LossEncoder")
