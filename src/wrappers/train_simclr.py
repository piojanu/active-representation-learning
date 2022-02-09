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
        projection_dim,
        temperature,
    ):
        super().__init__(env)

        self.batch_size = batch_size
        self.mixing_coef = mixing_coef

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
        # WA: Call `.contiguous()` to prevent "[...] grad and param do not
        #     obey the gradient layout contract [...]".
        x_i = self.scripted_transforms(batch).contiguous()
        x_j = self.scripted_transforms(batch).contiguous()

        return x_i, x_j

    def compute_loss(self, x_i, x_j):
        # TODO: You might want to concat. the positive pair into a single batch
        #       for better performance. Same with the transformations.
        _, _, z_i, z_j = self.model(x_i, x_j)

        return self.criterion(z_i, z_j)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.buffer[self.ptr].copy_(torch.from_numpy(obs))
        self.ptr += 1

        if self.ptr == self.batch_size:
            x_i, x_j = self.transform_batch(self.buffer)

            self.optimizer.zero_grad()
            loss = self.compute_loss(x_i, x_j)
            loss.backward()
            self.optimizer.step()

            info["LossEncoder"] = loss.item()
            self.ptr = 0

            with torch.no_grad():
                rew = (1 - self.mixing_coef) * rew + self.mixing_coef * (
                    5.0 - loss.item()
                )
        else:
            rew = (1 - self.mixing_coef) * rew

        return obs, rew, done, info

    @staticmethod
    def log_info(logger, info):
        if "LossEncoder" in info.keys():
            logger.store(LossEncoder=info["LossEncoder"])

    @staticmethod
    def compute_stats(logger):
        logger.log_tabular("LossEncoder")
