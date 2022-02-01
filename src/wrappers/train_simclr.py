import gym
import torch
import torchvision

from simclr.modules import NT_Xent
from simclr.simclr import SimCLR

from models.resnet import ResNetEncoder

class TrainSimCLR(gym.Wrapper):
    """Collects data and trains SimCLR encoder, where loss drop is reward."""
    def __init__(self, env, batch_size, learning_rate, num_residual_blocks,
                 projection_dim, temperature):
        super().__init__(env)

        self.batch_size = batch_size

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device = torch.device('cuda:1')
        else:
            self.device = torch.device('cpu')
        self.buffer = torch.zeros(self.batch_size,
                                  *self.observation_space.shape).to(self.device)
        self.ptr = 0

        # Create SimCLR transformation
        transforms = torch.nn.Sequential(
            # Normalize observations into [0, 1] range as required for floats
            torchvision.transforms.Normalize([0., 0., 0.], [255., 255., 255.]),
            torchvision.transforms.RandomResizedCrop(
                size=self.observation_space.shape[1:]),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply(torch.nn.ModuleList([
                torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ]), p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2)
        )
        self.scripted_transforms = torch.jit.script(transforms)

        # Create SimCLR encoder
        self.encoder = ResNetEncoder(layers=[2, 2])
        n_features = self.encoder(
            torch.randn(1, *env.observation_space.shape)).shape[1]
        self.encoder.to(self.device)


        # Create SimCLR projector
        self.model = SimCLR(self.encoder, projection_dim, n_features)
        self.model.to(self.device)

        # Create SimCLR criterion
        self.criterion = NT_Xent(batch_size, temperature, world_size=1)

        # Create Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)

    def compute_loss(self, batch):
        x_i = self.scripted_transforms(batch)
        x_j = self.scripted_transforms(batch)
        
        _, _, z_i, z_j = self.model(x_i, x_j)

        return self.criterion(z_i, z_j)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.buffer[self.ptr].copy_(torch.from_numpy(obs))
        self.ptr += 1

        if self.ptr == self.batch_size:
            self.optimizer.zero_grad()
            loss = self.compute_loss(self.buffer)
            loss.backward()
            self.optimizer.step()

            info['LossEncoder'] = loss.item()
            self.ptr = 0

        return obs, rew, done, info