import numpy as np
import torch
import torch.nn as nn
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

from nets import ConvNet26x26Encoder


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size, recurrent):
        super(ActorCritic, self).__init__()
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape, hidden_size, recurrent)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape, hidden_size, recurrent)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, old_log_probs):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, obs_shape, hidden_size=512, recurrent=False):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.encoder = ConvNet26x26Encoder(n_features=hidden_size)
        self.critic_linear = init(
            nn.Linear(hidden_size, 1),
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
        )

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.encoder(inputs / 255.0)
        x = torch.tanh(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, obs_shape, hidden_size=64, recurrent=False):
        super(MLPBase, self).__init__(recurrent, obs_shape[0], hidden_size)

        if recurrent:
            num_inputs = hidden_size
        else:
            num_inputs = obs_shape[0]

        # trunk-ignore(flake8/E731)
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class DummyActorCritic:
    def __init__(self, action_space):
        assert action_space.__class__.__name__ == "Discrete"
        self.num_actions = action_space.n

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError()

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        raise NotImplementedError()

    def get_value(self, inputs, rnn_hxs, masks):
        del rnn_hxs
        del masks
        return torch.zeros(inputs.shape[0], 1)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, old_log_probs):
        return (
            self.get_value(inputs, rnn_hxs, masks),
            torch.zeros_like(old_log_probs),
            0.0,
            rnn_hxs,
        )


class ConstantActorCritic(DummyActorCritic):
    def __init__(self, action_space, constant_action):
        super().__init__(action_space)
        self.constant_action = constant_action

        self.action = None

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        del deterministic
        num_processes = inputs.shape[0]
        if self.action is None:
            if self.constant_action is None:
                self.action = torch.randint(0, self.num_actions, (num_processes, 1))
            else:
                self.action = torch.ones(num_processes, 1) * self.constant_action

        return (
            self.get_value(inputs, rnn_hxs, masks),
            self.action,
            torch.zeros(num_processes, 1),
            rnn_hxs,
        )


class RandomActorCritic(DummyActorCritic):
    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        del deterministic
        num_processes = inputs.shape[0]
        return (
            self.get_value(inputs, rnn_hxs, masks),
            torch.randint(self.num_actions, (num_processes, 1)),
            torch.zeros(num_processes, 1),
            rnn_hxs,
        )


class RandomRepeatActorCritic(DummyActorCritic):
    def __init__(self, action_space, min_repeat, max_repeat):
        super().__init__(action_space)
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat

    @property
    def is_recurrent(self):
        return True

    @property
    def recurrent_hidden_state_size(self):
        return 3

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        del deterministic
        action, counter, repeat = rnn_hxs[:, 0:1], rnn_hxs[:, 1:2], rnn_hxs[:, 2:3]
        mask = (counter != repeat).float()

        action = mask * action + (1 - mask) * torch.randint_like(
            action, self.num_actions
        )
        counter = mask * (counter + torch.ones_like(counter))
        repeat = mask * repeat + (1 - mask) * torch.randint_like(
            repeat, self.min_repeat, self.max_repeat
        )

        return (
            self.get_value(inputs, rnn_hxs, masks),
            action,
            torch.zeros(inputs.shape[0], 1),
            torch.concat((action, counter, repeat), dim=-1),
        )
