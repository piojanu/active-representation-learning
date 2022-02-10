import torch


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
