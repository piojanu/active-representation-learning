import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(
        self,
        num_steps,
        num_processes,
        obs_shape,
        action_space,
        recurrent_hidden_state_size,
    ):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size
        )
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps, num_processes, 1)
        self.advantages = torch.zeros(num_steps, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = torch.zeros(num_steps, num_processes, 1, dtype=torch.long)
        else:
            self.actions = torch.zeros(num_steps, num_processes, action_space.shape[0])
        self.non_terminal_masks = torch.ones(num_steps + 1, num_processes, 1)
        # Masks that indicate whether it's a time limit end state (1.0) or
        # a true terminal state (0.0).
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.non_terminal_masks = self.non_terminal_masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(
        self,
        obs,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        non_terminal_masks,
        bad_masks,
    ):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.non_terminal_masks[self.step + 1].copy_(non_terminal_masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step += 1

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.non_terminal_masks[0].copy_(self.non_terminal_masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

        self.step = 0

    def compute_returns(
        self,
        last_value,
        gae_lambda,
        gamma,
        bootstrap_value_at_time_limit,
        force_non_episodic,
    ):
        """Compute the lambda-return (TD(lambda) estimate) and GAE advantage.

        The TD(lambda) estimator has also two special cases:
            - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
            - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375
        """
        self.value_preds[-1] = last_value

        if force_non_episodic:
            masks = torch.ones_like(self.non_terminal_masks)
        elif bootstrap_value_at_time_limit:
            masks = torch.logical_or(self.non_terminal_masks, self.bad_masks).float()
        else:
            masks = self.non_terminal_masks

        deltas = (
            self.rewards
            + gamma * self.value_preds[1:] * masks[1:]
            - self.value_preds[:-1]
        )

        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            gae = deltas[step] + gamma * gae_lambda * masks[step + 1] * gae
            self.advantages[step] = gae

        self.returns[:] = self.advantages[:] + self.value_preds[:-1]

        # Centre and normalize the advantages to lower the variance of
        # the policy gradient estimator.
        # See: http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/hw2_final.pdf
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-5
        )

    def feed_forward_generator(self, mini_batch_size):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns.view(-1, 1)[indices]
            masks_batch = self.non_terminal_masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = self.advantages.view(-1, 1)[indices]

            yield (
                obs_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    def recurrent_generator(self, mini_batch_size):
        num_steps, num_processes = self.rewards.size()[0:2]
        assert mini_batch_size % num_steps == 0, (
            "Recurrent PPO requires that the mini-batch size ({})"
            "is multiple of the steps num ({})".format(mini_batch_size, num_steps)
        )
        num_envs_per_batch = mini_batch_size // num_steps
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind]
                )
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:, ind])
                masks_batch.append(self.non_terminal_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(self.advantages[:, ind])

            T, N = num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield (
                obs_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )
