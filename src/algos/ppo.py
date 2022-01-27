import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPO():
    def __init__(self,
                 actor_critic,
                 # Loss params
                 clip_ratio_pi,
                 clip_ratio_vf,
                 entropy_coef,
                 value_coef,
                 # Training params
                 learning_rate,
                 num_epochs,
                 max_grad_norm,
                 max_kl,
                 mini_batch_size):

        self.actor_critic = actor_critic

        self.clip_ratio_pi = clip_ratio_pi
        self.clip_ratio_vf = clip_ratio_vf
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.max_kl = max_kl
        self.mini_batch_size = mini_batch_size

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    def update(self, rollouts):
        num_updates = (np.prod(rollouts.rewards.size()[0:2])
                       // self.mini_batch_size)
        for epoch in range(self.num_epochs):
            value_loss_epoch = 0
            policy_loss_epoch = 0
            dist_entropy_epoch = 0
            approx_kl_epoch = 0

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    self.mini_batch_size)
            else:
                data_generator = rollouts.feed_forward_generator(
                    self.mini_batch_size)

            for sample in data_generator:
                (obs_batch, recurrent_hidden_states_batch, actions_batch,
                 old_value_pred_batch, return_batch, masks_batch,
                 old_log_probs_batch, adv_targ) = sample

                values, log_probs, dist_entropy, _ = \
                    self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch, old_log_probs_batch)

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs_batch
                    approx_kl = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio_pi,
                                    1.0 + self.clip_ratio_pi) * adv_targ
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.clip_ratio_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = (
                        old_value_pred_batch
                        +  (values - old_value_pred_batch).clamp(
                            -self.clip_ratio_vf, self.clip_ratio_vf))
                value_loss = F.mse_loss(return_batch, values_pred)

                self.optimizer.zero_grad()
                (policy_loss
                 + self.value_coef * value_loss
                 - self.entropy_coef * dist_entropy).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                approx_kl_epoch += approx_kl.item()
            
            if approx_kl_epoch / num_updates > self.max_kl:
                break
            
        value_loss_epoch /= num_updates
        policy_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        approx_kl_epoch /= num_updates

        return (value_loss_epoch, policy_loss_epoch, dist_entropy_epoch,
                approx_kl_epoch, epoch + 1)
