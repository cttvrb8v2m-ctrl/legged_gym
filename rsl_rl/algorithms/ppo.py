# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    PARAMETER_GROUP_BASE_LRS = {
        'actor': 1e-5,
        'critic': 1e-4,
        'rma': 1e-4,
        'std': 1e-4,
        'residual': 1e-4,
    }
    MIN_LR_FACTOR = 0.1
    MAX_LR_FACTOR = 1.0

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 anchor_loss_coef=1e-3,
                 film_reg_loss_coef=1e-5,
                 residual_reg_loss_coef=0.0,
                 actor_learning_rate=1e-5,
                 critic_learning_rate=1e-4,
                 rma_learning_rate=1e-4,
                 std_learning_rate=1e-4,
                 residual_learning_rate=1e-4,
                 anchor_loss_during_joint_training=False,
                 reference_anchor_enabled=False,
                 reference_anchor_loss_coef=0.02,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.lr_factor = 1.0
        self.parameter_group_base_lrs = {
            'actor': float(actor_learning_rate),
            'critic': float(critic_learning_rate),
            'rma': float(rma_learning_rate),
            'std': float(std_learning_rate),
            'residual': float(residual_learning_rate),
        }

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        
        # Parameter group learning rates for RMA stability
        self._setup_parameter_groups()
        
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.anchor_loss_coef = anchor_loss_coef
        self.film_reg_loss_coef = film_reg_loss_coef
        self.residual_reg_loss_coef = float(residual_reg_loss_coef)
        self.anchor_loss_during_joint_training = (
            anchor_loss_during_joint_training
        )
        self.reference_anchor_enabled = bool(reference_anchor_enabled)
        self.reference_anchor_loss_coef = float(
            reference_anchor_loss_coef
        )
        self.reference_actor = None
        self._reference_actor_initial_state = None
        
        # Gradient norms for logging
        self._actor_grad_norm = 0.0
        self._critic_grad_norm = 0.0
        self._rma_grad_norm = 0.0
        self._total_grad_norm = 0.0
        self._actor_frozen = None
        self._std_frozen = None
        self._rma_frozen = None
        self._action_stats = {}
        self._auxiliary_loss_stats = {}
    
    def _compute_grad_norms(self):
        """Compute gradient norms for different parameter groups"""
        # Actor grad norm
        actor_params = [p for n, p in self.actor_critic.named_parameters() if n.startswith('actor.')]
        actor_grads = [torch.norm(p.grad.detach()) for p in actor_params if p.grad is not None]
        self._actor_grad_norm = torch.norm(torch.stack(actor_grads)) if actor_grads else 0.0

        critic_params = [
            p for n, p in self.actor_critic.named_parameters()
            if n.startswith('critic.')
        ]
        critic_grads = [
            torch.norm(p.grad.detach())
            for p in critic_params if p.grad is not None
        ]
        self._critic_grad_norm = (
            torch.norm(torch.stack(critic_grads))
            if critic_grads else 0.0
        )
        
        # RMA grad norm
        rma_param_names = ['history_encoder', 'adaptation_module', 'actor_film', 'critic_film', 'latent_norm']
        rma_params = [p for n, p in self.actor_critic.named_parameters() if any(n.startswith(name) for name in rma_param_names)]
        rma_grads = [torch.norm(p.grad.detach()) for p in rma_params if p.grad is not None]
        self._rma_grad_norm = torch.norm(torch.stack(rma_grads)) if rma_grads else 0.0
    
    def _setup_parameter_groups(self):
        """Setup parameter groups with different learning rates for RMA stability"""
        actor_params = []
        critic_params = []
        rma_params = []
        std_params = []
        residual_params = []
        
        for name, param in self.actor_critic.named_parameters():
            if name.startswith('actor.'):
                actor_params.append(param)
            elif name.startswith('critic.'):
                critic_params.append(param)
            elif name.startswith('history_encoder') or name.startswith('adaptation_module') or name.startswith('actor_film') or name.startswith('critic_film') or name.startswith('latent_norm'):
                rma_params.append(param)
            elif name == 'std':
                std_params.append(param)
            elif name.startswith('rear_hip_residual.'):
                residual_params.append(param)
            else:
                # Default to rma params
                rma_params.append(param)
        
        # Parameter groups with different learning rates
        param_groups = []
        if actor_params:
            param_groups.append({'params': actor_params, 'name': 'actor'})
        if critic_params:
            param_groups.append({'params': critic_params, 'name': 'critic'})
        if rma_params:
            param_groups.append({'params': rma_params, 'name': 'rma'})
        if std_params:
            param_groups.append({'params': std_params, 'name': 'std'})
        if residual_params:
            param_groups.append({'params': residual_params, 'name': 'residual'})

        for param_group in param_groups:
            base_lr = self.parameter_group_base_lrs[param_group['name']]
            param_group['base_lr'] = base_lr
            # Retain initial_lr for compatibility with existing checkpoints/tools.
            param_group['initial_lr'] = base_lr
            param_group['lr'] = base_lr * self.lr_factor
        
        self.optimizer = optim.Adam(param_groups)
        print("\n📌 PPO Parameter Groups:")
        for pg in param_groups:
            print(
                f"  {pg['name']}: base_lr={pg['base_lr']}, "
                f"current_lr={pg['lr']}, params={len(pg['params'])}"
            )

    def apply_lr_factor(self, lr_factor=None):
        """Clamp the shared LR factor and rebind every group to its fixed base LR."""
        if lr_factor is not None:
            self.lr_factor = min(
                self.MAX_LR_FACTOR,
                max(self.MIN_LR_FACTOR, float(lr_factor)),
            )

        seen_groups = set()
        for param_group in self.optimizer.param_groups:
            group_name = param_group.get('name')
            if group_name not in self.parameter_group_base_lrs:
                raise RuntimeError(
                    f"Optimizer parameter group has no fixed base LR: {group_name}"
                )
            base_lr = self.parameter_group_base_lrs[group_name]
            param_group['base_lr'] = base_lr
            param_group['initial_lr'] = base_lr
            param_group['lr'] = base_lr * self.lr_factor
            seen_groups.add(group_name)

        expected_groups = {
            name for name in self.parameter_group_base_lrs
            if any(group.get('name') == name for group in self.optimizer.param_groups)
        }
        if seen_groups != expected_groups:
            raise RuntimeError(
                f"Optimizer parameter group mismatch: seen={seen_groups}, "
                f"expected={expected_groups}"
            )
        self.learning_rate = self.initial_learning_rate * self.lr_factor

    def get_learning_rate_stats(self):
        stats = {'lr_factor': self.lr_factor}
        for param_group in self.optimizer.param_groups:
            group_name = param_group['name']
            stats[group_name] = {
                'base_lr': param_group['base_lr'],
                'current_lr': param_group['lr'],
            }
        return stats

    def _capture_action_stats(self):
        """Capture rollout policy-mean and sampled-action magnitude statistics."""
        policy_mean_abs = self.storage.mu.abs().reshape(-1, self.storage.mu.shape[-1])
        sampled_action_abs = self.storage.actions.abs().reshape(
            -1, self.storage.actions.shape[-1]
        )
        self._action_stats = {
            'policy_mean_p95': torch.quantile(policy_mean_abs, 0.95).item(),
            'policy_mean_p99': torch.quantile(policy_mean_abs, 0.99).item(),
            'policy_mean_max': policy_mean_abs.max().item(),
            'sampled_action_p95': torch.quantile(sampled_action_abs, 0.95).item(),
            'sampled_action_p99': torch.quantile(sampled_action_abs, 0.99).item(),
            'sampled_action_max': sampled_action_abs.max().item(),
            'per_joint_sampled_action_p99': torch.quantile(
                sampled_action_abs, 0.99, dim=0
            ).detach().cpu().tolist(),
            'per_joint_sampled_action_max': sampled_action_abs.max(
                dim=0
            ).values.detach().cpu().tolist(),
        }

    def get_action_stats(self):
        return self._action_stats

    def get_auxiliary_loss_stats(self):
        return self._auxiliary_loss_stats

    def set_actor_frozen(self, frozen):
        """Freeze or unfreeze the actor and action standard deviation in-place."""
        return self.set_training_freeze(frozen, frozen)

    def set_training_freeze(
        self, actor_frozen, std_frozen, rma_frozen=False
    ):
        """Control Actor, exploration std, and RMA gradients independently."""
        if (
            self._actor_frozen == actor_frozen
            and self._std_frozen == std_frozen
            and self._rma_frozen == rma_frozen
        ):
            return False

        rma_prefixes = (
            "history_encoder",
            "adaptation_module",
            "actor_film",
            "critic_film",
            "latent_norm",
        )
        for name, param in self.actor_critic.named_parameters():
            if name.startswith('actor.'):
                param.requires_grad_(not actor_frozen)
                if actor_frozen:
                    param.grad = None
            elif name == 'std':
                param.requires_grad_(not std_frozen)
                if std_frozen:
                    param.grad = None
            elif name.startswith(rma_prefixes):
                param.requires_grad_(not rma_frozen)
                if rma_frozen:
                    param.grad = None

        self._actor_frozen = actor_frozen
        self._std_frozen = std_frozen
        self._rma_frozen = rma_frozen
        return True

    def rebuild_optimizer_for_migration(self):
        """Create a fresh optimizer after loading a PPO policy into RMA."""
        self.lr_factor = 1.0
        self.learning_rate = self.initial_learning_rate
        self._setup_parameter_groups()
        print("Fresh RMA migration optimizer created; checkpoint optimizer was not loaded")

    def initialize_reference_actor(self, checkpoint_path=None):
        """Freeze an optimizer-independent snapshot of the loaded actor."""
        self.reference_actor = copy.deepcopy(
            self.actor_critic.actor
        ).to(self.device)
        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device
            )
            actor_state = {
                name[len("actor."):]: value
                for name, value in checkpoint["model_state_dict"].items()
                if name.startswith("actor.")
            }
            load_result = self.reference_actor.load_state_dict(
                actor_state, strict=True
            )
            if load_result.missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    "Reference actor checkpoint mismatch: "
                    f"missing={load_result.missing_keys}, "
                    f"unexpected={load_result.unexpected_keys}"
                )
        self.reference_actor.eval()
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        optimizer_parameter_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if any(
            id(parameter) in optimizer_parameter_ids
            for parameter in self.reference_actor.parameters()
        ):
            raise RuntimeError("Reference actor was added to the optimizer")
        self._reference_actor_initial_state = {
            name: parameter.detach().clone()
            for name, parameter in self.reference_actor.named_parameters()
        }
        print(
            "Reference actor initialized: eval=True, "
            "requires_grad=False, optimizer_membership=False"
        )
        if checkpoint_path is not None:
            print(f"Reference actor checkpoint: {checkpoint_path}")

    def get_reference_actor_max_abs_diff(self):
        if self.reference_actor is None:
            return None
        return max(
            (
                parameter.detach()
                - self._reference_actor_initial_state[name]
            ).abs().max().item()
            for name, parameter in self.reference_actor.named_parameters()
        )

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, obs_history=None, anchor_mask=None):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        if obs_history is not None:
            self.transition.actions = self.actor_critic.act(obs, obs_history=obs_history).detach()
            self.transition.values = self.actor_critic.evaluate(critic_obs, obs_history=obs_history).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs).detach()
            self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        # Save obs_history for RMA
        self.transition.obs_history = obs_history
        self.transition.anchor_mask = anchor_mask
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        anchor_losses = []
        anchor_mask_ratios = []
        film_reg_losses = []
        residual_reg_losses = []
        action_mean_deltas = []
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, obs_history_batch, anchor_mask_batch in generator:


                self.actor_critic.act(obs_batch, obs_history=obs_history_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, obs_history=obs_history_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                anchor_loss = torch.zeros((), device=mu_batch.device)
                anchor_loss_coef = self.anchor_loss_coef
                if self.reference_anchor_enabled:
                    if self.reference_actor is None:
                        raise RuntimeError(
                            "Reference anchor enabled without reference actor"
                        )
                    selected = anchor_mask_batch.reshape(-1).bool()
                    anchor_mask_ratios.append(selected.float().mean())
                    if selected.any():
                        with torch.inference_mode():
                            mu_reference = self.reference_actor(obs_batch)
                        squared_delta = (
                            mu_batch - mu_reference.detach()
                        ).pow(2).mean(dim=-1)
                        anchor_loss = squared_delta[selected].mean()
                        action_mean_deltas.append(
                            (
                                mu_batch[selected]
                                - mu_reference[selected]
                            ).abs().detach().reshape(-1)
                        )
                    anchor_loss_coef = self.reference_anchor_loss_coef
                elif (
                    (
                        self._actor_frozen
                        or self.anchor_loss_during_joint_training
                    )
                    and hasattr(self.actor_critic, 'compute_base_action_mean')
                ):
                    mu_base = self.actor_critic.compute_base_action_mean(
                        obs_batch
                    ).detach()
                    action_mean_delta = (mu_batch - mu_base).abs()
                    anchor_loss = action_mean_delta.pow(2).mean()
                    action_mean_deltas.append(
                        action_mean_delta.detach().reshape(-1)
                    )

                film_reg_loss = torch.zeros((), device=mu_batch.device)
                if hasattr(self.actor_critic, 'get_film_regularization_loss'):
                    current_film_reg_loss = (
                        self.actor_critic.get_film_regularization_loss()
                    )
                    if current_film_reg_loss is not None:
                        film_reg_loss = current_film_reg_loss

                residual_reg_loss = torch.zeros((), device=mu_batch.device)
                if hasattr(
                    self.actor_critic, "get_residual_regularization_loss"
                ):
                    current_residual_reg_loss = (
                        self.actor_critic.get_residual_regularization_loss()
                    )
                    if current_residual_reg_loss is not None:
                        residual_reg_loss = current_residual_reg_loss

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            new_lr_factor = self.lr_factor / 1.5
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            new_lr_factor = self.lr_factor * 1.5
                        else:
                            new_lr_factor = self.lr_factor

                        self.apply_lr_factor(new_lr_factor)


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + anchor_loss_coef * anchor_loss
                    + self.film_reg_loss_coef * film_reg_loss
                    + self.residual_reg_loss_coef * residual_reg_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norms for logging
                self._compute_grad_norms()
                
                # Clip RMA gradients separately (stricter)
                rma_param_names = ['history_encoder', 'adaptation_module', 'actor_film', 'critic_film', 'latent_norm']
                rma_params = [p for n, p in self.actor_critic.named_parameters() if any(n.startswith(name) for name in rma_param_names)]
                if rma_params:
                    self._rma_grad_norm = nn.utils.clip_grad_norm_(rma_params, 0.5)
                
                # Clip total gradients
                self._total_grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                self.optimizer.step()

                anchor_losses.append(anchor_loss.detach())
                film_reg_losses.append(film_reg_loss.detach())
                residual_reg_losses.append(residual_reg_loss.detach())
                if hasattr(self.actor_critic, 'clear_auxiliary_tensors'):
                    self.actor_critic.clear_auxiliary_tensors()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        delta_values = (
            torch.cat(action_mean_deltas)
            if action_mean_deltas
            else torch.zeros(1, device=self.device)
        )
        self._auxiliary_loss_stats = {
            'anchor_loss': torch.stack(anchor_losses).mean().item()
            if anchor_losses else 0.0,
            'anchor_mask_ratio': torch.stack(
                anchor_mask_ratios
            ).mean().item() if anchor_mask_ratios else 0.0,
            'reference_actor_max_abs_diff': (
                self.get_reference_actor_max_abs_diff()
                if self.reference_actor is not None else 0.0
            ),
            'film_reg_loss': torch.stack(film_reg_losses).mean().item()
            if film_reg_losses else 0.0,
            'residual_reg_loss': torch.stack(
                residual_reg_losses
            ).mean().item() if residual_reg_losses else 0.0,
            'action_mean_delta_mean': delta_values.mean().item(),
            'action_mean_delta_p95': torch.quantile(
                delta_values, 0.95
            ).item(),
            'action_mean_delta_p99': torch.quantile(
                delta_values, 0.99
            ).item(),
            'action_mean_delta_max': delta_values.max().item(),
        }
        self._capture_action_stats()
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
