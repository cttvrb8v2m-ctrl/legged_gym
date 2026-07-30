import torch
import torch.nn as nn
from torch.distributions import Normal


class RMAActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, history_len=10, latent_dim=32,
                 actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                 activation='elu', init_noise_std=1.0,
                 enable_rear_hip_residual=False,
                 rear_hip_residual_independent=False,
                 rear_hip_residual_max_action=0.08,
                 rear_hip_residual_speed_threshold_obs=4.0,
                 rear_hip_residual_full_speed_obs=5.0,
                 **kwargs):
        super().__init__()
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.enable_rear_hip_residual = bool(enable_rear_hip_residual)
        self.rear_hip_residual_independent = bool(
            rear_hip_residual_independent
        )
        self.rear_hip_residual_max_action = float(
            rear_hip_residual_max_action
        )
        self.rear_hip_residual_speed_threshold_obs = float(
            rear_hip_residual_speed_threshold_obs
        )
        self.rear_hip_residual_full_speed_obs = float(
            rear_hip_residual_full_speed_obs
        )
        if self.enable_rear_hip_residual:
            if num_actions != 12 or num_actor_obs < 48:
                raise ValueError(
                    "Rear-hip residual requires the 12-action, 48-base-observation Go1 layout"
                )
            if (
                self.rear_hip_residual_full_speed_obs
                <= self.rear_hip_residual_speed_threshold_obs
            ):
                raise ValueError(
                    "Rear-hip residual full-speed threshold must exceed its activation threshold"
                )
        
        # RMA alpha schedule (not learnable)
        self.rma_alpha = 0.0
        
        activation_fn = self._get_activation(activation)
        
        self.history_encoder = nn.Sequential(
            nn.Conv1d(num_actor_obs, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.adaptation_module = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # LayerNorm for latent stability
        self.latent_norm = nn.LayerNorm(latent_dim)
        
        # FiLM: h = h * (1 + rma_alpha * gamma_raw) + rma_alpha * beta_raw
        # When rma_alpha=0: h = h * 1 + 0 = h (identity mapping)
        # gamma_raw and beta_raw initialized to 0, bounded by tanh
        self.actor_film = nn.Linear(latent_dim, actor_hidden_dims[0] * 2)
        nn.init.zeros_(self.actor_film.weight)
        nn.init.zeros_(self.actor_film.bias)
        
        self.critic_film = nn.Linear(latent_dim, critic_hidden_dims[0] * 2)
        nn.init.zeros_(self.critic_film.weight)
        nn.init.zeros_(self.critic_film.bias)
        
        # FiLM output scale (0.1 * tanh limits output to [-0.1, 0.1])
        self.film_scale = 0.1
        
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        if self.enable_rear_hip_residual:
            self.rear_hip_residual = nn.Sequential(
                nn.Linear(48, 32),
                nn.ELU(),
                nn.Linear(
                    32, 2 if self.rear_hip_residual_independent else 1
                ),
            )
            nn.init.zeros_(self.rear_hip_residual[-1].weight)
            nn.init.constant_(self.rear_hip_residual[-1].bias, -4.0)
        
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)
        
        print("\n" + "=" * 80)
        print("📌 RMAActorCritic loaded (Residual FiLM mode)")
        print("=" * 80)
        print(f"  num_actor_obs={num_actor_obs}, num_critic_obs={num_critic_obs}, num_actions={num_actions}")
        print(f"  history_len={history_len}, latent_dim={latent_dim}")
        print(f"  actor_input_dim={num_actor_obs}, critic_input_dim={num_critic_obs}")
        print(f"  actor_hidden_dims={actor_hidden_dims}, critic_hidden_dims={critic_hidden_dims}")
        print(f"  FiLM modulation on hidden layer 0: {actor_hidden_dims[0]}")
        print(f"  RMA alpha: {self.rma_alpha} (schedule controlled)")
        print(f"  FiLM equation: h = h * (1 + alpha * gamma_raw) + alpha * beta_raw")
        print(f"  When alpha=0: identity mapping (h -> h)")
        if self.enable_rear_hip_residual:
            print(
                "  Rear-hip residual: enabled, base actor unchanged, "
                f"max_action={self.rear_hip_residual_max_action:.3f}, "
                f"independent={self.rear_hip_residual_independent}"
            )
        print("=" * 80 + "\n")
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
        
        self._debug_printed = False
        self._first_action_printed = False
        
        # For stats collection
        self._last_latent = None
        self._last_gamma_raw = None
        self._last_beta_raw = None
        self._last_gamma_logits = None
        self._last_beta_logits = None
        self._current_actor_gamma_logits = None
        self._current_actor_beta_logits = None
        self._current_critic_gamma_logits = None
        self._current_critic_beta_logits = None
        self._current_rear_hip_residual_amplitudes = None
    
    @staticmethod
    def _get_activation(act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "lrelu":
            return nn.LeakyReLU()
        elif act_name == "tanh":
            return nn.Tanh()
        else:
            return nn.ELU()
    
    def reset(self, dones=None):
        pass
    
    def test(self):
        self.eval()
    
    def train(self, mode=True):
        super().train(mode)
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def compute_latent(self, obs_history):
        if obs_history is None:
            return None
        x = obs_history.permute(0, 2, 1)
        x = self.history_encoder(x)
        x = x.squeeze(-1)
        latent = self.adaptation_module(x)
        # LayerNorm for stability
        latent = self.latent_norm(latent)
        return latent
    
    def set_rma_alpha(self, alpha):
        """Set the RMA alpha schedule value"""
        self.rma_alpha = alpha
    
    def get_rma_stats(self):
        """Get RMA statistics for logging"""
        stats = {
            'rma_alpha': self.rma_alpha,
        }
        if self._last_latent is not None:
            stats['latent_mean'] = self._last_latent.mean().item()
            stats['latent_std'] = self._last_latent.std().item()
            stats['latent_max_abs'] = self._last_latent.abs().max().item()
        if self._last_gamma_raw is not None:
            stats['gamma_raw_mean'] = self._last_gamma_raw.mean().item()
            stats['gamma_raw_std'] = self._last_gamma_raw.std().item()
            stats['gamma_raw_max_abs'] = self._last_gamma_raw.abs().max().item()
            stats['gamma_saturation_ratio'] = (
                self._last_gamma_raw.abs() > 0.095
            ).float().mean().item()
        if self._last_beta_raw is not None:
            stats['beta_raw_mean'] = self._last_beta_raw.mean().item()
            stats['beta_raw_std'] = self._last_beta_raw.std().item()
            stats['beta_raw_max_abs'] = self._last_beta_raw.abs().max().item()
            stats['beta_saturation_ratio'] = (
                self._last_beta_raw.abs() > 0.095
            ).float().mean().item()
        if self._last_gamma_logits is not None:
            stats['gamma_logits_mean'] = self._last_gamma_logits.mean().item()
            stats['gamma_logits_std'] = self._last_gamma_logits.std().item()
            stats['gamma_logits_max_abs'] = self._last_gamma_logits.abs().max().item()
        if self._last_beta_logits is not None:
            stats['beta_logits_mean'] = self._last_beta_logits.mean().item()
            stats['beta_logits_std'] = self._last_beta_logits.std().item()
            stats['beta_logits_max_abs'] = self._last_beta_logits.abs().max().item()
        return stats

    def compute_base_action_mean(self, obs):
        """Return the original PPO actor mean without FiLM modulation."""
        return self.actor(obs)

    def compute_rear_hip_residual_amplitudes(self, obs):
        if not self.enable_rear_hip_residual:
            return torch.zeros(
                obs.shape[0], 2, dtype=obs.dtype, device=obs.device
            )

        command_x_obs = obs[:, 9]
        speed_gate = torch.clamp(
            (
                command_x_obs
                - self.rear_hip_residual_speed_threshold_obs
            )
            / (
                self.rear_hip_residual_full_speed_obs
                - self.rear_hip_residual_speed_threshold_obs
            ),
            min=0.0,
            max=1.0,
        )
        amplitudes = (
            self.rear_hip_residual_max_action
            * torch.sigmoid(self.rear_hip_residual(obs[:, :48]))
            * speed_gate.unsqueeze(-1)
        )
        if amplitudes.shape[-1] == 1:
            amplitudes = amplitudes.expand(-1, 2)
        return amplitudes

    def compute_rear_hip_residual_amplitude(self, obs):
        """Return the larger leg amplitude for scalar diagnostics."""
        return self.compute_rear_hip_residual_amplitudes(obs).max(dim=-1).values

    def _apply_rear_hip_residual(self, mean, obs, track_regularization=False):
        if not self.enable_rear_hip_residual:
            return mean

        outward_amplitudes = self.compute_rear_hip_residual_amplitudes(obs)
        if track_regularization:
            self._current_rear_hip_residual_amplitudes = outward_amplitudes
        residual = torch.zeros_like(mean)
        # Go1 action order: FL, FR, RL, RR; positive y is outward for RL.
        residual[:, 6] = outward_amplitudes[:, 0]
        residual[:, 9] = -outward_amplitudes[:, 1]
        return mean + residual

    def get_residual_regularization_loss(self):
        if self._current_rear_hip_residual_amplitudes is None:
            return None
        return self._current_rear_hip_residual_amplitudes.mean()

    def get_film_regularization_loss(self):
        """Regularize actor and critic FiLM logits before the bounded tanh."""
        losses = []
        if self._current_actor_gamma_logits is not None:
            losses.append(
                self._current_actor_gamma_logits.pow(2).mean()
                + self._current_actor_beta_logits.pow(2).mean()
            )
        if self._current_critic_gamma_logits is not None:
            losses.append(
                self._current_critic_gamma_logits.pow(2).mean()
                + self._current_critic_beta_logits.pow(2).mean()
            )
        if not losses:
            return None
        return torch.stack(losses).mean()

    def clear_auxiliary_tensors(self):
        """Release references to the current minibatch computation graph."""
        self._current_actor_gamma_logits = None
        self._current_actor_beta_logits = None
        self._current_critic_gamma_logits = None
        self._current_critic_beta_logits = None
        self._current_rear_hip_residual_amplitudes = None
    
    def check_stability(self):
        """Check for NaN/Inf or latent explosion"""
        if self._last_latent is not None:
            if torch.isnan(self._last_latent).any() or torch.isinf(self._last_latent).any():
                return False, "Latent contains NaN or Inf"
            if self._last_latent.abs().max().item() > 20:
                return False, f"Latent max_abs={self._last_latent.abs().max().item():.2f} > 20"
        if self._last_gamma_raw is not None:
            if torch.isnan(self._last_gamma_raw).any() or torch.isinf(self._last_gamma_raw).any():
                return False, "Gamma_raw contains NaN or Inf"
        if self._last_beta_raw is not None:
            if torch.isnan(self._last_beta_raw).any() or torch.isinf(self._last_beta_raw).any():
                return False, "Beta_raw contains NaN or Inf"
        return True, "OK"
    
    def update_distribution(self, obs, obs_history=None):
        latent = self.compute_latent(obs_history)
        self._last_latent = latent.detach() if latent is not None else None
        
        x = self.actor[0](obs)
        
        if latent is not None:
            film_params = self.actor_film(latent)
            gamma_logits, beta_logits = torch.chunk(film_params, 2, dim=-1)
            self._current_actor_gamma_logits = gamma_logits
            self._current_actor_beta_logits = beta_logits
            # Bounded FiLM output: gamma_raw, beta_raw ∈ [-0.1, 0.1]
            gamma_raw = self.film_scale * torch.tanh(gamma_logits)
            beta_raw = self.film_scale * torch.tanh(beta_logits)
            self._last_gamma_raw = gamma_raw.detach()
            self._last_beta_raw = beta_raw.detach()
            self._last_gamma_logits = gamma_logits.detach()
            self._last_beta_logits = beta_logits.detach()
            
            # Residual FiLM: h = h * (1 + alpha * gamma_raw) + alpha * beta_raw
            # When alpha=0: h = h * 1 + 0 = h (identity)
            x = x * (1 + self.rma_alpha * gamma_raw) + self.rma_alpha * beta_raw
        
        for i in range(1, len(self.actor)):
            x = self.actor[i](x)
        
        mean = self._apply_rear_hip_residual(
            x, obs, track_regularization=True
        )
        self.distribution = Normal(mean, mean * 0. + self.std)
        return latent
    
    def act(self, obs, obs_history=None, masks=None, hidden_states=None):
        if not self._debug_printed:
            print(f"📌 RMA act() - obs.shape={obs.shape}")
            if obs_history is not None:
                print(f"📌 RMA act() - obs_history.shape={obs_history.shape}")
            self._debug_printed = True
        
        latent = self.update_distribution(obs, obs_history)
        action = self.distribution.sample()
        
        if not self._first_action_printed:
            print(f"📌 RMA first action - mean={action.mean().item():.4f}, std={action.std().item():.4f}")
            print(f"📌 RMA action_mean={self.distribution.mean.mean().item():.4f}, action_std={self.distribution.stddev.mean().item():.4f}")
            if latent is not None:
                print(f"📌 RMA latent - mean={latent.mean().item():.4f}, std={latent.std().item():.4f}")
            self._first_action_printed = True
        
        return action
    
    def act_inference(self, obs, obs_history=None):
        latent = self.compute_latent(obs_history)
        
        x = self.actor[0](obs)
        
        if latent is not None:
            film_params = self.actor_film(latent)
            gamma_logits, beta_logits = torch.chunk(film_params, 2, dim=-1)
            # Bounded FiLM output
            gamma_raw = self.film_scale * torch.tanh(gamma_logits)
            beta_raw = self.film_scale * torch.tanh(beta_logits)
            # Residual FiLM
            x = x * (1 + self.rma_alpha * gamma_raw) + self.rma_alpha * beta_raw
        
        for i in range(1, len(self.actor)):
            x = self.actor[i](x)
        
        return self._apply_rear_hip_residual(x, obs)
    
    def evaluate(self, critic_obs, obs_history=None, masks=None, hidden_states=None):
        latent = self.compute_latent(obs_history)
        self._last_latent = latent.detach() if latent is not None else None
        
        x = self.critic[0](critic_obs)
        
        if latent is not None:
            film_params = self.critic_film(latent)
            gamma_logits, beta_logits = torch.chunk(film_params, 2, dim=-1)
            self._current_critic_gamma_logits = gamma_logits
            self._current_critic_beta_logits = beta_logits
            # Bounded FiLM output
            gamma_raw = self.film_scale * torch.tanh(gamma_logits)
            beta_raw = self.film_scale * torch.tanh(beta_logits)
            
            # Residual FiLM
            x = x * (1 + self.rma_alpha * gamma_raw) + self.rma_alpha * beta_raw
        
        for i in range(1, len(self.critic)):
            x = self.critic[i](x)
        
        return x
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_hidden_states(self):
        return None
