import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    def __init__(self, obs_dim, history_length, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_length = history_length
        
        self.conv1d = nn.Conv1d(
            in_channels=obs_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, obs_history):
        x = obs_history.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x


class AdaptationModule(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, encoded_history):
        x = self.fc1(encoded_history)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        latent = self.fc3(x)
        return latent


class RMAActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=[256, 256, 256], critic_hidden_dims=[256, 256, 256],
                 activation='elu', init_noise_std=1.0,
                 history_length=10, latent_dim=32, **kwargs):
        super().__init__()
        
        activation_fn = self._get_activation(activation)
        
        self.num_actor_obs = num_actor_obs
        self.latent_dim = latent_dim
        
        self.history_encoder = HistoryEncoder(num_actor_obs, history_length, hidden_dim=128)
        self.adaptation_module = AdaptationModule(128, latent_dim, hidden_dim=128)
        
        mlp_input_dim_a = num_actor_obs + latent_dim
        mlp_input_dim_c = num_critic_obs + latent_dim
        
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)
        
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"RMA Actor MLP: input_dim={mlp_input_dim_a}, {self.actor}")
        print(f"RMA Critic MLP: input_dim={mlp_input_dim_c}, {self.critic}")
        print(f"RMA latent_dim={latent_dim}, history_length={history_length}")
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        from torch.distributions import Normal
        Normal.set_default_validate_args = False
    
    @staticmethod
    def _get_activation(act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "selu":
            return nn.SELU()
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
        encoded = self.history_encoder(obs_history)
        latent = self.adaptation_module(encoded)
        return latent
    
    def update_distribution(self, observations, obs_history=None):
        if obs_history is not None:
            latent = self.compute_latent(obs_history)
            inputs = torch.cat([observations, latent], dim=-1)
        else:
            inputs = observations
        mean = self.actor(inputs)
        self.distribution = torch.distributions.Normal(mean, mean * 0. + self.std)
        return latent if obs_history is not None else None
    
    def act(self, observations, obs_history=None, **kwargs):
        self.update_distribution(observations, obs_history)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations, obs_history=None):
        if obs_history is not None:
            latent = self.compute_latent(obs_history)
            inputs = torch.cat([observations, latent], dim=-1)
        else:
            inputs = observations
        actions_mean = self.actor(inputs)
        return actions_mean
    
    def evaluate(self, critic_observations, obs_history=None, **kwargs):
        if obs_history is not None:
            latent = self.compute_latent(obs_history)
            inputs = torch.cat([critic_observations, latent], dim=-1)
        else:
            inputs = critic_observations
        value = self.critic(inputs)
        return value
