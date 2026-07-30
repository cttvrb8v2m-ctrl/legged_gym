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

import time
import os
import inspect
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        self.use_rma = self.cfg.get("use_rma", False)
        print(f"OnPolicyRunner: use_rma={self.use_rma}")
        self.freeze_actor_iters = self.cfg.get("freeze_actor_iters", 500)
        self.rma_warmup_iters = self.cfg.get("rma_warmup_iters", 20)
        self.rma_alpha_ramp_iters = self.cfg.get("rma_alpha_ramp_iters", 500)
        self.rma_max_alpha = self.cfg.get("rma_max_alpha", 0.2)
        self.joint_stair_training = self.cfg.get(
            "joint_stair_training", False
        )
        self.joint_training_alpha = self.cfg.get(
            "joint_training_alpha", 0.03
        )
        self.freeze_std_during_joint_training = self.cfg.get(
            "freeze_std_during_joint_training", True
        )
        self.freeze_rma_during_joint_training = self.cfg.get(
            "freeze_rma_during_joint_training", False
        )
        self.freeze_actor_during_joint_training = self.cfg.get(
            "freeze_actor_during_joint_training", False
        )
        self.reference_anchor_enabled = self.cfg.get(
            "reference_anchor_enabled", False
        )
        self.reference_anchor_max_stair_height = float(
            self.cfg.get("reference_anchor_max_stair_height", 0.145)
        )
        self.reference_actor_checkpoint = self.cfg.get(
            "reference_actor_checkpoint"
        )
        self.restore_terrain_levels = self.cfg.get(
            "restore_terrain_levels", True
        )
        self.rma_stats_interval = self.cfg.get("rma_stats_interval", 20)
        self.rma_checkpoint_validated = not self.use_rma
        
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        
        if self.use_rma:
            from legged_gym.algorithms.rma_actor_critic import RMAActorCritic
            actor_critic_class = RMAActorCritic
            self.policy_cfg['history_len'] = getattr(self.env, 'history_length', 10)
            self.policy_cfg['latent_dim'] = getattr(self.env.cfg.rma, 'latent_dim', 32)
        else:
            actor_critic_class = eval(self.cfg["policy_class_name"])
        
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        if self.use_rma and not self.cfg.get("resume", False):
            migration_checkpoint = self.cfg.get("rma_migration_checkpoint")
            if not migration_checkpoint:
                raise RuntimeError(
                    "RMA migration requires runner.rma_migration_checkpoint"
                )
            self._load_and_validate_ppo_migration(
                actor_critic, migration_checkpoint, num_critic_obs
            )
            self.rma_checkpoint_validated = True

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        if self.use_rma and not self.cfg.get("resume", False):
            print("Fresh RMA migration optimizer created; old PPO optimizer not loaded")
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._joint_actor_reference = None
        self._joint_std_reference = None

        _, _ = self.env.reset()

    def _reference_anchor_mask(self):
        if not self.reference_anchor_enabled:
            return None
        if not (
            hasattr(self.env, "_stair_env_mask")
            and hasattr(self.env, "_stair_step_heights")
        ):
            raise RuntimeError(
                "Reference anchor requires stair environment metadata"
            )
        return (
            self.env._stair_env_mask()
            & (
                self.env._stair_step_heights()
                <= self.reference_anchor_max_stair_height + 1e-8
            )
        ).detach()

    def _load_and_validate_ppo_migration(
        self, rma_actor_critic, checkpoint_path, num_critic_obs
    ):
        checkpoint_path = os.path.realpath(checkpoint_path)
        rma_source = os.path.realpath(inspect.getfile(type(rma_actor_critic)))
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"RMAActorCritic source: {rma_source}")

        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"Migration checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" not in checkpoint:
            raise RuntimeError("Checkpoint has no model_state_dict")
        checkpoint_state = checkpoint["model_state_dict"]
        current_state = rma_actor_critic.state_dict()

        actor_keys = [key for key in current_state if key.startswith("actor.")]
        critic_keys = [key for key in current_state if key.startswith("critic.")]
        actor_matches = [
            key for key in actor_keys
            if key in checkpoint_state
            and checkpoint_state[key].shape == current_state[key].shape
        ]
        critic_matches = [
            key for key in critic_keys
            if key in checkpoint_state
            and checkpoint_state[key].shape == current_state[key].shape
        ]
        actor_match_rate = len(actor_matches) / len(actor_keys)
        critic_match_rate = len(critic_matches) / len(critic_keys)
        print(
            f"Actor checkpoint loaded: {len(actor_matches)}/{len(actor_keys)} "
            f"({actor_match_rate * 100:.1f}%)"
        )
        print(
            f"Critic checkpoint loaded: {len(critic_matches)}/{len(critic_keys)} "
            f"({critic_match_rate * 100:.1f}%)"
        )
        if actor_match_rate != 1.0 or critic_match_rate != 1.0:
            raise RuntimeError("Actor/Critic checkpoint parameter match rate is not 100%")

        if "std" not in checkpoint_state:
            raise RuntimeError("Checkpoint std parameter is missing")
        checkpoint_std = checkpoint_state["std"]
        if checkpoint_std.shape != current_state["std"].shape:
            raise RuntimeError("Checkpoint std shape mismatch")
        std_min = checkpoint_std.min().item()
        std_max = checkpoint_std.max().item()
        print(f"std loaded: 100%, min={std_min:.6f}, max={std_max:.6f}")
        if torch.allclose(checkpoint_std, torch.ones_like(checkpoint_std)):
            raise RuntimeError("Checkpoint std is all ones; refusing RMA migration")

        ppo_policy = ActorCritic(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            actor_hidden_dims=self.policy_cfg["actor_hidden_dims"],
            critic_hidden_dims=self.policy_cfg["critic_hidden_dims"],
            activation=self.policy_cfg["activation"],
            init_noise_std=self.policy_cfg["init_noise_std"],
        ).to(self.device)
        ppo_state = {
            key: value for key, value in checkpoint_state.items()
            if key.startswith("actor.") or key.startswith("critic.") or key == "std"
        }
        ppo_policy.load_state_dict(ppo_state, strict=True)
        ppo_policy.eval()

        load_result = rma_actor_critic.load_state_dict(ppo_state, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected migration keys: {load_result.unexpected_keys}"
            )

        actor_param_max_abs_diff = max(
            (
                rma_actor_critic.state_dict()[key] - ppo_policy.state_dict()[key]
            ).abs().max().item()
            for key in actor_keys
        )
        print(
            "actor_param_max_abs_diff="
            f"{actor_param_max_abs_diff:.10e}"
        )
        if actor_param_max_abs_diff >= 1e-7:
            raise RuntimeError("Loaded RMA actor differs from PPO actor")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(12345)
        validation_obs = torch.randn(
            64, self.env.num_obs, generator=generator
        ).to(self.device)
        validation_history = torch.randn(
            64,
            getattr(self.env, "history_length", 10),
            self.env.num_obs,
            generator=generator,
        ).to(self.device)
        rma_actor_critic.set_rma_alpha(0.0)
        rma_actor_critic.eval()
        with torch.inference_mode():
            ppo_action_mean = ppo_policy.act_inference(validation_obs)
            rma_action_mean = rma_actor_critic.act_inference(
                validation_obs, obs_history=validation_history
            )
        action_max_abs_diff = (
            ppo_action_mean - rma_action_mean
        ).abs().max().item()
        print(f"alpha=0 action max_abs_diff={action_max_abs_diff:.10e}")
        if action_max_abs_diff >= 1e-5:
            raise RuntimeError("RMA alpha=0 action is not equivalent to PPO")

        rma_actor_critic.train()
        print("RMA migration validation PASSED")
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.use_rma and not self.rma_checkpoint_validated:
            raise RuntimeError(
                "RMA checkpoint was not loaded and validated; freeze/training is forbidden"
            )
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        
        # Print obs_history shape for RMA debugging
        if self.use_rma:
            obs_history = self.env.obs_history.to(self.device)
            print(f"📌 RMA obs_history shape: {obs_history.shape}")
            print(f"📌 RMA obs shape: {obs.shape}")
            print(f"📌 Expected obs_history shape: [num_envs={self.env.num_envs}, history_len={getattr(self.env, 'history_length', 10)}, num_actor_obs={self.env.num_obs}]")
            if self.joint_stair_training:
                print("📌 Joint stair training: Actor/Critic/RMA trainable")
                print(
                    "📌 Joint stair training: std "
                    f"{'frozen' if self.freeze_std_during_joint_training else 'trainable'}"
                )
                print(
                    "📌 Joint stair training: RMA "
                    f"{'frozen' if self.freeze_rma_during_joint_training else 'trainable'}"
                )
                print(
                    "📌 Joint stair training fixed alpha: "
                    f"{self.joint_training_alpha:.6f}"
                )
            else:
                # RMA alpha ramp config (conservative)
                print(f"📌 RMA alpha ramp config: warmup={self.rma_warmup_iters} iters, ramp={self.rma_alpha_ramp_iters} iters, max_alpha={self.rma_max_alpha}")
                print(f"📌 RMA alpha will be 0 for first {self.rma_warmup_iters} iterations")
                print(f"📌 RMA alpha will ramp from 0 to {self.rma_max_alpha} over {self.rma_alpha_ramp_iters} iterations")
                print(f"📌 RMA actor/std frozen for first {self.freeze_actor_iters} iterations")
            print(f"📌 Current iteration: {self.current_learning_iteration}")
            print(f"📌 Current rma_alpha: {self.alg.actor_critic.rma_alpha:.6f}")

            if self.joint_stair_training:
                self._joint_actor_reference = {
                    name: parameter.detach().clone()
                    for name, parameter in self.alg.actor_critic.named_parameters()
                    if name.startswith("actor.")
                }
                self._joint_std_reference = (
                    self.alg.actor_critic.std.detach().clone()
                )

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            # RMA alpha ramp (conservative schedule)
            if self.use_rma:
                if self.joint_stair_training:
                    actor_frozen = self.freeze_actor_during_joint_training
                    std_frozen = self.freeze_std_during_joint_training
                    rma_frozen = self.freeze_rma_during_joint_training
                    rma_alpha = self.joint_training_alpha
                else:
                    actor_frozen = it < self.freeze_actor_iters
                    std_frozen = actor_frozen
                    rma_frozen = False
                    rma_alpha = self._compute_rma_alpha(it)

                if self.alg.set_training_freeze(
                    actor_frozen, std_frozen, rma_frozen
                ):
                    group_lrs = {
                        param_group.get('name'): param_group['lr']
                        for param_group in self.alg.optimizer.param_groups
                    }
                    print("\nRMA stage switch")
                    print(f"Iteration {it}:")
                    print(f"Actor {'frozen' if actor_frozen else 'unfrozen'}")
                    print(f"std {'frozen' if std_frozen else 'unfrozen'}")
                    print(f"RMA {'frozen' if rma_frozen else 'unfrozen'}")
                    print(f"Actor lr = {group_lrs['actor']:.6e}")
                    print(f"RMA lr = {group_lrs['rma']:.6e}\n")

                self.alg.actor_critic.set_rma_alpha(rma_alpha)
            
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if self.use_rma:
                        obs_history = self.env.obs_history.to(self.device)
                        actions = self.alg.act(
                            obs,
                            critic_obs,
                            obs_history=obs_history,
                            anchor_mask=self._reference_anchor_mask(),
                        )
                    else:
                        actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            self.current_learning_iteration = it + 1
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if self.current_learning_iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            
            # Verify that adaptive KL preserves the parameter-group LR ratios
            if self.use_rma and it % 50 == 0:
                self._print_learning_rates()
            
            # RMA stats logging at the configured interval.
            if self.use_rma and it % self.rma_stats_interval == 0:
                rma_stats = self.alg.actor_critic.get_rma_stats()
                print("\n" + "=" * 60)
                print(f"📊 RMA Stats - Iteration {it}")
                print("=" * 60)
                print(f"  rma_alpha: {rma_stats.get('rma_alpha', 0.0):.4f}")
                print(f"  latent_mean: {rma_stats.get('latent_mean', 'N/A'):.4f}")
                print(f"  latent_std: {rma_stats.get('latent_std', 'N/A'):.4f}")
                print(f"  latent_max_abs: {rma_stats.get('latent_max_abs', 'N/A'):.4f}")
                print(f"  gamma_raw_mean: {rma_stats.get('gamma_raw_mean', 'N/A'):.4f}")
                print(f"  gamma_raw_std: {rma_stats.get('gamma_raw_std', 'N/A'):.4f}")
                print(f"  gamma_raw_max_abs: {rma_stats.get('gamma_raw_max_abs', 'N/A'):.4f}")
                print(f"  gamma_saturation_ratio: {rma_stats.get('gamma_saturation_ratio', 'N/A'):.6f}")
                print(f"  gamma_logits_mean: {rma_stats.get('gamma_logits_mean', 'N/A'):.4f}")
                print(f"  gamma_logits_std: {rma_stats.get('gamma_logits_std', 'N/A'):.4f}")
                print(f"  gamma_logits_max_abs: {rma_stats.get('gamma_logits_max_abs', 'N/A'):.4f}")
                print(f"  beta_raw_mean: {rma_stats.get('beta_raw_mean', 'N/A'):.4f}")
                print(f"  beta_raw_std: {rma_stats.get('beta_raw_std', 'N/A'):.4f}")
                print(f"  beta_raw_max_abs: {rma_stats.get('beta_raw_max_abs', 'N/A'):.4f}")
                print(f"  beta_saturation_ratio: {rma_stats.get('beta_saturation_ratio', 'N/A'):.6f}")
                print(f"  beta_logits_mean: {rma_stats.get('beta_logits_mean', 'N/A'):.4f}")
                print(f"  beta_logits_std: {rma_stats.get('beta_logits_std', 'N/A'):.4f}")
                print(f"  beta_logits_max_abs: {rma_stats.get('beta_logits_max_abs', 'N/A'):.4f}")
                auxiliary_stats = self.alg.get_auxiliary_loss_stats()
                print(f"  anchor_loss: {auxiliary_stats.get('anchor_loss', 'N/A'):.8f}")
                print(
                    "  anchor_mask_ratio: "
                    f"{auxiliary_stats.get('anchor_mask_ratio', 0.0):.6f}"
                )
                print(
                    "  reference_actor_max_abs_diff: "
                    f"{auxiliary_stats.get('reference_actor_max_abs_diff', 0.0):.10e}"
                )
                print(f"  film_reg_loss: {auxiliary_stats.get('film_reg_loss', 'N/A'):.8f}")
                print(f"  action_mean_delta_mean: {auxiliary_stats.get('action_mean_delta_mean', 'N/A'):.6f}")
                print(f"  action_mean_delta_p95: {auxiliary_stats.get('action_mean_delta_p95', 'N/A'):.6f}")
                print(f"  action_mean_delta_p99: {auxiliary_stats.get('action_mean_delta_p99', 'N/A'):.6f}")
                print(f"  action_mean_delta_max: {auxiliary_stats.get('action_mean_delta_max', 'N/A'):.6f}")
                # Full-rollout action stats, separated by distribution mean/sample.
                action_stats = self.alg.get_action_stats()
                print(f"  policy_mean_p95: {action_stats.get('policy_mean_p95', 'N/A'):.4f}")
                print(f"  policy_mean_p99: {action_stats.get('policy_mean_p99', 'N/A'):.4f}")
                print(f"  policy_mean_max: {action_stats.get('policy_mean_max', 'N/A'):.4f}")
                print(f"  sampled_action_p95: {action_stats.get('sampled_action_p95', 'N/A'):.4f}")
                print(f"  sampled_action_p99: {action_stats.get('sampled_action_p99', 'N/A'):.4f}")
                print(f"  sampled_action_max: {action_stats.get('sampled_action_max', 'N/A'):.4f}")
                print(
                    "  per_joint_sampled_action_p99: "
                    f"{action_stats.get('per_joint_sampled_action_p99', 'N/A')}"
                )
                print(
                    "  per_joint_sampled_action_max: "
                    f"{action_stats.get('per_joint_sampled_action_max', 'N/A')}"
                )
                print(f"  raw_action_max_abs: {getattr(self.env, 'raw_action_max_abs', 0.0):.4f}")
                print(f"  action_clip_ratio: {getattr(self.env, 'action_clip_ratio', 0.0):.6f}")
                print(f"  joint_target_clamp_ratio: {getattr(self.env, 'joint_target_clamp_ratio', 0.0):.6f}")
                # Gradient norms
                print(f"  actor_grad_norm: {getattr(self.alg, '_actor_grad_norm', 'N/A')}")
                print(f"  critic_grad_norm: {getattr(self.alg, '_critic_grad_norm', 'N/A')}")
                print(f"  rma_grad_norm: {getattr(self.alg, '_rma_grad_norm', 'N/A')}")
                print(f"  total_grad_norm: {getattr(self.alg, '_total_grad_norm', 'N/A')}")
                if self._joint_actor_reference is not None:
                    actor_parameter_max_abs_delta = max(
                        (
                            parameter.detach()
                            - self._joint_actor_reference[name]
                        ).abs().max().item()
                        for name, parameter
                        in self.alg.actor_critic.named_parameters()
                        if name in self._joint_actor_reference
                    )
                    print(
                        "  actor_parameter_max_abs_delta: "
                        f"{actor_parameter_max_abs_delta:.10e}"
                    )
                if self._joint_std_reference is not None:
                    std_parameter_max_abs_delta = (
                        self.alg.actor_critic.std.detach()
                        - self._joint_std_reference
                    ).abs().max().item()
                    print(
                        "  std_parameter_max_abs_delta: "
                        f"{std_parameter_max_abs_delta:.10e}"
                    )
                print("=" * 60 + "\n")
            
            # RMA stability check
            if self.use_rma:
                stable, reason = self.alg.actor_critic.check_stability()
                if not stable:
                    print(f"\n❌ RMA Stability Error - Iteration {it}")
                    print(f"   Reason: {reason}")
                    print(f"   Saving diagnostic checkpoint...")
                    self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}_DIAGNOSTIC.pt'))
                    print(f"   Training stopped due to instability")
                    return
            
            ep_infos.clear()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def _compute_rma_alpha(self, iteration):
        if iteration < self.rma_warmup_iters:
            return 0.0
        progress = (iteration - self.rma_warmup_iters) / self.rma_alpha_ramp_iters
        return min(self.rma_max_alpha, max(0.0, progress * self.rma_max_alpha))

    def _print_learning_rates(self):
        lr_stats = self.alg.get_learning_rate_stats()
        print("\n📌 PPO Learning Rates")
        print(f"  lr_factor: {lr_stats['lr_factor']:.6f}")
        for group_name in ('actor', 'critic', 'rma', 'std'):
            group_stats = lr_stats.get(group_name)
            if group_stats is not None:
                print(
                    f"  {group_name}: "
                    f"base_lr={group_stats['base_lr']:.6e}, "
                    f"current_lr={group_stats['current_lr']:.6e}"
                )
        print("")

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        auxiliary_stats = self.alg.get_auxiliary_loss_stats()
        self.writer.add_scalar(
            'Loss/anchor',
            auxiliary_stats.get('anchor_loss', 0.0),
            locs['it'],
        )
        self.writer.add_scalar(
            'Loss/anchor_mask_ratio',
            auxiliary_stats.get('anchor_mask_ratio', 0.0),
            locs['it'],
        )
        self.writer.add_scalar(
            'Diagnostics/reference_actor_max_abs_diff',
            auxiliary_stats.get('reference_actor_max_abs_diff', 0.0),
            locs['it'],
        )
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        rma_alpha = getattr(self.alg.actor_critic, 'rma_alpha', 0.0)
        checkpoint = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iteration': self.current_learning_iteration,
            'iter': self.current_learning_iteration,
            'rma_alpha': rma_alpha,
            'lr_factor': self.alg.lr_factor,
            'infos': infos,
        }
        if hasattr(self.env, "terrain_levels"):
            checkpoint["terrain_levels"] = (
                self.env.terrain_levels.detach().cpu().clone()
            )
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True):
        path = os.path.realpath(path)
        loaded_dict = torch.load(path, map_location=self.device)
        
        model_state_dict = loaded_dict['model_state_dict']
        current_state_dict = self.alg.actor_critic.state_dict()
        is_rma_checkpoint = any(
            key.startswith("history_encoder")
            or key.startswith("adaptation_module")
            or key.startswith("actor_film")
            for key in model_state_dict
        )

        if self.use_rma and not is_rma_checkpoint:
            print("PPO checkpoint supplied through resume; entering migration mode")
            self._load_and_validate_ppo_migration(
                self.alg.actor_critic,
                path,
                self.env.num_privileged_obs
                if self.env.num_privileged_obs is not None
                else self.env.num_obs,
            )
            self.alg.rebuild_optimizer_for_migration()
            self.current_learning_iteration = 0
            self.alg.actor_critic.set_rma_alpha(0.0)
            self.rma_checkpoint_validated = True
            print("Loaded iteration: 0")
            print("Loaded rma_alpha: 0.000000")
            return loaded_dict.get("infos")
        
        print("\n" + "=" * 80)
        print("Loading model checkpoint...")
        print("=" * 80)
        
        matched_keys = []
        unmatched_current_keys = []
        unmatched_loaded_keys = []
        shape_mismatch_keys = []
        
        for key in current_state_dict.keys():
            if key in model_state_dict:
                if current_state_dict[key].shape == model_state_dict[key].shape:
                    matched_keys.append(key)
                else:
                    shape_mismatch_keys.append((key, current_state_dict[key].shape, model_state_dict[key].shape))
            else:
                unmatched_current_keys.append(key)
        
        for key in model_state_dict.keys():
            if key not in current_state_dict:
                unmatched_loaded_keys.append(key)
        
        print(f"\n✓ Successfully loaded {len(matched_keys)} parameters:")
        for key in matched_keys[:10]:
            print(f"  - {key}")
        if len(matched_keys) > 10:
            print(f"  ... and {len(matched_keys) - 10} more")
        
        if shape_mismatch_keys:
            print(f"\n⚠️ Shape mismatch (will reinitialize):")
            for key, curr_shape, load_shape in shape_mismatch_keys:
                print(f"  - {key}: current={curr_shape}, loaded={load_shape}")
        
        if unmatched_current_keys:
            print(f"\n⚠️ New parameters (reinitialized):")
            for key in unmatched_current_keys[:10]:
                print(f"  - {key}")
            if len(unmatched_current_keys) > 10:
                print(f"  ... and {len(unmatched_current_keys) - 10} more")
        
        if unmatched_loaded_keys:
            print(f"\n⚠️ Old parameters (ignored):")
            for key in unmatched_loaded_keys[:10]:
                print(f"  - {key}")
            if len(unmatched_loaded_keys) > 10:
                print(f"  ... and {len(unmatched_loaded_keys) - 10} more")
        
        if self.use_rma:
            if is_rma_checkpoint:
                print("\n📌 RMA checkpoint detected - loading full model")
                load_result = self.alg.actor_critic.load_state_dict(model_state_dict, strict=False)
                allowed_missing = {
                    key for key in load_result.missing_keys
                    if key.startswith("rear_hip_residual.")
                    and getattr(
                        self.alg.actor_critic,
                        "enable_rear_hip_residual",
                        False,
                    )
                }
                disallowed_missing = set(load_result.missing_keys) - allowed_missing
                if disallowed_missing or load_result.unexpected_keys:
                    raise RuntimeError(
                        "RMA resume checkpoint mismatch: "
                        f"missing={sorted(disallowed_missing)}, "
                        f"unexpected={load_result.unexpected_keys}"
                    )
                if allowed_missing:
                    print(
                        "  New rear-hip residual initialized from its "
                        "identity-preserving defaults"
                    )
                
                rma_params = sum(1 for k in current_state_dict.keys() if k.startswith('history_encoder') or k.startswith('adaptation_module') or k.startswith('actor_film') or k.startswith('critic_film'))
                actor_critic_params = sum(1 for k in current_state_dict.keys() if k.startswith('actor.') or k.startswith('critic.') or k.startswith('std'))
                total_loaded = len(model_state_dict.keys())
                
                print(f"  ✓ Full model loaded: {total_loaded}/{len(current_state_dict.keys())} ({total_loaded/len(current_state_dict.keys())*100:.1f}%)")
                print(f"  ✓ Actor/Critic params: {actor_critic_params}")
                print(f"  ✓ RMA params: {rma_params}")
            else:
                print("\n📌 PPO checkpoint detected - migrating to RMA")
                print("  → Loading actor/critic/std from PPO checkpoint")
                print("  → RMA modules (history_encoder, adaptation_module, film) will be initialized")
                
                filtered_state_dict = {}
                for key, val in model_state_dict.items():
                    if key.startswith('actor.') or key.startswith('critic.') or key.startswith('std'):
                        if key in current_state_dict and current_state_dict[key].shape == val.shape:
                            filtered_state_dict[key] = val
                
                load_result = self.alg.actor_critic.load_state_dict(filtered_state_dict, strict=False)
                
                actor_critic_params = sum(1 for k in current_state_dict.keys() if k.startswith('actor.') or k.startswith('critic.') or k.startswith('std'))
                loaded_count = len(filtered_state_dict)
                rma_params = len(current_state_dict.keys()) - actor_critic_params
                
                print(f"\n  ✓ Actor/Critic loaded: {loaded_count}/{actor_critic_params} ({loaded_count/actor_critic_params*100:.1f}%)")
                print(f"  ✓ std loaded")
                print(f"  ⚠️ RMA modules initialized (history_encoder, adaptation_module, film): {rma_params} params")
                print(f"  ⚠️ Missing keys (expected): {len(load_result.missing_keys)}")
        else:
            load_result = self.alg.actor_critic.load_state_dict(model_state_dict, strict=False)
            print(f"\nLoaded with strict=False: {len(load_result.missing_keys)} missing, {len(load_result.unexpected_keys)} unexpected")
        
        total_current = len(current_state_dict.keys())
        total_loaded = len(model_state_dict.keys())
        print(f"\n📊 Loading Summary:")
        print(f"  Current model: RMAActorCritic ({total_current} params)")
        print(f"  Checkpoint: {'RMA' if any(k.startswith('history_encoder') for k in model_state_dict.keys()) else 'PPO'} ({total_loaded} params)")
        print(f"  Matched params: {len(matched_keys)}")
        print(f"  New RMA params: {len(unmatched_current_keys)}")
        print(f"  Ignored old params: {len(unmatched_loaded_keys)}")
        
        print("=" * 80 + "\n")
        
        if load_optimizer:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
                optimizer_state_count = len(self.alg.optimizer.state)
                print(
                    "Optimizer loaded successfully: "
                    f"Adam state entries={optimizer_state_count}"
                )
                if optimizer_state_count == 0:
                    raise RuntimeError("Loaded optimizer has no Adam momentum state")
            except Exception as e:
                raise RuntimeError(f"Failed to load resume optimizer: {e}") from e
        else:
            print("⚠️ Skipping checkpoint optimizer; using fresh optimizer")
        
        checkpoint_lr_factor = (
            loaded_dict.get('lr_factor', 1.0)
            if load_optimizer else 1.0
        )
        self.alg.apply_lr_factor(checkpoint_lr_factor)
        if self.alg.lr_factor != checkpoint_lr_factor:
            print(
                "Checkpoint lr_factor clamped: "
                f"{checkpoint_lr_factor:.6f} -> {self.alg.lr_factor:.6f}"
            )
        self._print_learning_rates()
        self.current_learning_iteration = loaded_dict.get('iteration', loaded_dict.get('iter', 0))
        loaded_rma_alpha = loaded_dict.get('rma_alpha', self._compute_rma_alpha(self.current_learning_iteration))
        if self.use_rma:
            self.alg.actor_critic.set_rma_alpha(loaded_rma_alpha)
            self.rma_checkpoint_validated = True
        if self.reference_anchor_enabled:
            reference_checkpoint = (
                os.path.realpath(self.reference_actor_checkpoint)
                if self.reference_actor_checkpoint else None
            )
            self.alg.initialize_reference_actor(reference_checkpoint)
        if (
            self.restore_terrain_levels
            and "terrain_levels" in loaded_dict
            and hasattr(
            self.env, "terrain_levels"
            )
        ):
            saved_levels = loaded_dict["terrain_levels"].to(
                device=self.env.terrain_levels.device,
                dtype=self.env.terrain_levels.dtype,
            )
            if saved_levels.shape != self.env.terrain_levels.shape:
                raise RuntimeError(
                    "terrain_levels shape mismatch: "
                    f"checkpoint={tuple(saved_levels.shape)}, "
                    f"env={tuple(self.env.terrain_levels.shape)}"
                )
            self.env.terrain_levels.copy_(saved_levels)
            self.env.env_origins[:] = self.env.terrain_origins[
                self.env.terrain_levels, self.env.terrain_types
            ]
            counts = torch.bincount(
                self.env.terrain_levels.long(),
                minlength=int(self.env.cfg.terrain.num_rows),
            )
            print(f"Restored terrain levels: {counts.tolist()}")
        elif (
            "terrain_levels" in loaded_dict
            and hasattr(self.env, "terrain_levels")
        ):
            counts = torch.bincount(
                self.env.terrain_levels.long(),
                minlength=int(self.env.cfg.terrain.num_rows),
            )
            print(
                "Skipped checkpoint terrain levels; using current task "
                f"distribution: {counts.tolist()}"
            )
        print(f"Loaded iteration: {self.current_learning_iteration}")
        print(f"Loaded rma_alpha: {loaded_rma_alpha:.6f}")
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
