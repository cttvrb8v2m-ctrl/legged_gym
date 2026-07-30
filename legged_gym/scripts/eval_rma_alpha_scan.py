"""Paired deterministic RMA-alpha scan for the joint-trained Go1 policy."""

import json
import math
import os
import random
import types

import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym.algorithms.rma_actor_critic import RMAActorCritic
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


CHECKPOINT = os.path.realpath(
    "/home/gjt/legged_gym/logs/rough_go1/"
    "Jul26_18-44-30_stairs_joint_100_resume310/model_410.pt"
)
OUTPUT_DIR = os.environ.get(
    "GO1_RMA_ALPHA_SCAN_OUTPUT", "/tmp/go1_rma_alpha_scan"
)
SEED = int(os.environ.get("GO1_RMA_ALPHA_SCAN_SEED", "20260723"))
STEP_HEIGHT = float(os.environ["GO1_RMA_ALPHA_SCAN_HEIGHT"])
ALPHAS = (0.0, 0.03, 0.1, 0.3, 0.6, 1.0)
EPISODES = 10
GROUP_SIZE = EPISODES
NUM_ENVS = GROUP_SIZE * len(ALPHAS)


class DistributionStats:
    def __init__(self, keep_values=False):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.max_abs = 0.0
        self.keep_values = keep_values
        self.values = []

    def update(self, tensor):
        value = tensor.detach().float()
        finite = value[torch.isfinite(value)]
        if finite.numel() == 0:
            return
        self.count += finite.numel()
        self.total += finite.double().sum().item()
        self.total_sq += finite.double().square().sum().item()
        self.max_abs = max(self.max_abs, finite.abs().max().item())
        if self.keep_values:
            self.values.append(finite.abs().cpu())

    def summary(self):
        if self.count == 0:
            return {
                "mean": None, "std": None, "max_abs": None,
                "p95_abs": None, "p99_abs": None,
            }
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        result = {
            "mean": mean,
            "std": math.sqrt(variance),
            "max_abs": self.max_abs,
        }
        if self.keep_values:
            values = torch.cat(self.values)
            result["p95_abs"] = torch.quantile(values, 0.95).item()
            result["p99_abs"] = torch.quantile(values, 0.99).item()
        else:
            result["p95_abs"] = None
            result["p99_abs"] = None
        return result


def configure_environment(cfg):
    cfg.env.num_envs = NUM_ENVS
    cfg.env.episode_length_s = 20.0
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.curriculum = True
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 1
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.specialized_stair_training = True
    cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
    cfg.terrain.stair_stage_height_ranges = [
        [STEP_HEIGHT, STEP_HEIGHT],
        [STEP_HEIGHT, STEP_HEIGHT],
        [STEP_HEIGHT, STEP_HEIGHT],
    ]
    cfg.terrain.stair_tread_depth = 0.30
    cfg.terrain.stair_step_count = 9
    cfg.normalization.clip_actions = 100.0
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [1.0, 1.0]
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_limb_mass = False
    cfg.domain_rand.push_robots = False
    cfg.commands.resampling_time = 1000.0
    cfg.commands.heading_command = False
    cfg.commands.ranges.lin_vel_x = [0.4, 0.8]
    cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]


def make_rma(device):
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model = RMAActorCritic(
        235,
        235,
        12,
        history_len=10,
        latent_dim=32,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, float(checkpoint["rma_alpha"])


def group_ids(device, index):
    return torch.arange(
        index * GROUP_SIZE,
        (index + 1) * GROUP_SIZE,
        device=device,
    )


def pair_initial_state(env):
    source = group_ids(env.device, 0)
    for index in range(1, len(ALPHAS)):
        target = group_ids(env.device, index)
        env.root_states[target] = env.root_states[source]
        env.dof_state[target] = env.dof_state[source]

    all_ids = torch.arange(NUM_ENVS, device=env.device, dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(all_ids),
        NUM_ENVS,
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(all_ids),
        NUM_ENVS,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 9001)
    commands = torch.zeros(GROUP_SIZE, env.commands.shape[1])
    commands[:, 0] = 0.4 + 0.4 * torch.rand(
        GROUP_SIZE, generator=generator
    )
    commands = commands.to(env.device)
    for index in range(len(ALPHAS)):
        env.commands[group_ids(env.device, index)] = commands

    env.actions.zero_()
    env.last_actions.zero_()
    env.last_dof_vel.zero_()
    env.last_root_vel.zero_()
    env.obs_history.zero_()
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(
        env.base_quat, env.root_states[:, 7:10]
    )
    env.base_ang_vel[:] = quat_rotate_inverse(
        env.base_quat, env.root_states[:, 10:13]
    )
    env.projected_gravity[:] = quat_rotate_inverse(
        env.base_quat, env.gravity_vec
    )
    env.measured_heights = env._get_heights()
    env.compute_observations()
    for index in range(1, len(ALPHAS)):
        target = group_ids(env.device, index)
        env.obs_buf[target] = env.obs_buf[source]
        env.obs_history[target] = env.obs_history[source]

    diffs = {}
    for name, tensor in (
        ("root", env.root_states),
        ("dof", env.dof_state),
        ("command", env.commands),
        ("observation", env.obs_buf),
        ("history", env.obs_history),
    ):
        diffs[name] = max(
            (tensor[source] - tensor[group_ids(env.device, index)])
            .abs().max().item()
            for index in range(1, len(ALPHAS))
        )
    return diffs


def install_eval_callbacks(env, fixed_commands):
    def fixed_resample(self, env_ids):
        self.commands[env_ids] = fixed_commands[env_ids]

    def no_reset(self, env_ids):
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf.clone()

    env._resample_commands = types.MethodType(fixed_resample, env)
    env.reset_idx = types.MethodType(no_reset, env)


def model_forward(model, obs, history, alpha):
    latent = model.compute_latent(history)
    film_params = model.actor_film(latent)
    gamma_logits, beta_logits = torch.chunk(film_params, 2, dim=-1)
    gamma = model.film_scale * torch.tanh(gamma_logits)
    beta = model.film_scale * torch.tanh(beta_logits)
    hidden = model.actor[0](obs)
    hidden = hidden * (1.0 + alpha * gamma) + alpha * beta
    for layer_index in range(1, len(model.actor)):
        hidden = model.actor[layer_index](hidden)
    base = model.compute_base_action_mean(obs)
    return hidden, base, latent, gamma, beta


def root_roll_pitch(quaternion):
    x, y, z, w = quaternion.unbind(dim=-1)
    roll = torch.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x.square() + y.square()),
    )
    pitch = torch.asin(
        torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    )
    return roll, pitch


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    args = get_args()
    env_cfg, _ = task_registry.get_cfgs("go1_stairs_joint")
    configure_environment(env_cfg)
    env_cfg.seed = SEED
    env, _ = task_registry.make_env(
        name="go1_stairs_joint", args=args, env_cfg=env_cfg
    )
    env.reset()
    model, checkpoint_alpha = make_rma(env.device)
    initial_diff = pair_initial_state(env)
    fixed_commands = env.commands.clone()
    install_eval_callbacks(env, fixed_commands)

    active = torch.ones(NUM_ENVS, dtype=torch.bool, device=env.device)
    elapsed_steps = torch.zeros(
        NUM_ENVS, dtype=torch.long, device=env.device
    )
    fall = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(fall)
    max_foot_level = torch.zeros(
        NUM_ENVS, len(env.feet_indices), dtype=torch.long, device=env.device
    )
    front_collision_count = torch.zeros(NUM_ENVS, device=env.device)
    calf_collision_count = torch.zeros(NUM_ENVS, device=env.device)
    last_front_collision = torch.zeros(
        NUM_ENVS, len(env.front_foot_slots),
        dtype=torch.bool, device=env.device,
    )
    last_calf_collision = torch.zeros(
        NUM_ENVS, len(env.calf_indices),
        dtype=torch.bool, device=env.device,
    )
    rear_slip_episode = torch.zeros_like(fall)
    severe_jitter_count = torch.zeros(NUM_ENVS, device=env.device)
    last_policy_action = torch.zeros(
        NUM_ENVS, env.num_actions, device=env.device
    )
    invalid = torch.zeros(len(ALPHAS), dtype=torch.bool)
    action_stats = [
        DistributionStats(keep_values=True) for _ in ALPHAS
    ]
    delta_stats = [
        DistributionStats(keep_values=True) for _ in ALPHAS
    ]
    latent_stats = [DistributionStats() for _ in ALPHAS]
    gamma_saturated = [0 for _ in ALPHAS]
    gamma_count = [0 for _ in ALPHAS]
    beta_saturated = [0 for _ in ALPHAS]
    beta_count = [0 for _ in ALPHAS]

    env.gym.refresh_rigid_body_state_tensor(env.sim)
    initial_foot_pos = env.rigid_body_state[
        :, env.feet_indices, :3
    ].clone()
    initial_ground = env._sample_terrain_height_at(initial_foot_pos)

    obs = env.obs_buf
    for _ in range(int(env.max_episode_length) + 1):
        if not active.any():
            break
        actions = torch.zeros(NUM_ENVS, env.num_actions, device=env.device)
        with torch.inference_mode():
            for index, alpha in enumerate(ALPHAS):
                ids = group_ids(env.device, index)
                group_active = active[ids]
                policy, base, latent, gamma, beta = model_forward(
                    model, obs[ids], env.obs_history[ids], alpha
                )
                finite = (
                    torch.isfinite(policy).all()
                    and torch.isfinite(latent).all()
                    and torch.isfinite(gamma).all()
                    and torch.isfinite(beta).all()
                )
                if not finite:
                    invalid[index] = True
                    active[ids] = False
                    continue
                actions[ids] = policy
                action_stats[index].update(policy[group_active])
                delta_stats[index].update((policy - base)[group_active])
                latent_stats[index].update(latent[group_active])
                active_gamma = gamma[group_active]
                active_beta = beta[group_active]
                gamma_saturated[index] += int(
                    (active_gamma.abs() > 0.095).sum().item()
                )
                gamma_count[index] += active_gamma.numel()
                beta_saturated[index] += int(
                    (active_beta.abs() > 0.095).sum().item()
                )
                beta_count[index] += active_beta.numel()
                # A deliberately loose hard stop catches genuine explosions
                # without treating the known PPO action tail as instability.
                if policy.abs().max().item() > 50.0:
                    invalid[index] = True
                    active[ids] = False

        actions[~active] = 0.0
        was_active = active.clone()
        severe_jitter_count += (
            (actions - last_policy_action).abs().max(dim=1).values > 5.0
        ).float() * was_active
        last_policy_action[:] = actions
        obs, _, _, dones, _ = env.step(actions)
        elapsed_steps[was_active] += 1

        env.gym.refresh_rigid_body_state_tensor(env.sim)
        foot_state = env.rigid_body_state[:, env.feet_indices]
        foot_pos = foot_state[..., :3]
        foot_vel = foot_state[..., 7:10]
        terrain_height = env._sample_terrain_height_at(foot_pos)
        foot_contact = env.contact_forces[:, env.feet_indices, 2] > 1.0
        height_level = torch.clamp(
            torch.round(
                (terrain_height - initial_ground) / STEP_HEIGHT
            ).long(),
            0,
            env.cfg.terrain.stair_step_count,
        )
        contacted_level = torch.where(
            foot_contact, height_level, torch.zeros_like(height_level)
        )
        max_foot_level = torch.maximum(max_foot_level, contacted_level)

        front_force = env.contact_forces[
            :, env.feet_indices[env.front_foot_slots]
        ]
        front_collision = (
            torch.norm(front_force[..., :2], dim=-1)
            > torch.maximum(
                torch.full_like(front_force[..., 2], 5.0),
                2.0 * torch.abs(front_force[..., 2]),
            )
        )
        calf_collision = (
            torch.norm(env.contact_forces[:, env.calf_indices], dim=-1)
            > 5.0
        )
        front_collision_count += (
            front_collision & ~last_front_collision
        ).float().sum(dim=1) * was_active
        calf_collision_count += (
            calf_collision & ~last_calf_collision
        ).float().sum(dim=1) * was_active
        last_front_collision[:] = front_collision
        last_calf_collision[:] = calf_collision

        rear_contact = foot_contact[:, env.rear_foot_slots]
        rear_speed = torch.norm(
            foot_vel[:, env.rear_foot_slots, :2], dim=-1
        )
        rear_slip_episode |= (
            torch.any(rear_contact & (rear_speed > 0.20), dim=1)
            & was_active
        )

        reached_top = torch.all(
            max_foot_level >= env.cfg.terrain.stair_step_count, dim=1
        )
        new_success = active & reached_top
        success[new_success] = True
        new_done = active & dones.bool()
        fall[new_done] = ~env.time_out_buf[new_done]
        active &= ~(new_success | new_done)

    passed_steps = torch.min(max_foot_level, dim=1).values
    records = []
    summaries = []
    for index, alpha in enumerate(ALPHAS):
        ids = group_ids(env.device, index)
        for replicate, env_id in enumerate(ids.tolist()):
            records.append({
                "alpha": alpha,
                "replicate": replicate,
                "success": bool(success[env_id].item()),
                "passed_steps": int(passed_steps[env_id].item()),
                "fall": bool(fall[env_id].item()),
                "rear_slip": bool(rear_slip_episode[env_id].item()),
                "foot_top_landing_ratio": (
                    max_foot_level[env_id].float()
                    / env.cfg.terrain.stair_step_count
                ).tolist(),
                "front_collision_count": float(
                    front_collision_count[env_id].item()
                ),
                "calf_collision_count": float(
                    calf_collision_count[env_id].item()
                ),
                "episode_length": int(elapsed_steps[env_id].item()),
                "severe_jitter_events": int(
                    severe_jitter_count[env_id].item()
                ),
            })
        group_records = records[-GROUP_SIZE:]
        summaries.append({
            "alpha": alpha,
            "success_rate": sum(x["success"] for x in group_records)
            / GROUP_SIZE,
            "mean_passed_steps": sum(
                x["passed_steps"] for x in group_records
            ) / GROUP_SIZE,
            "fall_rate": sum(x["fall"] for x in group_records)
            / GROUP_SIZE,
            "rear_slip_rate": sum(
                x["rear_slip"] for x in group_records
            ) / GROUP_SIZE,
            "foot_top_landing_ratio": [
                sum(x["foot_top_landing_ratio"][foot] for x in group_records)
                / GROUP_SIZE
                for foot in range(4)
            ],
            "front_collision_mean": sum(
                x["front_collision_count"] for x in group_records
            ) / GROUP_SIZE,
            "calf_collision_mean": sum(
                x["calf_collision_count"] for x in group_records
            ) / GROUP_SIZE,
            "severe_jitter_mean": sum(
                x["severe_jitter_events"] for x in group_records
            ) / GROUP_SIZE,
            "policy_action": action_stats[index].summary(),
            "rma_delta": delta_stats[index].summary(),
            "latent": latent_stats[index].summary(),
            "gamma_saturation_ratio": (
                gamma_saturated[index] / gamma_count[index]
                if gamma_count[index] else None
            ),
            "beta_saturation_ratio": (
                beta_saturated[index] / beta_count[index]
                if beta_count[index] else None
            ),
            "nan_inf_or_hard_stop": bool(invalid[index].item()),
        })

    output = {
        "checkpoint": CHECKPOINT,
        "checkpoint_alpha": checkpoint_alpha,
        "seed": SEED,
        "step_height_m": STEP_HEIGHT,
        "tread_depth_m": 0.30,
        "step_count": 9,
        "friction": 1.0,
        "base_mass_kg": 5.205080032348633,
        "deterministic_act_inference": True,
        "training_updates": False,
        "clip_actions": 100.0,
        "alphas": ALPHAS,
        "episodes_per_alpha": EPISODES,
        "initial_pairing_max_abs_diff": initial_diff,
        "summaries": summaries,
        "records": records,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    height_tag = f"{STEP_HEIGHT:.3f}".replace(".", "p")
    output_path = os.path.join(
        OUTPUT_DIR, f"alpha_scan_height{height_tag}_seed{SEED}.json"
    )
    with open(output_path, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
    print(json.dumps({
        "output": output_path,
        "height": STEP_HEIGHT,
        "initial_max_diff": max(initial_diff.values()),
        "summaries": summaries,
    }))
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
