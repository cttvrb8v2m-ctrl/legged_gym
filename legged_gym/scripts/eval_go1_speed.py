"""Deterministic model_550 speed baseline on fixed flat or rough terrain."""

import hashlib
import json
import os
import random
import types

import isaacgym
import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs import *
from legged_gym.algorithms.rma_actor_critic import RMAActorCritic
from legged_gym.envs.base.legged_robot import gymtorch
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic


CHECKPOINT = os.path.realpath(
    os.environ.get(
        "GO1_SPEED_EVAL_CHECKPOINT",
        "/home/gjt/legged_gym/logs/rough_go1/6/model_550.pt",
    )
)
RMA_ALPHA = float(os.environ.get("GO1_SPEED_EVAL_ALPHA", "0.0"))
RESIDUAL_MAX_ACTION = float(
    os.environ.get("GO1_SPEED_EVAL_RESIDUAL_MAX_ACTION", "0.08")
)
OUTPUT_DIR = os.environ.get(
    "GO1_SPEED_EVAL_OUTPUT",
    "/tmp/go1_model550_speed",
)
SEED = int(os.environ["GO1_SPEED_EVAL_SEED"])
COMMAND_X = float(os.environ["GO1_SPEED_EVAL_COMMAND_X"])
TERRAIN_KIND = os.environ["GO1_SPEED_EVAL_TERRAIN"]
EPISODES = int(os.environ.get("GO1_SPEED_EVAL_EPISODES", "100"))


def configure_environment(cfg):
    cfg.env.num_envs = EPISODES
    cfg.env.episode_length_s = 20.0
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.curriculum = True
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 1
    cfg.terrain.max_init_terrain_level = 0
    # A 20 s run at 3 m/s travels about 60 m. Keep the robot on the
    # generated surface for the whole posture evaluation.
    cfg.terrain.terrain_length = 160.0
    cfg.terrain.terrain_width = 8.0
    cfg.terrain.specialized_stair_training = True
    cfg.terrain.rough_height = 0.05
    cfg.terrain.stair_stage_height_ranges = [
        [0.10, 0.10],
        [0.10, 0.10],
        [0.10, 0.10],
    ]
    cfg.terrain.stair_step_count = 9
    if TERRAIN_KIND == "flat":
        # Use the simulator's unbounded plane so a 20 s high-speed run
        # cannot leave a finite height-field tile.
        cfg.terrain.mesh_type = "plane"
        cfg.terrain.specialized_stair_training = False
        cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    elif TERRAIN_KIND == "rough":
        cfg.terrain.terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0]
    else:
        raise ValueError(f"Unsupported terrain: {TERRAIN_KIND}")
    cfg.normalization.clip_actions = 100.0
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [1.0, 1.0]
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_limb_mass = False
    cfg.domain_rand.push_robots = False
    cfg.commands.resampling_time = 1000.0
    cfg.commands.heading_command = False
    cfg.commands.ranges.lin_vel_x = [COMMAND_X, COMMAND_X]
    cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]


def load_policy(device):
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    is_rma = any(
        key.startswith("history_encoder.") for key in state_dict
    )
    if is_rma:
        residual_enabled = any(
            key.startswith("rear_hip_residual.") for key in state_dict
        )
        residual_independent = (
            residual_enabled
            and state_dict["rear_hip_residual.2.weight"].shape[0] == 2
        )
        policy = RMAActorCritic(
            235,
            235,
            12,
            history_len=10,
            latent_dim=32,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=1.0,
            enable_rear_hip_residual=residual_enabled,
            rear_hip_residual_independent=residual_independent,
            rear_hip_residual_max_action=RESIDUAL_MAX_ACTION,
        ).to(device)
        policy.set_rma_alpha(RMA_ALPHA)
    else:
        policy = ActorCritic(
            235,
            235,
            12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=1.0,
        ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


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


def install_fixed_episode_callbacks(env, commands):
    def fixed_resample(self, env_ids):
        self.commands[env_ids] = commands[env_ids]

    def no_reset(self, env_ids):
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf.clone()

    env._resample_commands = types.MethodType(fixed_resample, env)
    env.reset_idx = types.MethodType(no_reset, env)


def tensor_hash(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def restore_or_save_paired_initial_state(env):
    """Use one bit-identical physical state for all commands of a seed/terrain."""
    state_dir = os.path.join(OUTPUT_DIR, "paired_initial_states")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(
        state_dir, f"{TERRAIN_KIND}_seed{SEED}.pt"
    )
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=env.device)
        env.root_states.copy_(state["root_states"])
        env.dof_state.copy_(state["dof_state"])
        env.gym.set_actor_root_state_tensor(
            env.sim, gymtorch.unwrap_tensor(env.root_states)
        )
        env.gym.set_dof_state_tensor(
            env.sim, gymtorch.unwrap_tensor(env.dof_state)
        )
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
    else:
        torch.save({
            "root_states": env.root_states.detach().cpu().clone(),
            "dof_state": env.dof_state.detach().cpu().clone(),
        }, state_path)

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
    return os.path.realpath(state_path)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    args = get_args()
    env_cfg, _ = task_registry.get_cfgs("go1")
    configure_environment(env_cfg)
    env_cfg.seed = SEED
    env, _ = task_registry.make_env(
        name="go1", args=args, env_cfg=env_cfg
    )
    device = env.device
    env.reset()
    paired_initial_state_path = restore_or_save_paired_initial_state(env)

    commands = torch.zeros_like(env.commands)
    commands[:, 0] = COMMAND_X
    env.commands[:] = commands
    env.actions.zero_()
    env.last_actions.zero_()
    env.last_dof_vel.zero_()
    env.last_root_vel.zero_()
    if hasattr(env, "obs_history"):
        env.obs_history.zero_()
    env.compute_observations()
    install_fixed_episode_callbacks(env, commands.clone())

    policy = load_policy(device)
    active = torch.ones(EPISODES, dtype=torch.bool, device=device)
    elapsed_steps = torch.zeros(
        EPISODES, dtype=torch.long, device=device
    )
    reward_sum = torch.zeros(EPISODES, device=device)
    velocity_sum = torch.zeros(EPISODES, device=device)
    abs_error_sum = torch.zeros(EPISODES, device=device)
    tracking_sum = torch.zeros(EPISODES, device=device)
    pitch_sum = torch.zeros(EPISODES, device=device)
    abs_pitch_sum = torch.zeros(EPISODES, device=device)
    abs_roll_sum = torch.zeros(EPISODES, device=device)
    max_abs_pitch = torch.zeros(EPISODES, device=device)
    max_abs_roll = torch.zeros(EPISODES, device=device)
    base_height_sum = torch.zeros(EPISODES, device=device)
    min_base_height = torch.full(
        (EPISODES,), float("inf"), device=device
    )
    fall = torch.zeros(EPISODES, dtype=torch.bool, device=device)
    max_steps = int(env.max_episode_length)
    action_abs = torch.full(
        (max_steps + 1, EPISODES, env.num_actions),
        float("nan"), device=device,
    )
    dof_position_samples = torch.full(
        (max_steps + 1, EPISODES, env.num_dof),
        float("nan"), device=device,
    )
    dof_torque_samples = torch.full_like(
        dof_position_samples, float("nan")
    )
    residual_amplitude_samples = torch.full(
        (max_steps + 1, EPISODES),
        float("nan"), device=device,
    )
    foot_position_sum = torch.zeros(
        EPISODES, len(env.feet_indices), 3, device=device
    )
    foot_contact_steps = torch.zeros(
        EPISODES, len(env.feet_indices), device=device
    )
    foot_swing_height_samples = torch.full(
        (max_steps + 1, EPISODES, len(env.feet_indices)),
        float("nan"), device=device,
    )
    rear_separation_samples = torch.full(
        (max_steps + 1, EPISODES),
        float("nan"), device=device,
    )
    rear_contact_separation_samples = torch.full_like(
        rear_separation_samples, float("nan")
    )
    rear_touchdown_separation_samples = torch.full_like(
        rear_separation_samples, float("nan")
    )
    last_rear_contact = torch.zeros(
        EPISODES, 2, dtype=torch.bool, device=device
    )

    physical_initial_state_hash = tensor_hash(
        env.root_states,
        env.dof_pos,
        env.dof_vel,
    )
    initial_observation_hash = tensor_hash(env.obs_buf)
    terrain_parameters = {
        "kind": TERRAIN_KIND,
        "rough_height": 0.05 if TERRAIN_KIND == "rough" else 0.0,
        "friction": 1.0,
        "rows": 1,
        "cols": 1,
        "terrain_length": env.cfg.terrain.terrain_length,
        "terrain_width": env.cfg.terrain.terrain_width,
    }
    terrain_digest = hashlib.sha256(
        json.dumps(terrain_parameters, sort_keys=True).encode("utf-8")
    )
    if env.height_samples is not None:
        terrain_digest.update(
            env.height_samples.detach().cpu().numpy().tobytes()
        )
    terrain_hash = terrain_digest.hexdigest()

    obs = env.obs_buf
    for step_index in range(max_steps + 1):
        if not active.any():
            break
        actions = torch.zeros(
            EPISODES, env.num_actions, device=device
        )
        with torch.inference_mode():
            if isinstance(policy, RMAActorCritic):
                actions[active] = policy.act_inference(
                    obs[active],
                    obs_history=env.obs_history[active],
                )
                residual_amplitude_samples[
                    step_index, active
                ] = policy.compute_rear_hip_residual_amplitude(
                    obs[active]
                )
            else:
                actions[active] = policy.act_inference(obs[active])
                residual_amplitude_samples[step_index, active] = 0.0
        was_active = active.clone()
        action_abs[step_index, was_active] = actions[was_active].abs()
        obs, _, rewards, dones, _ = env.step(actions)
        dof_position_samples[step_index, was_active] = env.dof_pos[
            was_active
        ]
        dof_torque_samples[step_index, was_active] = env.torques[
            was_active
        ]
        foot_world = env.rigid_body_state[:, env.feet_indices, :3]
        foot_offset = foot_world - env.root_states[:, None, :3]
        foot_base = quat_rotate_inverse(
            env.base_quat[:, None, :]
            .expand(-1, len(env.feet_indices), -1)
            .reshape(-1, 4),
            foot_offset.reshape(-1, 3),
        ).reshape(EPISODES, len(env.feet_indices), 3)
        foot_position_sum[was_active] += foot_base[was_active]
        # Isaac Gym body order for this Go1 asset is FL, FR, RL, RR.
        # Positive separation means the
        # rear feet remain correctly ordered in the base frame.
        rear_separation = foot_base[:, 2, 1] - foot_base[:, 3, 1]
        rear_separation_samples[
            step_index, was_active
        ] = rear_separation[was_active]
        foot_contact = (
            env.contact_forces[:, env.feet_indices, 2] > 1.0
        )
        rear_contact = foot_contact[:, [2, 3]]
        any_rear_contact = torch.any(rear_contact, dim=1) & was_active
        new_rear_touchdown = (
            torch.any(rear_contact & ~last_rear_contact, dim=1)
            & was_active
        )
        rear_contact_separation_samples[
            step_index, any_rear_contact
        ] = rear_separation[any_rear_contact]
        rear_touchdown_separation_samples[
            step_index, new_rear_touchdown
        ] = rear_separation[new_rear_touchdown]
        last_rear_contact[:] = rear_contact
        foot_contact_steps[was_active] += foot_contact[was_active]
        swing = was_active[:, None] & ~foot_contact
        foot_height = foot_world[..., 2] - env.env_origins[:, None, 2]
        step_swing_heights = foot_swing_height_samples[step_index]
        step_swing_heights[swing] = foot_height[swing]

        velocity = env.base_lin_vel[:, 0]
        error = torch.abs(velocity - COMMAND_X)
        tracking = torch.exp(
            -torch.sum(
                torch.square(
                    env.commands[:, :2] - env.base_lin_vel[:, :2]
                ),
                dim=1,
            )
            / env.cfg.rewards.tracking_sigma
        )
        elapsed_steps[was_active] += 1
        reward_sum[was_active] += rewards[was_active]
        velocity_sum[was_active] += velocity[was_active]
        abs_error_sum[was_active] += error[was_active]
        tracking_sum[was_active] += tracking[was_active]
        roll, pitch = root_roll_pitch(env.root_states[:, 3:7])
        abs_roll = roll.abs()
        abs_pitch = pitch.abs()
        base_height = env.root_states[:, 2] - env.env_origins[:, 2]
        pitch_sum[was_active] += pitch[was_active]
        abs_pitch_sum[was_active] += abs_pitch[was_active]
        abs_roll_sum[was_active] += abs_roll[was_active]
        max_abs_pitch[was_active] = torch.maximum(
            max_abs_pitch[was_active], abs_pitch[was_active]
        )
        max_abs_roll[was_active] = torch.maximum(
            max_abs_roll[was_active], abs_roll[was_active]
        )
        min_base_height[was_active] = torch.minimum(
            min_base_height[was_active], base_height[was_active]
        )
        base_height_sum[was_active] += base_height[was_active]

        new_done = active & dones.bool()
        fall[new_done] = ~env.time_out_buf[new_done]
        active &= ~new_done

    records = []
    safe_steps = torch.clamp(elapsed_steps, min=1)
    for env_id in range(EPISODES):
        values = action_abs[:, env_id].reshape(-1)
        values = values[torch.isfinite(values)]
        rear_separation = rear_separation_samples[:, env_id]
        rear_separation = rear_separation[
            torch.isfinite(rear_separation)
        ]
        rear_contact_separation = rear_contact_separation_samples[
            :, env_id
        ]
        rear_contact_separation = rear_contact_separation[
            torch.isfinite(rear_contact_separation)
        ]
        rear_touchdown_separation = rear_touchdown_separation_samples[
            :, env_id
        ]
        rear_touchdown_separation = rear_touchdown_separation[
            torch.isfinite(rear_touchdown_separation)
        ]
        residual_amplitude = residual_amplitude_samples[:, env_id]
        residual_amplitude = residual_amplitude[
            torch.isfinite(residual_amplitude)
        ]
        def optional_quantile(values, quantile):
            if values.numel() == 0:
                return None
            return float(torch.quantile(values, quantile).item())
        records.append({
            "seed": SEED,
            "terrain": TERRAIN_KIND,
            "command_x": COMMAND_X,
            "replicate": env_id,
            "mean_x_velocity": float(
                (velocity_sum[env_id] / safe_steps[env_id]).item()
            ),
            "mean_abs_tracking_error": float(
                (abs_error_sum[env_id] / safe_steps[env_id]).item()
            ),
            "tracking_lin_vel": float(
                (tracking_sum[env_id] / safe_steps[env_id]).item()
            ),
            "mean_pitch_rad": float(
                (pitch_sum[env_id] / safe_steps[env_id]).item()
            ),
            "mean_abs_pitch_rad": float(
                (abs_pitch_sum[env_id] / safe_steps[env_id]).item()
            ),
            "max_abs_pitch_rad": float(max_abs_pitch[env_id].item()),
            "mean_abs_roll_rad": float(
                (abs_roll_sum[env_id] / safe_steps[env_id]).item()
            ),
            "max_abs_roll_rad": float(max_abs_roll[env_id].item()),
            "mean_base_height_m": float(
                (base_height_sum[env_id] / safe_steps[env_id]).item()
            ),
            "min_base_height_m": float(min_base_height[env_id].item()),
            "fall": bool(fall[env_id].item()),
            "episode_length": int(elapsed_steps[env_id].item()),
            "reward": float(reward_sum[env_id].item()),
            "action_p99": float(torch.quantile(values, 0.99).item()),
            "action_max": float(values.max().item()),
            "rear_width_m_p05": float(torch.quantile(
                rear_separation.abs(), 0.05
            ).item()),
            "rear_width_m_p50": float(torch.quantile(
                rear_separation.abs(), 0.50
            ).item()),
            "rear_width_m_p95": float(torch.quantile(
                rear_separation.abs(), 0.95
            ).item()),
            "rear_width_below_0p12_ratio": float(
                (rear_separation.abs() < 0.12).float().mean().item()
            ),
            "rear_width_above_0p14_ratio": float(
                (rear_separation.abs() > 0.14).float().mean().item()
            ),
            "rear_foot_crossing_ratio": float(
                (rear_separation < 0.0).float().mean().item()
            ),
            "rear_contact_width_m_p05": optional_quantile(
                rear_contact_separation.abs(), 0.05
            ),
            "rear_contact_width_m_p50": optional_quantile(
                rear_contact_separation.abs(), 0.50
            ),
            "rear_touchdown_width_m_p05": optional_quantile(
                rear_touchdown_separation.abs(), 0.05
            ),
            "rear_touchdown_width_m_p50": optional_quantile(
                rear_touchdown_separation.abs(), 0.50
            ),
            "rear_touchdown_width_m_p95": optional_quantile(
                rear_touchdown_separation.abs(), 0.95
            ),
            "rear_touchdown_below_0p12_ratio": (
                None if rear_touchdown_separation.numel() == 0 else float(
                (rear_touchdown_separation.abs() < 0.12)
                .float().mean().item()
                )
            ),
            "rear_touchdown_above_0p14_ratio": (
                None if rear_touchdown_separation.numel() == 0 else float(
                    (rear_touchdown_separation.abs() > 0.14)
                    .float().mean().item()
                )
            ),
            "rear_touchdown_count": int(
                rear_touchdown_separation.numel()
            ),
            "rear_hip_residual_action_p50": float(
                torch.quantile(residual_amplitude, 0.50).item()
            ),
            "rear_hip_residual_action_p95": float(
                torch.quantile(residual_amplitude, 0.95).item()
            ),
            "rear_hip_residual_action_max": float(
                residual_amplitude.max().item()
            ),
            "rear_hip_residual_saturation_ratio": float(
                (
                    residual_amplitude
                    >= 0.99 * getattr(
                        policy, "rear_hip_residual_max_action", 0.08
                    )
                ).float().mean().item()
            ),
        })

    with open(CHECKPOINT, "rb") as checkpoint_stream:
        checkpoint_sha256 = hashlib.sha256(
            checkpoint_stream.read()
        ).hexdigest()
    valid_dof = torch.isfinite(dof_position_samples)
    joint_statistics = {}
    for joint_index, joint_name in enumerate(env.dof_names):
        values = dof_position_samples[..., joint_index]
        values = values[valid_dof[..., joint_index]]
        torque_values = dof_torque_samples[..., joint_index]
        torque_values = torque_values[
            torch.isfinite(torque_values)
        ].abs()
        joint_statistics[joint_name] = {
            "mean_rad": float(values.mean().item()),
            "p05_rad": float(torch.quantile(values, 0.05).item()),
            "p95_rad": float(torch.quantile(values, 0.95).item()),
            "min_rad": float(values.min().item()),
            "max_rad": float(values.max().item()),
            "abs_torque_p95_nm": float(
                torch.quantile(torque_values, 0.95).item()
            ),
            "abs_torque_max_nm": float(torque_values.max().item()),
        }

    foot_names = list(env.feet_names)
    total_active_steps = torch.clamp(elapsed_steps.float(), min=1.0)
    foot_statistics = {}
    for foot_index, foot_name in enumerate(foot_names):
        swing_values = foot_swing_height_samples[..., foot_index]
        swing_values = swing_values[torch.isfinite(swing_values)]
        mean_position = (
            foot_position_sum[:, foot_index]
            / total_active_steps[:, None]
        ).mean(dim=0)
        foot_statistics[foot_name] = {
            "mean_base_position_m": mean_position.tolist(),
            "mean_contact_ratio": float(
                (
                    foot_contact_steps[:, foot_index]
                    / total_active_steps
                ).mean().item()
            ),
            "swing_height_mean_m": float(swing_values.mean().item()),
            "swing_height_p95_m": float(
                torch.quantile(swing_values, 0.95).item()
            ),
            "swing_height_max_m": float(swing_values.max().item()),
        }

    output = {
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_filename_iteration": 550,
        "checkpoint_internal_iter_ignored": True,
        "seed": SEED,
        "terrain": TERRAIN_KIND,
        "command_x": COMMAND_X,
        "episodes": EPISODES,
        "deterministic_act_inference": True,
        "rma_alpha": RMA_ALPHA,
        "rear_hip_residual_max_action": RESIDUAL_MAX_ACTION,
        "training_updates": False,
        "clip_actions": 100.0,
        "friction": 1.0,
        "normal_mass": True,
        "paired_initial_state_path": paired_initial_state_path,
        "physical_initial_state_hash": physical_initial_state_hash,
        "initial_observation_hash": initial_observation_hash,
        "terrain_parameters": terrain_parameters,
        "terrain_hash": terrain_hash,
        "joint_statistics": joint_statistics,
        "foot_statistics": foot_statistics,
        "records": records,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    speed_tag = f"{COMMAND_X:.1f}".replace(".", "p")
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{TERRAIN_KIND}_speed{speed_tag}_seed{SEED}.json",
    )
    with open(output_path, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
    print(json.dumps({
        "output": output_path,
        "physical_initial_state_hash": physical_initial_state_hash,
        "terrain_hash": terrain_hash,
    }))
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
