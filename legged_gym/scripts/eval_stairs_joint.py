"""Deterministic, paired A/B/C stair evaluation for the joint-trained Go1."""

import json
import hashlib
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
from rsl_rl.modules import ActorCritic


PPO_CHECKPOINT = os.path.realpath(
    "/home/gjt/legged_gym/logs/rough_go1/6/model_550.pt"
)
JOINT_CHECKPOINT = os.path.realpath(os.environ.get(
    "GO1_STAIRS_JOINT_CHECKPOINT",
    "/home/gjt/legged_gym/logs/rough_go1/"
    "Jul26_18-44-30_stairs_joint_100_resume310/model_410.pt",
))
RESIDUAL_MAX_ACTION = float(
    os.environ.get("GO1_STAIRS_RESIDUAL_MAX_ACTION", "0.08")
)
PAIR_CHECKPOINT = os.environ.get("GO1_STAIRS_PAIR_CHECKPOINT")
PAIR_MODE = PAIR_CHECKPOINT is not None
SERIES_CHECKPOINTS_RAW = os.environ.get(
    "GO1_STAIRS_CHECKPOINT_SERIES"
)
SERIES_MODE = SERIES_CHECKPOINTS_RAW is not None
ALPHA_SERIES_RAW = os.environ.get("GO1_STAIRS_ALPHA_SERIES")
ALPHA_SERIES = (
    tuple(float(value) for value in ALPHA_SERIES_RAW.split(","))
    if ALPHA_SERIES_RAW else None
)
COMMAND_SERIES_RAW = os.environ.get("GO1_STAIRS_EVAL_COMMAND_SERIES")
COMMAND_SERIES = (
    tuple(float(value) for value in COMMAND_SERIES_RAW.split(","))
    if COMMAND_SERIES_RAW else None
)
REFERENCE_CHECKPOINT = os.environ.get(
    "GO1_STAIRS_REFERENCE_CHECKPOINT"
)
if REFERENCE_CHECKPOINT is not None:
    REFERENCE_CHECKPOINT = os.path.realpath(REFERENCE_CHECKPOINT)
if SERIES_MODE:
    SERIES_CHECKPOINTS = tuple(
        os.path.realpath(path)
        for path in SERIES_CHECKPOINTS_RAW.split(",")
        if path
    )
    if not SERIES_CHECKPOINTS:
        raise RuntimeError("GO1_STAIRS_CHECKPOINT_SERIES is empty")
    if (
        ALPHA_SERIES is not None
        and len(ALPHA_SERIES) != len(SERIES_CHECKPOINTS)
    ):
        raise RuntimeError(
            "RMA alpha series must match checkpoint series length"
        )
PPO_ONLY = os.environ.get("GO1_STAIRS_PPO_ONLY", "0") == "1"
if PAIR_MODE:
    PAIR_CHECKPOINT = os.path.realpath(PAIR_CHECKPOINT)
OUTPUT_DIR = os.environ.get(
    "GO1_STAIRS_JOINT_EVAL_OUTPUT",
    "/tmp/go1_stairs_joint_eval",
)
SEED = int(os.environ["GO1_STAIRS_EVAL_SEED"])
STEP_HEIGHT = float(os.environ["GO1_STAIRS_EVAL_HEIGHT"])
EPISODES = int(os.environ.get("GO1_STAIRS_EVAL_EPISODES", "100"))
COMMAND_X = float(os.environ.get("GO1_STAIRS_EVAL_COMMAND_X", "0.6"))
SAVE_POSTURE_STATS = (
    os.environ.get("GO1_STAIRS_SAVE_POSTURE_STATS", "0") == "1"
)
if SERIES_MODE:
    group_names = os.environ.get("GO1_STAIRS_GROUP_NAMES")
    GROUPS = (
        tuple(group_names.split(","))
        if group_names else tuple(
            os.path.splitext(os.path.basename(path))[0]
            for path in SERIES_CHECKPOINTS
        )
    )
elif PPO_ONLY:
    GROUPS = ("model550",)
elif PAIR_MODE:
    GROUPS = ("model510", "model520")
else:
    GROUPS = ("A", "B", "C")
GROUP_SIZE = EPISODES
NUM_ENVS = GROUP_SIZE * len(GROUPS)
if len(set(GROUPS)) != len(GROUPS):
    raise RuntimeError("Evaluation group names must be unique")
if COMMAND_SERIES is not None and len(COMMAND_SERIES) != len(GROUPS):
    raise RuntimeError("Command series must match evaluation group count")


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
    cfg.commands.ranges.lin_vel_x = [COMMAND_X, COMMAND_X]
    cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]


def make_ppo(device):
    checkpoint = torch.load(PPO_CHECKPOINT, map_location=device)
    model = ActorCritic(
        235,
        235,
        12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_rma(device, alpha, checkpoint_path=JOINT_CHECKPOINT):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    residual_enabled = any(
        key.startswith("rear_hip_residual.")
        for key in checkpoint["model_state_dict"]
    )
    residual_independent = (
        residual_enabled
        and checkpoint["model_state_dict"][
            "rear_hip_residual.2.weight"
        ].shape[0] == 2
    )
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
        enable_rear_hip_residual=residual_enabled,
        rear_hip_residual_independent=residual_independent,
        rear_hip_residual_max_action=RESIDUAL_MAX_ACTION,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.set_rma_alpha(alpha)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pair_initial_state(env):
    device = env.device
    group_ids = [
        torch.arange(
            group_index * GROUP_SIZE,
            (group_index + 1) * GROUP_SIZE,
            device=device,
        )
        for group_index in range(len(GROUPS))
    ]
    source_ids = group_ids[0]
    target_groups = group_ids[1:]
    for target_ids in target_groups:
        env.root_states[target_ids] = env.root_states[source_ids]
        # env.dof_state is the raw flattened Isaac Gym tensor. Pair through
        # its [num_envs, num_dof] views so each target receives all 12 DOFs.
        env.dof_pos[target_ids] = env.dof_pos[source_ids]
        env.dof_vel[target_ids] = env.dof_vel[source_ids]

    all_ids = torch.arange(NUM_ENVS, device=device, dtype=torch.int32)
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

    if COMMAND_SERIES is None:
        commands = torch.zeros(GROUP_SIZE, env.commands.shape[1])
        commands[:, 0] = COMMAND_X
        commands = commands.to(device)
        for ids in group_ids:
            env.commands[ids] = commands
    else:
        for command_x, ids in zip(COMMAND_SERIES, group_ids):
            env.commands[ids].zero_()
            env.commands[ids, 0] = command_x

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
    # Heightfield sampling can differ by one discretized cell across Isaac
    # Gym environment handles even after root/DOF tensors are copied. The
    # paired protocol explicitly starts all policies from the exact same
    # policy observation and history.
    for target_ids in target_groups:
        if COMMAND_SERIES is None:
            env.obs_buf[target_ids] = env.obs_buf[source_ids]
        env.obs_history[target_ids] = env.obs_history[source_ids]

    initial_diff = {
        "root": max((
            (
                env.root_states[source_ids] - env.root_states[target]
            ).abs().max().item()
            for target in target_groups
        ), default=0.0),
        "dof": max((
            max(
                (
                    env.dof_pos[source_ids] - env.dof_pos[target]
                ).abs().max().item(),
                (
                    env.dof_vel[source_ids] - env.dof_vel[target]
                ).abs().max().item(),
            )
            for target in target_groups
        ), default=0.0),
        "command": max((
            (
                env.commands[source_ids] - env.commands[target]
            ).abs().max().item()
            for target in target_groups
        ), default=0.0),
        "observation": max((
            (
                env.obs_buf[source_ids] - env.obs_buf[target]
            ).abs().max().item()
            for target in target_groups
        ), default=0.0),
        "history": max((
            (
                env.obs_history[source_ids] - env.obs_history[target]
            ).abs().max().item()
            for target in target_groups
        ), default=0.0),
    }
    initial_state_hasher = hashlib.sha256()
    initial_state_hasher.update(
        env.root_states[source_ids].detach().cpu().numpy().tobytes()
    )
    initial_state_hasher.update(
        env.dof_pos[source_ids].detach().cpu().numpy().tobytes()
    )
    initial_state_hasher.update(
        env.dof_vel[source_ids].detach().cpu().numpy().tobytes()
    )
    return group_ids, initial_diff, initial_state_hasher.hexdigest()


def install_eval_callbacks(env, fixed_commands):
    def fixed_resample(self, env_ids):
        self.commands[env_ids] = fixed_commands[env_ids]

    def no_reset(self, env_ids):
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf.clone()

    env._resample_commands = types.MethodType(fixed_resample, env)
    env.reset_idx = types.MethodType(no_reset, env)


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


def group_slice(group):
    start = GROUPS.index(group) * GROUP_SIZE
    return slice(start, start + GROUP_SIZE)


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
    device = env.device
    # make_env constructs actors at the terrain origin; the explicit reset is
    # required to apply base_init_state (including the 0.32 m base height).
    env.reset()

    checkpoint_alpha = None
    if SERIES_MODE:
        series_alphas = (
            ALPHA_SERIES
            if ALPHA_SERIES is not None
            else (0.0,) * len(SERIES_CHECKPOINTS)
        )
        policies = tuple(
            make_rma(device, alpha, checkpoint_path)
            for checkpoint_path, alpha in zip(
                SERIES_CHECKPOINTS, series_alphas
            )
        )
        checkpoint_alpha = series_alphas
    elif PPO_ONLY:
        ppo = make_ppo(device)
    elif PAIR_MODE:
        joint_checkpoint = torch.load(
            JOINT_CHECKPOINT, map_location=device
        )
        checkpoint_alpha = float(joint_checkpoint["rma_alpha"])
        policies = (
            make_rma(device, 0.0, JOINT_CHECKPOINT),
            make_rma(device, 0.0, PAIR_CHECKPOINT),
        )
    else:
        joint_checkpoint = torch.load(
            JOINT_CHECKPOINT, map_location=device
        )
        checkpoint_alpha = float(joint_checkpoint["rma_alpha"])
        ppo = make_ppo(device)
        rma_off = make_rma(device, 0.0)
        rma_on = make_rma(device, checkpoint_alpha)
    reference_policy = (
        make_rma(device, 0.0, REFERENCE_CHECKPOINT)
        if REFERENCE_CHECKPOINT is not None else None
    )

    group_ids, initial_diff, initial_state_hash = pair_initial_state(env)
    fixed_commands = env.commands.clone()
    install_eval_callbacks(env, fixed_commands)

    if SERIES_MODE:
        actor_param_diff = None
        alpha_zero_internal_diff = None
    elif PPO_ONLY:
        actor_param_diff = 0.0
        alpha_zero_internal_diff = 0.0
    elif PAIR_MODE:
        actor_param_diff = max(
            (
                policies[0].state_dict()[key]
                - policies[1].state_dict()[key]
            ).abs().max().item()
            for key in policies[0].state_dict()
            if key.startswith("actor.")
        )
        with torch.inference_mode():
            first_actions = [
                policy.act_inference(
                    env.obs_buf[ids],
                    obs_history=env.obs_history[ids],
                )
                for policy, ids in zip(policies, group_ids)
            ]
        alpha_zero_internal_diff = (
            first_actions[0] - first_actions[1]
        ).abs().max().item()
    else:
        actor_param_diff = max(
            (
                rma_off.state_dict()[key] - rma_on.state_dict()[key]
            ).abs().max().item()
            for key in rma_off.state_dict()
            if key.startswith("actor.")
        )
        a_ids, b_ids, c_ids = group_ids
        with torch.inference_mode():
            b_initial = rma_off.act_inference(
                env.obs_buf[b_ids], obs_history=env.obs_history[b_ids]
            )
            c_base_initial = rma_on.compute_base_action_mean(
                env.obs_buf[c_ids]
            )
        alpha_zero_internal_diff = (
            b_initial - c_base_initial
        ).abs().max().item()

    active = torch.ones(NUM_ENVS, dtype=torch.bool, device=device)
    elapsed_steps = torch.zeros(
        NUM_ENVS, dtype=torch.long, device=device
    )
    reward_sum = torch.zeros(NUM_ENVS, device=device)
    fall = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
    success = torch.zeros_like(fall)
    summit_time = torch.full(
        (NUM_ENVS,), float("nan"), device=device
    )
    max_pitch = torch.zeros(NUM_ENVS, device=device)
    max_roll = torch.zeros(NUM_ENVS, device=device)
    max_foot_level = torch.zeros(
        NUM_ENVS, len(env.feet_indices),
        dtype=torch.long, device=device
    )
    max_swing_clearance = torch.zeros(
        NUM_ENVS, len(env.feet_indices), device=device
    )
    front_collision_count = torch.zeros(NUM_ENVS, device=device)
    calf_collision_count = torch.zeros(NUM_ENVS, device=device)
    foot_vertical_surface_hit_count = torch.zeros(
        NUM_ENVS, device=device
    )
    last_front_collision = torch.zeros(
        NUM_ENVS, len(env.front_foot_slots),
        dtype=torch.bool, device=device
    )
    last_calf_collision = torch.zeros(
        NUM_ENVS, len(env.calf_indices),
        dtype=torch.bool, device=device
    )
    rear_slip_episode = torch.zeros_like(fall)
    rear_contact_speed_sum = torch.zeros(
        NUM_ENVS, len(env.rear_foot_slots), device=device
    )
    rear_contact_steps = torch.zeros(
        NUM_ENVS, len(env.rear_foot_slots),
        dtype=torch.long, device=device
    )
    rear_slip_steps = torch.zeros_like(rear_contact_steps)
    rear_slide_distance = torch.zeros_like(
        rear_contact_speed_sum
    )
    rear_speed_samples = torch.full(
        (
            NUM_ENVS,
            int(env.max_episode_length) + 1,
            len(env.rear_foot_slots),
        ),
        float("nan"), device=device
    )

    env.gym.refresh_rigid_body_state_tensor(env.sim)
    initial_foot_pos = env.rigid_body_state[
        :, env.feet_indices, :3
    ].clone()
    last_rear_foot_pos = initial_foot_pos[
        :, env.rear_foot_slots
    ].clone()
    rear_swing_steps = torch.zeros(
        NUM_ENVS, len(env.rear_foot_slots),
        dtype=torch.long, device=device
    )
    rear_swing_forward_distance = torch.zeros(
        NUM_ENVS, len(env.rear_foot_slots), device=device
    )
    rear_follow_forward_distance = torch.zeros_like(
        rear_swing_forward_distance
    )
    initial_ground = env._sample_terrain_height_at(initial_foot_pos)
    terrain_parameters = {
        "step_height_m": STEP_HEIGHT,
        "tread_depth_m": 0.30,
        "step_count": 9,
        "friction": 1.0,
        "terrain_rows": int(env.cfg.terrain.num_rows),
        "terrain_cols": int(env.cfg.terrain.num_cols),
    }
    terrain_hasher = hashlib.sha256()
    terrain_hasher.update(
        json.dumps(terrain_parameters, sort_keys=True).encode("utf-8")
    )
    terrain_hasher.update(
        env.height_samples.detach().cpu().numpy().tobytes()
    )
    terrain_hash = terrain_hasher.hexdigest()
    initial_ground_pair_diff = max((
        (
            initial_ground[group_ids[0]] - initial_ground[target_ids]
        ).abs().max().item()
        for target_ids in group_ids[1:]
    ), default=0.0)
    print(json.dumps({
        "initial_root_z_min": env.root_states[:, 2].min().item(),
        "initial_root_z_max": env.root_states[:, 2].max().item(),
        "origin_z_min": env.env_origins[:, 2].min().item(),
        "origin_z_max": env.env_origins[:, 2].max().item(),
        "initial_ground_min": initial_ground.min().item(),
        "initial_ground_max": initial_ground.max().item(),
        "initial_base_contact_max": torch.norm(
            env.contact_forces[:, env.termination_contact_indices],
            dim=-1,
        ).max().item(),
    }))

    max_steps = int(env.max_episode_length)
    dt = env.dt
    obs = env.obs_buf
    action_abs_samples = torch.full(
        (max_steps + 1, NUM_ENVS, env.num_actions),
        float("nan"), device=device,
    )
    if SAVE_POSTURE_STATS:
        stair_dof_position_samples = torch.full(
            (max_steps + 1, NUM_ENVS, env.num_actions),
            float("nan"), device=device,
        )
        stair_pitch_samples = torch.full(
            (max_steps + 1, NUM_ENVS),
            float("nan"), device=device,
        )
        stair_roll_samples = torch.full_like(
            stair_pitch_samples, float("nan")
        )
        stair_foot_contact_samples = torch.full(
            (
                max_steps + 1,
                NUM_ENVS,
                len(env.feet_indices),
            ),
            float("nan"), device=device,
        )
        stair_front_width_samples = torch.full(
            (max_steps + 1, NUM_ENVS),
            float("nan"), device=device,
        )
        stair_rear_width_samples = torch.full_like(
            stair_front_width_samples, float("nan")
        )
    first_step_divergence = None
    reference_anchor_mse_sum = torch.zeros(NUM_ENVS, device=device)
    for step_index in range(max_steps + 1):
        if not active.any():
            break
        actions = torch.zeros(
            NUM_ENVS, env.num_actions, device=device
        )
        with torch.inference_mode():
            if SERIES_MODE:
                for policy, ids in zip(policies, group_ids):
                    actions[ids] = policy.act_inference(
                        obs[ids], obs_history=env.obs_history[ids]
                    )
                    if reference_policy is not None:
                        current_mean = policy.compute_base_action_mean(
                            obs[ids]
                        )
                        reference_mean = (
                            reference_policy.compute_base_action_mean(
                                obs[ids]
                            )
                        )
                        reference_anchor_mse_sum[ids] += (
                            current_mean - reference_mean
                        ).pow(2).mean(dim=-1) * active[ids]
            elif PPO_ONLY:
                ids = group_ids[0]
                actions[ids] = ppo.act_inference(obs[ids])
            elif PAIR_MODE:
                for policy, ids in zip(policies, group_ids):
                    actions[ids] = policy.act_inference(
                        obs[ids], obs_history=env.obs_history[ids]
                    )
            else:
                actions[a_ids] = ppo.act_inference(obs[a_ids])
                actions[b_ids] = rma_off.act_inference(
                    obs[b_ids], obs_history=env.obs_history[b_ids]
                )
                actions[c_ids] = rma_on.act_inference(
                    obs[c_ids], obs_history=env.obs_history[c_ids]
                )
        actions[~active] = 0.0
        was_active = active.clone()
        action_abs_samples[step_index, was_active] = (
            actions[was_active].abs()
        )
        obs, _, rewards, dones, _ = env.step(actions)
        if PAIR_MODE and step_index == 0:
            first_step_divergence = {
                "action": (
                    actions[group_ids[0]] - actions[group_ids[1]]
                ).abs().max().item(),
                "observation": (
                    obs[group_ids[0]] - obs[group_ids[1]]
                ).abs().max().item(),
                "history": (
                    env.obs_history[group_ids[0]]
                    - env.obs_history[group_ids[1]]
                ).abs().max().item(),
                "root": (
                    env.root_states[group_ids[0]]
                    - env.root_states[group_ids[1]]
                ).abs().max().item(),
                "dof": (
                    torch.maximum(
                        (
                            env.dof_pos[group_ids[0]]
                            - env.dof_pos[group_ids[1]]
                        ).abs().max(),
                        (
                            env.dof_vel[group_ids[0]]
                            - env.dof_vel[group_ids[1]]
                        ).abs().max(),
                    ).item()
                ),
            }
        elapsed_steps[was_active] += 1
        reward_sum[was_active] += rewards[was_active]

        env.gym.refresh_rigid_body_state_tensor(env.sim)
        foot_state = env.rigid_body_state[:, env.feet_indices]
        foot_pos = foot_state[..., :3]
        foot_vel = foot_state[..., 7:10]
        if SAVE_POSTURE_STATS:
            foot_offset = foot_pos - env.root_states[:, None, :3]
            base_quat = env.root_states[:, None, 3:7].expand(
                -1, len(env.feet_indices), -1
            ).reshape(-1, 4)
            foot_pos_base = quat_rotate_inverse(
                base_quat, foot_offset.reshape(-1, 3)
            ).reshape(NUM_ENVS, len(env.feet_indices), 3)
            front_width = torch.abs(
                foot_pos_base[:, env.front_foot_slots[0], 1]
                - foot_pos_base[:, env.front_foot_slots[1], 1]
            )
            rear_width = torch.abs(
                foot_pos_base[:, env.rear_foot_slots[0], 1]
                - foot_pos_base[:, env.rear_foot_slots[1], 1]
            )
        terrain_height = env._sample_terrain_height_at(foot_pos)
        foot_contact = (
            env.contact_forces[:, env.feet_indices, 2] > 1.0
        )
        swing = ~foot_contact & was_active.unsqueeze(1)
        foot_clearance = foot_pos[..., 2] - terrain_height
        max_swing_clearance = torch.where(
            swing,
            torch.maximum(max_swing_clearance, foot_clearance),
            max_swing_clearance,
        )
        height_level = torch.clamp(
            torch.round(
                (terrain_height - initial_ground) / STEP_HEIGHT
            ).long(),
            0,
            env.cfg.terrain.stair_step_count,
        )
        contacted_level = torch.where(
            foot_contact, height_level,
            torch.zeros_like(height_level),
        )
        max_foot_level = torch.maximum(
            max_foot_level, contacted_level
        )
        rear_foot_pos = foot_pos[:, env.rear_foot_slots]
        rear_forward_delta = torch.clamp(
            rear_foot_pos[..., 0] - last_rear_foot_pos[..., 0],
            min=0.0,
        )
        metric_active = was_active & ~dones.bool()
        rear_swing = (
            ~foot_contact[:, env.rear_foot_slots]
            & metric_active.unsqueeze(1)
        )
        rear_swing_steps += rear_swing.long()
        rear_swing_forward_distance += (
            rear_forward_delta * rear_swing.float()
        )
        current_front_level = torch.min(
            max_foot_level[:, env.front_foot_slots], dim=1
        ).values
        current_rear_levels = max_foot_level[
            :, env.rear_foot_slots
        ]
        rear_following = (
            current_front_level.unsqueeze(1) > current_rear_levels
        )
        rear_follow_forward_distance += (
            rear_forward_delta
            * rear_swing.float()
            * rear_following.float()
        )
        last_rear_foot_pos[:] = rear_foot_pos

        foot_force = env.contact_forces[:, env.feet_indices]
        foot_vertical_surface_hit = torch.any(
            torch.norm(foot_force[..., :2], dim=-1)
            > 5.0 * torch.abs(foot_force[..., 2]),
            dim=1,
        )
        foot_vertical_surface_hit_count += (
            foot_vertical_surface_hit.float() * was_active
        )

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
            torch.norm(
                env.contact_forces[:, env.calf_indices], dim=-1
            ) > 5.0
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
        active_rear_contact = rear_contact & was_active.unsqueeze(1)
        active_rear_slip = active_rear_contact & (
            rear_speed > env.cfg.rewards.rear_slip_velocity_threshold
        )
        for foot_slot in range(len(env.rear_foot_slots)):
            valid = (
                active_rear_contact[:, foot_slot]
                & (
                    rear_contact_steps[:, foot_slot]
                    < rear_speed_samples.shape[1]
                )
            )
            valid_ids = torch.nonzero(
                valid, as_tuple=False
            ).squeeze(-1)
            if valid_ids.numel() > 0:
                indices = rear_contact_steps[valid_ids, foot_slot]
                rear_speed_samples[
                    valid_ids, indices, foot_slot
                ] = rear_speed[valid_ids, foot_slot]
        contact_float = active_rear_contact.float()
        rear_contact_speed_sum += rear_speed * contact_float
        rear_contact_steps += active_rear_contact.long()
        rear_slide_distance += rear_speed * contact_float * dt
        rear_slip_steps += active_rear_slip.long()
        rear_slip_episode |= (
            torch.any(active_rear_slip, dim=1)
            & was_active
        )
        roll, pitch = root_roll_pitch(env.root_states[:, 3:7])
        max_roll = torch.maximum(max_roll, roll.abs() * was_active)
        max_pitch = torch.maximum(max_pitch, pitch.abs() * was_active)
        if SAVE_POSTURE_STATS:
            on_stairs = (
                torch.any(max_foot_level > 0, dim=1)
                & was_active
            )
            stair_dof_position_samples[
                step_index, on_stairs
            ] = env.dof_pos[on_stairs]
            stair_pitch_samples[
                step_index, on_stairs
            ] = pitch[on_stairs]
            stair_roll_samples[
                step_index, on_stairs
            ] = roll[on_stairs]
            stair_foot_contact_samples[
                step_index, on_stairs
            ] = foot_contact[on_stairs].float()
            stair_front_width_samples[
                step_index, on_stairs
            ] = front_width[on_stairs]
            stair_rear_width_samples[
                step_index, on_stairs
            ] = rear_width[on_stairs]

        reached_top = torch.all(
            max_foot_level >= env.cfg.terrain.stair_step_count,
            dim=1,
        )
        new_success = active & reached_top
        success[new_success] = True
        summit_time[new_success] = elapsed_steps[new_success] * dt

        new_done = active & dones.bool()
        fall[new_done] = ~env.time_out_buf[new_done]
        active &= ~(new_success | new_done)

    passed_steps = torch.min(max_foot_level, dim=1).values
    front_steps = torch.min(
        max_foot_level[:, env.front_foot_slots], dim=1
    ).values
    rear_steps = torch.min(
        max_foot_level[:, env.rear_foot_slots], dim=1
    ).values
    rear_failure = (~success) & (front_steps > rear_steps)
    safe_rear_contact_steps = torch.clamp(rear_contact_steps, min=1)
    rear_contact_mean_speed = (
        rear_contact_speed_sum / safe_rear_contact_steps
    )
    sorted_rear_speeds = torch.sort(
        rear_speed_samples.nan_to_num(nan=float("inf")), dim=1
    ).values
    rear_p95_indices = torch.clamp(
        torch.ceil(0.95 * rear_contact_steps.float()).long() - 1,
        min=0,
        max=sorted_rear_speeds.shape[1] - 1,
    )
    rear_contact_speed_p95 = torch.gather(
        sorted_rear_speeds, 1, rear_p95_indices.unsqueeze(1)
    ).squeeze(1)
    rear_contact_speed_p95 = torch.where(
        rear_contact_steps > 0,
        rear_contact_speed_p95,
        torch.zeros_like(rear_contact_speed_p95),
    )
    rear_slip_duration = rear_slip_steps.float() * dt
    rear_contact_duration = rear_contact_steps.float() * dt
    rear_slip_time_ratio = rear_slip_duration / torch.clamp(
        rear_contact_duration, min=dt
    )

    records = []
    for group in GROUPS:
        ids = torch.arange(
            GROUPS.index(group) * GROUP_SIZE,
            (GROUPS.index(group) + 1) * GROUP_SIZE,
            device=device,
        )
        for replicate, env_id in enumerate(ids.tolist()):
            env_action_abs = action_abs_samples[:, env_id].reshape(-1)
            env_action_abs = env_action_abs[
                torch.isfinite(env_action_abs)
            ]
            record = {
                "seed": SEED,
                "height_m": STEP_HEIGHT,
                "group": group,
                "replicate": replicate,
                "success": bool(success[env_id].item()),
                "passed_steps": int(passed_steps[env_id].item()),
                "max_passed_height_m": (
                    int(passed_steps[env_id].item()) * STEP_HEIGHT
                ),
                "summit_time_s": (
                    None if torch.isnan(summit_time[env_id])
                    else float(summit_time[env_id].item())
                ),
                "fall": bool(fall[env_id].item()),
                "rear_leg_failure": bool(
                    rear_failure[env_id].item()
                ),
                "rear_slip": bool(rear_slip_episode[env_id].item()),
                "rear_contact_mean_speed": (
                    rear_contact_mean_speed[env_id].tolist()
                ),
                "rear_contact_speed_p95": (
                    rear_contact_speed_p95[env_id].tolist()
                ),
                "rear_slide_distance": (
                    rear_slide_distance[env_id].tolist()
                ),
                "rear_slip_duration": (
                    rear_slip_duration[env_id].tolist()
                ),
                "rear_slip_time_ratio": (
                    rear_slip_time_ratio[env_id].tolist()
                ),
                "rear_swing_time_s": (
                    rear_swing_steps[env_id].float() * dt
                ).tolist(),
                "rear_swing_forward_distance_m": (
                    rear_swing_forward_distance[env_id].tolist()
                ),
                "rear_follow_forward_distance_m": (
                    rear_follow_forward_distance[env_id].tolist()
                ),
                "foot_top_landing_ratio": (
                    max_foot_level[env_id].float()
                    / env.cfg.terrain.stair_step_count
                ).tolist(),
                "swing_max_clearance_m": (
                    max_swing_clearance[env_id].tolist()
                ),
                "front_collision_count": float(
                    front_collision_count[env_id].item()
                ),
                "calf_collision_count": float(
                    calf_collision_count[env_id].item()
                ),
                "limb_penetration_count": float(
                    env.stair_soft_limb_penetration_count[
                        env_id
                    ].item()
                ),
                "body_penetration_count": float(
                    env.stair_soft_body_penetration_count[
                        env_id
                    ].item()
                ),
                "limb_penetration_mean_depth_m": float((
                    env.stair_soft_limb_penetration_depth_sum[env_id]
                    / torch.clamp(
                        env.stair_soft_limb_penetration_samples[env_id],
                        min=1.0,
                    )
                ).item()),
                "body_penetration_mean_depth_m": float((
                    env.stair_soft_body_penetration_depth_sum[env_id]
                    / torch.clamp(
                        env.stair_soft_body_penetration_samples[env_id],
                        min=1.0,
                    )
                ).item()),
                "limb_penetration_max_depth_m": float(
                    env.stair_soft_limb_penetration_max_depth[
                        env_id
                    ].item()
                ),
                "body_penetration_max_depth_m": float(
                    env.stair_soft_body_penetration_max_depth[
                        env_id
                    ].item()
                ),
                "foot_vertical_surface_hit_count": float(
                    foot_vertical_surface_hit_count[env_id].item()
                ),
                "max_pitch_rad": float(max_pitch[env_id].item()),
                "max_roll_rad": float(max_roll[env_id].item()),
                "reward": float(reward_sum[env_id].item()),
                "episode_length": int(elapsed_steps[env_id].item()),
                "action_p99": float(
                    torch.quantile(env_action_abs, 0.99).item()
                ),
                "action_max": float(env_action_abs.max().item()),
                "reference_anchor_mse": float(
                    (
                        reference_anchor_mse_sum[env_id]
                        / torch.clamp(elapsed_steps[env_id], min=1)
                    ).item()
                ),
            }
            if SAVE_POSTURE_STATS:
                dof_samples = stair_dof_position_samples[:, env_id]
                valid_posture = torch.isfinite(
                    dof_samples[:, 0]
                )
                dof_samples = torch.rad2deg(
                    dof_samples[valid_posture]
                )
                pitch_samples = torch.rad2deg(
                    stair_pitch_samples[valid_posture, env_id]
                )
                roll_samples = torch.rad2deg(
                    stair_roll_samples[valid_posture, env_id]
                )
                contact_samples = stair_foot_contact_samples[
                    valid_posture, env_id
                ]
                front_width_samples = stair_front_width_samples[
                    valid_posture, env_id
                ]
                rear_width_samples = stair_rear_width_samples[
                    valid_posture, env_id
                ]
                if dof_samples.numel() > 0:
                    record.update({
                        "stair_joint_angle_deg_p05": torch.quantile(
                            dof_samples, 0.05, dim=0
                        ).tolist(),
                        "stair_joint_angle_deg_p50": torch.quantile(
                            dof_samples, 0.50, dim=0
                        ).tolist(),
                        "stair_joint_angle_deg_p95": torch.quantile(
                            dof_samples, 0.95, dim=0
                        ).tolist(),
                        "stair_pitch_deg_p05_p50_p95": [
                            float(torch.quantile(
                                pitch_samples, quantile
                            ).item())
                            for quantile in (0.05, 0.50, 0.95)
                        ],
                        "stair_roll_deg_p05_p50_p95": [
                            float(torch.quantile(
                                roll_samples, quantile
                            ).item())
                            for quantile in (0.05, 0.50, 0.95)
                        ],
                        "stair_foot_contact_ratio": (
                            contact_samples.mean(dim=0).tolist()
                        ),
                        "stair_front_width_m_p05_p50_p95": [
                            float(torch.quantile(
                                front_width_samples, quantile
                            ).item())
                            for quantile in (0.05, 0.50, 0.95)
                        ],
                        "stair_rear_width_m_p05_p50_p95": [
                            float(torch.quantile(
                                rear_width_samples, quantile
                            ).item())
                            for quantile in (0.05, 0.50, 0.95)
                        ],
                        "stair_rear_width_below_0p12_ratio": float(
                            (rear_width_samples < 0.12)
                            .float().mean().item()
                        ),
                        "stair_posture_sample_count": int(
                            dof_samples.shape[0]
                        ),
                    })
            records.append(record)

    with open(PPO_CHECKPOINT, "rb") as checkpoint_stream:
        checkpoint_sha256 = hashlib.sha256(
            checkpoint_stream.read()
        ).hexdigest()
    output = {
        "seed": SEED,
        "step_height_m": STEP_HEIGHT,
        "tread_depth_m": 0.30,
        "step_count": 9,
        "friction": 1.0,
        "command_x_mps": COMMAND_X,
        "command_x_series_mps": COMMAND_SERIES,
        "command_y_mps": 0.0,
        "command_yaw_radps": 0.0,
        "base_mass_kg": 5.205080032348633,
        "deterministic_act_inference": True,
        "training_updates": False,
        "posture_stats_enabled": SAVE_POSTURE_STATS,
        "clip_actions": 100.0,
        "ppo_checkpoint": PPO_CHECKPOINT,
        "ppo_checkpoint_sha256": checkpoint_sha256,
        "joint_checkpoint": JOINT_CHECKPOINT,
        "pair_checkpoint": PAIR_CHECKPOINT,
        "series_checkpoints": (
            SERIES_CHECKPOINTS if SERIES_MODE else None
        ),
        "series_alphas": (
            series_alphas if SERIES_MODE else None
        ),
        "reference_checkpoint": REFERENCE_CHECKPOINT,
        "checkpoint_alpha": checkpoint_alpha,
        "initial_triplet_max_abs_diff": initial_diff,
        "initial_state_hash": initial_state_hash,
        "initial_ground_pair_max_abs_diff": initial_ground_pair_diff,
        "terrain_parameters": terrain_parameters,
        "terrain_hash": terrain_hash,
        "first_step_divergence": first_step_divergence,
        "joint_actor_bc_max_abs_diff": actor_param_diff,
        "alpha_zero_internal_action_diff": alpha_zero_internal_diff,
        "records": records,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    height_tag = f"{STEP_HEIGHT:.3f}".replace(".", "p")
    output_path = os.path.join(
        OUTPUT_DIR, f"height{height_tag}_seed{SEED}.json"
    )
    with open(output_path, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
    print(json.dumps({
        "output": output_path,
        "seed": SEED,
        "height": STEP_HEIGHT,
        "initial_max_diff": max(initial_diff.values()),
    }))
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
