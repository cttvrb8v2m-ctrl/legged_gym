"""Deterministic model_800 stair failure diagnosis (inference only)."""

import hashlib
import json
import os
import random

import isaacgym
from isaacgym.torch_utils import quat_rotate_inverse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import eval_stairs_joint as common
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


MODEL_800 = os.path.realpath(
    "/home/gjt/legged_gym/logs/rough_go1/"
    "Jul27_09-38-10_stairs_high_300_resume_mid500/model_800.pt"
)
OUTPUT_DIR = os.path.realpath(os.environ[
    "GO1_STAIRS_FAILURE_OUTPUT"
])
PITCH_THRESHOLD = 0.55
SEVERE_COLLISION_FORCE = 100.0
TORQUE_RATIO_THRESHOLD = 0.95
TORQUE_DURATION_S = 0.10


def max_consecutive(mask):
    best = 0
    current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def ordered_window(array, end_index, valid_steps):
    count = min(valid_steps, array.shape[0])
    indices = [
        (end_index - count + 1 + offset) % array.shape[0]
        for offset in range(count)
    ]
    return array[indices]


def classify_failure(
    trace, front_steps, rear_steps, fell, timed_out, dt, slip_threshold
):
    if front_steps > rear_steps:
        return "rear_not_clear"

    rear_top_contact = (
        trace["rear_contact"].astype(bool)
        & (trace["foot_levels"][:, 2:4] > 0)
    )
    rear_slip = rear_top_contact & (
        trace["rear_slip_speed"] > slip_threshold
    )
    required = int(np.ceil(0.10 / dt))
    slip_sustained = any(
        max_consecutive(rear_slip[:, foot].tolist()) >= required
        for foot in range(rear_slip.shape[1])
    )
    if fell and slip_sustained:
        return "rear_slip"

    torque_sustained = any(
        max_consecutive(
            (trace["rear_torque_ratio"][:, joint]
             > TORQUE_RATIO_THRESHOLD).tolist()
        ) >= int(np.ceil(TORQUE_DURATION_S / dt))
        for joint in range(trace["rear_torque_ratio"].shape[1])
    )
    if torque_sustained:
        return "torque_saturation"

    severe_force = max(
        np.max(np.linalg.norm(trace["base_force"], axis=-1), initial=0.0),
        np.max(np.linalg.norm(trace["thigh_force"], axis=-1), initial=0.0),
        np.max(np.linalg.norm(trace["calf_force"], axis=-1), initial=0.0),
    )
    if fell and (
        np.max(np.abs(trace["base_pitch"]), initial=0.0) > PITCH_THRESHOLD
        or severe_force > SEVERE_COLLISION_FORCE
    ):
        return "pitch_or_body_collision"
    if timed_out:
        return "timeout"
    return "other"


def plot_trace(trace, title, output_path, dt, rear_joint_names):
    time_axis = np.arange(-len(trace["base_pitch"]) + 1, 1) * dt
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    foot_names = ("FL", "FR", "RL", "RR")
    for index, name in enumerate(foot_names):
        axes[0].plot(
            time_axis, trace["foot_levels"][:, index], label=name
        )
    axes[0].set_ylabel("stair level")
    axes[0].legend(ncol=4)

    for index, name in enumerate(("RL", "RR")):
        axes[1].plot(
            time_axis, trace["rear_slip_speed"][:, index],
            label=f"{name} slip speed"
        )
        contact = trace["rear_contact"][:, index].astype(float)
        axes[1].fill_between(
            time_axis, 0, contact * 0.08, alpha=0.15
        )
    axes[1].axhline(0.20, color="red", linestyle="--")
    axes[1].set_ylabel("m/s")
    axes[1].legend(ncol=2)

    for index, name in enumerate(rear_joint_names):
        axes[2].plot(
            time_axis, trace["rear_torque_ratio"][:, index],
            label=name
        )
    axes[2].axhline(0.95, color="red", linestyle="--")
    axes[2].set_ylabel("|torque| / limit")
    axes[2].legend(ncol=3, fontsize=7)

    axes[3].plot(
        time_axis, trace["base_pitch"], label="base pitch (rad)"
    )
    axes[3].plot(
        time_axis, trace["base_height"], label="base z (m)"
    )
    for key, label in (
        ("feet_force", "feet force"),
        ("calf_force", "calf force"),
        ("thigh_force", "thigh force"),
        ("base_force", "base force"),
    ):
        force_norm = np.linalg.norm(trace[key], axis=-1)
        axes[3].plot(
            time_axis, force_norm.max(axis=1), label=label, alpha=0.7
        )
    axes[3].axhline(PITCH_THRESHOLD, color="orange", linestyle=":")
    axes[3].axhline(
        SEVERE_COLLISION_FORCE, color="red", linestyle="--"
    )
    axes[3].set_xlabel("time before failure (s)")
    axes[3].legend(ncol=3, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main():
    seed = common.SEED
    height = common.STEP_HEIGHT
    num_envs = common.EPISODES
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = get_args()
    env_cfg, _ = task_registry.get_cfgs("go1_stairs_joint")
    common.configure_environment(env_cfg)
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = seed
    env, _ = task_registry.make_env(
        name="go1_stairs_joint", args=args, env_cfg=env_cfg
    )
    env.reset()
    device = env.device
    body_names = env.gym.get_actor_rigid_body_names(
        env.envs[0], env.actor_handles[0]
    )
    thigh_indices = torch.tensor(
        [
            env.gym.find_actor_rigid_body_handle(
                env.envs[0], env.actor_handles[0], name
            )
            for name in body_names if name.endswith("_thigh")
        ],
        device=device,
        dtype=torch.long,
    )
    policy = common.make_rma(device, 0.0, MODEL_800)

    group_ids, initial_diff, initial_state_hash = (
        common.pair_initial_state(env)
    )
    fixed_commands = env.commands.clone()
    common.install_eval_callbacks(env, fixed_commands)
    ids = group_ids[0]

    env.gym.refresh_rigid_body_state_tensor(env.sim)
    initial_foot_pos = env.rigid_body_state[
        :, env.feet_indices, :3
    ].clone()
    initial_ground = env._sample_terrain_height_at(initial_foot_pos)

    dt = env.dt
    window_steps = int(round(0.5 / dt))
    max_steps = int(env.max_episode_length)
    rear_joint_id_list = [
        index for index, name in enumerate(env.dof_names)
        if name.startswith("RL_") or name.startswith("RR_")
    ]
    rear_joint_ids = torch.tensor(
        rear_joint_id_list, device=device, dtype=torch.long
    )
    rear_joint_names = [
        env.dof_names[index] for index in rear_joint_id_list
    ]

    shapes = {
        "foot_levels": (window_steps, num_envs, 4),
        "rear_pos": (window_steps, num_envs, 2, 3),
        "rear_vel": (window_steps, num_envs, 2, 3),
        "rear_contact": (window_steps, num_envs, 2),
        "rear_slip_speed": (window_steps, num_envs, 2),
        "torque": (window_steps, num_envs, env.num_dof),
        "torque_ratio": (window_steps, num_envs, env.num_dof),
        "base_pitch": (window_steps, num_envs),
        "base_height": (window_steps, num_envs),
        "feet_force": (window_steps, num_envs, 4, 3),
        "calf_force": (window_steps, num_envs, 4, 3),
        "thigh_force": (window_steps, num_envs, 4, 3),
        "base_force": (
            window_steps, num_envs, len(env.base_indices), 3
        ),
    }
    ring = {
        key: torch.full(
            shape, float("nan"), device=device
        ) for key, shape in shapes.items()
    }
    # LeggedRobot resets terminated environments inside env.step(). Capture
    # the terminal simulator tensors immediately before that reset so the
    # final sample in each failure trace is the true failure state.
    terminal_snapshot = {}
    original_reset_idx = env.reset_idx

    def recording_reset_idx(env_ids):
        if len(env_ids) > 0:
            terminal_foot_state = env.rigid_body_state[
                env_ids[:, None], env.feet_indices[None, :]
            ]
            terminal_foot_pos = terminal_foot_state[..., :3]
            terminal_foot_vel = terminal_foot_state[..., 7:10]
            terminal_ground = env._sample_terrain_height_at(
                terminal_foot_pos
            )
            terminal_levels = torch.clamp(
                torch.round(
                    (
                        terminal_ground
                        - initial_ground[env_ids]
                    ) / height
                ).long(),
                0,
                env.cfg.terrain.stair_step_count,
            )
            terminal_contact = (
                env.contact_forces[
                    env_ids[:, None], env.feet_indices[None, :], 2
                ] > 1.0
            )
            _, terminal_pitch = common.root_roll_pitch(
                env.root_states[env_ids, 3:7]
            )
            terminal_snapshot.clear()
            terminal_snapshot["ids"] = env_ids.clone()
            terminal_snapshot["foot_contact"] = terminal_contact.clone()
            terminal_snapshot["values"] = {
                "foot_levels": terminal_levels.float().clone(),
                "rear_pos": terminal_foot_pos[
                    :, env.rear_foot_slots
                ].clone(),
                "rear_vel": terminal_foot_vel[
                    :, env.rear_foot_slots
                ].clone(),
                "rear_contact": terminal_contact[
                    :, env.rear_foot_slots
                ].float().clone(),
                "rear_slip_speed": torch.norm(
                    terminal_foot_vel[
                        :, env.rear_foot_slots, :2
                    ],
                    dim=-1,
                ).clone(),
                "torque": env.torques[env_ids].clone(),
                "torque_ratio": (
                    torch.abs(env.torques[env_ids])
                    / torch.clamp(env.torque_limits, min=1e-6)
                ).clone(),
                "base_pitch": terminal_pitch.clone(),
                "base_height": env.root_states[env_ids, 2].clone(),
                "feet_force": env.contact_forces[
                    env_ids[:, None], env.feet_indices[None, :]
                ].clone(),
                "calf_force": env.contact_forces[
                    env_ids[:, None], env.calf_indices[None, :]
                ].clone(),
                "thigh_force": env.contact_forces[
                    env_ids[:, None], thigh_indices[None, :]
                ].clone(),
                "base_force": env.contact_forces[
                    env_ids[:, None], env.base_indices[None, :]
                ].clone(),
            }
        original_reset_idx(env_ids)

    env.reset_idx = recording_reset_idx
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    elapsed = torch.zeros(num_envs, dtype=torch.long, device=device)
    last_ring_index = torch.zeros(
        num_envs, dtype=torch.long, device=device
    )
    success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros_like(success)
    max_foot_level = torch.zeros(
        num_envs, 4, dtype=torch.long, device=device
    )
    obs = env.obs_buf

    for step_index in range(max_steps + 1):
        if not active.any():
            break
        with torch.inference_mode():
            actions = policy.act_inference(
                obs, obs_history=env.obs_history
            )
        actions = torch.where(
            active.unsqueeze(1), actions, torch.zeros_like(actions)
        )
        was_active = active.clone()
        obs, _, _, dones, _ = env.step(actions)
        elapsed[was_active] += 1

        env.gym.refresh_rigid_body_state_tensor(env.sim)
        foot_state = env.rigid_body_state[:, env.feet_indices]
        foot_pos = foot_state[..., :3]
        foot_vel = foot_state[..., 7:10]
        terrain_height = env._sample_terrain_height_at(foot_pos)
        foot_contact = (
            env.contact_forces[:, env.feet_indices, 2] > 1.0
        )
        levels = torch.clamp(
            torch.round(
                (terrain_height - initial_ground) / height
            ).long(), 0, env.cfg.terrain.stair_step_count
        )
        rear_contact = foot_contact[:, env.rear_foot_slots]
        rear_vel = foot_vel[:, env.rear_foot_slots]
        rear_speed = torch.norm(rear_vel[..., :2], dim=-1)
        _, pitch = common.root_roll_pitch(env.root_states[:, 3:7])
        torque_ratio = torch.abs(env.torques) / torch.clamp(
            env.torque_limits, min=1e-6
        )

        position = step_index % window_steps
        active_ids = torch.nonzero(
            was_active, as_tuple=False
        ).squeeze(-1)
        values = {
            "foot_levels": levels.float(),
            "rear_pos": foot_pos[:, env.rear_foot_slots],
            "rear_vel": rear_vel,
            "rear_contact": rear_contact.float(),
            "rear_slip_speed": rear_speed,
            "torque": env.torques,
            "torque_ratio": torque_ratio,
            "base_pitch": pitch,
            "base_height": env.root_states[:, 2],
            "feet_force": env.contact_forces[:, env.feet_indices],
            "calf_force": env.contact_forces[:, env.calf_indices],
            "thigh_force": env.contact_forces[:, thigh_indices],
            "base_force": env.contact_forces[:, env.base_indices],
        }
        if terminal_snapshot:
            terminal_ids = terminal_snapshot["ids"]
            foot_contact[terminal_ids] = terminal_snapshot[
                "foot_contact"
            ]
            for key, terminal_value in terminal_snapshot[
                "values"
            ].items():
                values[key][terminal_ids] = terminal_value
            terminal_snapshot.clear()
        levels = values["foot_levels"].long()
        max_foot_level = torch.maximum(
            max_foot_level,
            torch.where(
                foot_contact, levels, torch.zeros_like(levels)
            ),
        )
        for key, value in values.items():
            ring[key][position, active_ids] = value[active_ids]
        last_ring_index[active_ids] = position

        reached_top = torch.all(
            max_foot_level >= env.cfg.terrain.stair_step_count, dim=1
        )
        new_success = active & reached_top
        success[new_success] = True
        new_done = active & dones.bool()
        fell[new_done] = ~env.time_out_buf[new_done]
        active &= ~(new_success | new_done)

    front_steps = torch.min(
        max_foot_level[:, env.front_foot_slots], dim=1
    ).values.cpu().numpy()
    rear_steps = torch.min(
        max_foot_level[:, env.rear_foot_slots], dim=1
    ).values.cpu().numpy()
    success_np = success.cpu().numpy()
    fell_np = fell.cpu().numpy()
    elapsed_np = elapsed.cpu().numpy()
    last_np = last_ring_index.cpu().numpy()
    ring_cpu = {
        key: value.cpu().numpy() for key, value in ring.items()
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trace_dir = os.path.join(OUTPUT_DIR, "failure_traces")
    plot_dir = os.path.join(OUTPUT_DIR, "failure_plots")
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    records = []
    traces = {}
    for env_id in range(num_envs):
        timed_out = bool(
            not success_np[env_id] and not fell_np[env_id]
        )
        trace = {
            key: ordered_window(
                value[:, env_id],
                int(last_np[env_id]),
                int(elapsed_np[env_id]),
            )
            for key, value in ring_cpu.items()
        }
        trace["rear_torque_ratio"] = trace["torque_ratio"][
            :, rear_joint_ids.cpu().numpy()
        ]
        reason = None
        trace_path = None
        if not success_np[env_id]:
            reason = classify_failure(
                trace,
                int(front_steps[env_id]),
                int(rear_steps[env_id]),
                bool(fell_np[env_id]),
                timed_out,
                dt,
                env.cfg.rewards.rear_slip_velocity_threshold,
            )
            trace_path = os.path.join(
                trace_dir, f"episode_{env_id:03d}_{reason}.npz"
            )
            np.savez_compressed(
                trace_path,
                **trace,
                rear_joint_names=np.asarray(rear_joint_names),
            )
            traces[env_id] = trace
        records.append({
            "episode": env_id,
            "success": bool(success_np[env_id]),
            "fall": bool(fell_np[env_id]),
            "timeout": timed_out,
            "front_steps": int(front_steps[env_id]),
            "rear_steps": int(rear_steps[env_id]),
            "passed_steps": int(min(
                front_steps[env_id], rear_steps[env_id]
            )),
            "failure_reason": reason,
            "trace_path": trace_path,
        })

    categories = (
        "rear_not_clear", "rear_slip", "torque_saturation",
        "pitch_or_body_collision", "timeout", "other",
    )
    rng = random.Random(seed + int(round(height * 1000)))
    plot_paths = {category: [] for category in categories}
    for category in categories:
        candidates = [
            record["episode"] for record in records
            if record["failure_reason"] == category
        ]
        selected = rng.sample(candidates, min(5, len(candidates)))
        for env_id in selected:
            plot_path = os.path.join(
                plot_dir, f"{category}_episode_{env_id:03d}.png"
            )
            plot_trace(
                traces[env_id],
                f"{category} | h={height:.3f} m | episode={env_id}",
                plot_path,
                dt,
                rear_joint_names,
            )
            plot_paths[category].append(plot_path)

    failure_count = sum(not record["success"] for record in records)
    counts = {
        category: sum(
            record["failure_reason"] == category for record in records
        )
        for category in categories
    }
    with open(MODEL_800, "rb") as stream:
        checkpoint_sha256 = hashlib.sha256(stream.read()).hexdigest()
    output = {
        "checkpoint": MODEL_800,
        "checkpoint_sha256": checkpoint_sha256,
        "training_updates": False,
        "deterministic": True,
        "alpha": 0.0,
        "seed": seed,
        "height_m": height,
        "command_x_mps": 0.6,
        "episodes": num_envs,
        "success_count": int(success_np.sum()),
        "failure_count": failure_count,
        "failure_counts": counts,
        "failure_percent_of_failures": {
            key: 100.0 * value / max(failure_count, 1)
            for key, value in counts.items()
        },
        "thresholds": {
            "rear_slip_speed_mps": (
                env.cfg.rewards.rear_slip_velocity_threshold
            ),
            "torque_ratio": TORQUE_RATIO_THRESHOLD,
            "torque_duration_s": TORQUE_DURATION_S,
            "pitch_rad": PITCH_THRESHOLD,
            "severe_collision_force_n": SEVERE_COLLISION_FORCE,
            "history_window_s": 0.5,
        },
        "initial_state_hash": initial_state_hash,
        "initial_diff": initial_diff,
        "plot_paths": plot_paths,
        "records": records,
    }
    output_path = os.path.join(
        OUTPUT_DIR,
        f"failure_summary_h{height:.3f}_seed{seed}.json",
    )
    with open(output_path, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
    print(json.dumps({
        "output": output_path,
        "success": int(success_np.sum()),
        "failures": failure_count,
        "failure_counts": counts,
    }))
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
