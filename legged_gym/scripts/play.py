# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import time
from collections import deque
import numpy as np
import tkinter as tk

# The Isaac Gym examples tree also ships an rsl_rl package.  Play must use
# this repository's runner because its checkpoint contract includes the Go1
# residual modules.
if LEGGED_GYM_ROOT_DIR in sys.path:
    sys.path.remove(LEGGED_GYM_ROOT_DIR)
sys.path.insert(0, LEGGED_GYM_ROOT_DIR)

import isaacgym
from isaacgym import gymapi

import torch
from legged_gym.envs import *
from legged_gym.utils import (
    Logger,
    export_policy_as_jit,
    get_args,
    get_load_path,
    task_registry,
)


def configure_policy_from_checkpoint(train_cfg, args):
    """Match optional policy modules to the checkpoint before runner creation."""
    experiment_name = (
        args.experiment_name
        if args.experiment_name is not None
        else train_cfg.runner.experiment_name
    )
    load_run = (
        args.load_run
        if args.load_run is not None
        else train_cfg.runner.load_run
    )
    checkpoint_index = (
        args.checkpoint
        if args.checkpoint is not None
        else train_cfg.runner.checkpoint
    )
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", experiment_name
    )
    checkpoint_path = get_load_path(
        log_root,
        load_run=load_run,
        checkpoint=checkpoint_index,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    residual_enabled = any(
        key.startswith("rear_hip_residual.") for key in state_dict
    )
    residual_independent = (
        residual_enabled
        and state_dict["rear_hip_residual.2.weight"].shape[0] == 2
    )

    train_cfg.policy.enable_rear_hip_residual = residual_enabled
    train_cfg.policy.rear_hip_residual_independent = residual_independent
    if residual_enabled:
        configured_max_action = getattr(
            train_cfg.policy, "rear_hip_residual_max_action", 0.08
        )
        train_cfg.policy.rear_hip_residual_max_action = float(
            os.environ.get(
                "GO1_REAR_HIP_RESIDUAL_MAX_ACTION",
                configured_max_action,
            )
        )
        train_cfg.policy.rear_hip_residual_speed_threshold_obs = 4.0
        train_cfg.policy.rear_hip_residual_full_speed_obs = 5.0
    train_cfg.runner.load_optimizer = False
    # Training curriculum state has one entry per training environment and
    # cannot be restored into play's single environment.
    train_cfg.runner.restore_terrain_levels = False
    print(
        "Play checkpoint policy configuration: "
        f"rear_hip_residual={'enabled' if residual_enabled else 'disabled'}, "
        f"residual_max_action="
        f"{train_cfg.policy.rear_hip_residual_max_action:.3f}, "
        f"path={checkpoint_path}"
    )
    return residual_enabled


def ground_frame_velocity(root_state):
    """Project world XY velocity onto the robot's yaw heading."""
    qx, qy, qz, qw = root_state[3:7]
    heading_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    heading_y = 2.0 * (qx * qy + qw * qz)
    heading_norm = torch.sqrt(
        heading_x * heading_x + heading_y * heading_y
    ).clamp_min(1e-6)
    heading_x = heading_x / heading_norm
    heading_y = heading_y / heading_norm
    world_vx, world_vy = root_state[7], root_state[8]
    forward = world_vx * heading_x + world_vy * heading_y
    lateral = -world_vx * heading_y + world_vy * heading_x
    horizontal = torch.sqrt(world_vx * world_vx + world_vy * world_vy)
    return forward, lateral, horizontal


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    residual_enabled = configure_policy_from_checkpoint(train_cfg, args)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 8
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5

    # ============================================================
    #  演示地形配置 (直接修改这里, 独立于训练配置)
    # ============================================================
    # mesh_type: "plane" = 纯平地, "trimesh" = 生成地形
    env_cfg.terrain.mesh_type = 'trimesh'
    # terrain_proportions: [flat, rough, stairs_up, stairs_down, obstacle]
    #   平地:       [1.0, 0.0, 0.0, 0.0, 0.0]  (或 mesh_type='plane')
    #   崎岖路面:   [0.0, 1.0, 0.0, 0.0, 0.0]
    #   上台阶:     [0.0, 0.0, 1.0, 0.0, 0.0]
    #   混合:       [0.33, 0.33, 0.34, 0.0, 0.0]
    env_cfg.terrain.terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0]
    # Force a task-independent pure rough-terrain branch. The generated
    # surface ranges from -0.025 m to +0.025 m: 0.05 m peak-to-valley.
    env_cfg.terrain.specialized_stair_training = False
    env_cfg.terrain.pure_rough_terrain = True
    env_cfg.terrain.rough_height = 0.05
    env_cfg.terrain.step_height_min = 0.03     # 最小台阶高度 [m]
    env_cfg.terrain.step_height_max = 0.15    # 最大台阶高度 [m]
    # ============================================================

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_limb_mass = False

    # 键盘动态控制命令, 初始停止
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.heading_command = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY and not residual_enabled:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    elif EXPORT_POLICY:
        print(
            "Skipping actor-only JIT export: it would omit the trained "
            "rear-hip residual."
        )

    # --- 键盘事件注册 ---
    viewer = env.viewer
    gym = env.gym

    key_map = {
        gymapi.KEY_W: "forward",
        gymapi.KEY_S: "backward",
        gymapi.KEY_A: "left",
        gymapi.KEY_D: "right",
        gymapi.KEY_Q: "turn_left",
        gymapi.KEY_E: "turn_right",
        gymapi.KEY_R: "stop",
        gymapi.KEY_F: "reset",
        gymapi.KEY_V: "toggle_camera",
        gymapi.KEY_C: "switch_view",
        gymapi.KEY_UP: "gear_up",
        gymapi.KEY_DOWN: "gear_down",
    }
    for key in key_map:
        gym.subscribe_viewer_keyboard_event(viewer, key, key_map[key])

    # --- 速度档位控制 ---
    speed_gear = 0.0       # 当前速度档位
    gear_step = 0.5        # 每档间隔
    max_gear = 6.0
    vel_x, vel_y, vel_yaw = 0.0, 0.0, 0.0
    pressed_keys = set()
    camera_auto_follow = True  # 相机自动跟随模式
    camera_view_mode = 0  # 0=跟随, 1=侧面, 2=正面, 3=俯视
    view_names = ["跟随", "侧面", "正面", "俯视"]

    print("\n" + "=" * 60)
    print("  Go1 交互式控制 (档位模式)")
    print("=" * 60)
    print("  W/S    长按 前进/后退 (按当前档位速度)")
    print("  A/D    长按 左移/右移")
    print("  Q/E    长按 左转/右转")
    print(f"  ↑/↓    加档/减档 (当前: {speed_gear:.1f} m/s)")
    print("  R      立即停止")
    print("  F      重置机器人")
    print("  V      切换相机模式 (自动跟随/手动)")
    print("  C      切换视角 (跟随/侧面/正面/俯视)")
    print("  ESC    退出")
    print("=" * 60 + "\n")

    robot_index = 0
    step_count = 0
    sim_dt = env.dt
    wall_start_time = time.time()
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0
    render_every = 2     # 每2个物理步渲染1帧
    camera_every = 2     # 每2步更新一次相机

    # --- 速度验证: 基于仿真时间的滚动 1 秒窗口 ---
    speed_window_steps = max(1, int(round(1.0 / sim_dt)))
    forward_speed_window = deque(maxlen=speed_window_steps)
    ground_speed_window = deque(maxlen=speed_window_steps)
    avg_forward_speed_1s = 0.0
    avg_ground_speed_1s = 0.0

    # --- tkinter 浮动数据窗口 ---
    info_win = tk.Tk()
    info_win.title("Go1 实时数据")
    info_win.geometry("350x440+0+0")
    info_win.attributes("-topmost", True)
    info_win.resizable(False, False)
    info_text = tk.Text(info_win, font=("Consolas", 11), width=40, height=26, bg="#1e1e1e", fg="#00ff88", bd=0)
    info_text.pack()
    def update_info(text):
        info_text.config(state=tk.NORMAL)
        info_text.delete("1.0", tk.END)
        info_text.insert(tk.END, text)
        info_text.config(state=tk.DISABLED)

    # 关闭env内部的渲染同步, 改为手动控制渲染频率
    env.enable_viewer_sync = False

    # --- 设置初始相机位置 ---
    robot_pos = env.root_states[robot_index, :3].cpu().numpy()
    robot_quat = env.root_states[robot_index, 3:7].cpu().numpy()
    qx, qy, qz, qw = robot_quat
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    dist, height = 3.5, 1.5
    camera_position = np.array([
        robot_pos[0] - dist * np.cos(yaw),
        robot_pos[1] - dist * np.sin(yaw),
        robot_pos[2] + height
    ])
    lookat = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.3])
    env.set_camera(camera_position, lookat)

    for i in range(10*int(env.max_episode_length)):
        # --- 处理键盘事件 ---
        for evt in gym.query_viewer_action_events(viewer):
            action = evt.action
            if evt.value > 0:
                if action == "stop":
                    vel_x, vel_y, vel_yaw = 0.0, 0.0, 0.0
                    pressed_keys.clear()
                    print(f"  [停止] 档位: {speed_gear:.1f} m/s")
                elif action == "reset":
                    env.reset_idx(torch.arange(env.num_envs, device=env.device))
                    vel_x, vel_y, vel_yaw = 0.0, 0.0, 0.0
                    pressed_keys.clear()
                    forward_speed_window.clear()
                    ground_speed_window.clear()
                    avg_forward_speed_1s = 0.0
                    avg_ground_speed_1s = 0.0
                    print("  [重置] 机器人已复位")
                elif action == "toggle_camera":
                    camera_auto_follow = not camera_auto_follow
                    mode = "自动跟随" if camera_auto_follow else "手动模式"
                    print(f"  [相机] {mode}")
                elif action == "switch_view":
                    camera_view_mode = (camera_view_mode + 1) % 4
                    print(f"  [视角] {view_names[camera_view_mode]}")
                elif action == "gear_up":
                    speed_gear = min(speed_gear + gear_step, max_gear)
                    print(f"  [加档] {speed_gear:.1f} m/s")
                elif action == "gear_down":
                    speed_gear = max(speed_gear - gear_step, 0.0)
                    print(f"  [减档] {speed_gear:.1f} m/s")
                else:
                    pressed_keys.add(action)
            else:
                pressed_keys.discard(action)

        # --- 按档位速度直接设置 ---
        if "forward" in pressed_keys:
            vel_x = speed_gear
        elif "backward" in pressed_keys:
            vel_x = -speed_gear
        else:
            vel_x = 0.0

        if "left" in pressed_keys:
            vel_y = speed_gear * 0.6
        elif "right" in pressed_keys:
            vel_y = -speed_gear * 0.6
        else:
            vel_y = 0.0

        if "turn_left" in pressed_keys:
            vel_yaw = speed_gear * 0.7
        elif "turn_right" in pressed_keys:
            vel_yaw = -speed_gear * 0.7
        else:
            vel_yaw = 0.0

        # --- 设置速度命令 ---
        env.commands[:, 0] = vel_x
        env.commands[:, 1] = vel_y
        env.commands[:, 2] = vel_yaw

        # Commands are changed between environment steps.  Update their
        # observation slice so this action sees the displayed command now,
        # instead of one 20 ms policy step later.
        obs[:, 9:12] = env.commands[:, :3] * env.commands_scale

        # --- 策略推理 + 步进 ---
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # --- 摄像头跟随 (降低频率) ---
        if camera_auto_follow and step_count % camera_every == 0:
            robot_pos = env.root_states[robot_index, :3].cpu().numpy()
            robot_quat = env.root_states[robot_index, 3:7].cpu().numpy()
            qx, qy, qz, qw = robot_quat
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

            # 根据视角模式计算相机位置
            if camera_view_mode == 0:  # 跟随 (右后方45°)
                dist, height = 3.5, 1.5
                camera_position = np.array([
                    robot_pos[0] - dist * np.cos(yaw),
                    robot_pos[1] - dist * np.sin(yaw),
                    robot_pos[2] + height
                ])
                lookat = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.3])
            elif camera_view_mode == 1:  # 侧面 (右侧90°)
                dist, height = 3.0, 0.8
                camera_position = np.array([
                    robot_pos[0] + dist * np.sin(yaw),
                    robot_pos[1] - dist * np.cos(yaw),
                    robot_pos[2] + height
                ])
                lookat = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.3])
            elif camera_view_mode == 2:  # 正面 (前方)
                dist, height = 3.5, 0.6
                camera_position = np.array([
                    robot_pos[0] + dist * np.cos(yaw),
                    robot_pos[1] + dist * np.sin(yaw),
                    robot_pos[2] + height
                ])
                lookat = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.3])
            else:  # 俯视 (上方45°)
                dist, height = 2.0, 4.0
                camera_position = np.array([
                    robot_pos[0],
                    robot_pos[1],
                    robot_pos[2] + height
                ])
                lookat = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.2])

            env.set_camera(camera_position, lookat)

        # --- 手动渲染 (控制频率, 避免每步都渲染) ---
        if step_count % render_every == 0:
            gym.step_graphics(env.sim)
            gym.draw_viewer(viewer, env.sim, False)
            gym.sync_frame_time(env.sim)
            fps_counter += 1
        else:
            gym.poll_viewer_events(viewer)

        # --- 速度验证: 滚动 1 秒, reset 时不混入跳变状态 ---
        if bool(dones[robot_index].item()):
            forward_speed_window.clear()
            ground_speed_window.clear()
            avg_forward_speed_1s = 0.0
            avg_ground_speed_1s = 0.0
        else:
            step_forward, _, step_horizontal = ground_frame_velocity(
                env.root_states[robot_index]
            )
            forward_speed_window.append(float(step_forward.item()))
            ground_speed_window.append(float(step_horizontal.item()))
            avg_forward_speed_1s = float(np.mean(forward_speed_window))
            avg_ground_speed_1s = float(np.mean(ground_speed_window))

        # --- 实时参数显示 (每5步刷新, 50 Hz策略下约10 Hz) ---
        if step_count % 5 == 0:
            elapsed = (step_count + 1) * sim_dt
            wall_elapsed = max(time.time() - wall_start_time, 1e-6)
            real_time_factor = elapsed / wall_elapsed
            root_pos = env.root_states[robot_index, :3].cpu().numpy()
            root_quat = env.root_states[robot_index, 3:7].cpu().numpy()
            world_lin_vel = env.root_states[robot_index, 7:10].cpu().numpy()
            dog_forward_speed, dog_lateral_speed, _ = (
                ground_frame_velocity(env.root_states[robot_index])
            )
            dog_forward_speed = float(dog_forward_speed.item())
            dog_lateral_speed = float(dog_lateral_speed.item())
            dof_pos = env.dof_pos[robot_index].cpu().numpy()
            contact = env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            torques = env.torques[robot_index].cpu().numpy()

            # 姿态角 (roll, pitch)
            qx, qy, qz, qw = root_quat
            roll = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
            pitch = np.degrees(np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qy*qy + qz*qz)))

            # 瞬时水平速度 (root_states世界坐标系)
            horiz_speed = np.sqrt(world_lin_vel[0]**2 + world_lin_vel[1]**2)
            command_speed = np.hypot(vel_x, vel_y)

            # 足端接触
            foot_contacts = (contact > 1.0).astype(int)
            foot_names = ["FL", "FR", "RL", "RR"]
            gait = "  ".join(f"{n}:{'●' if foot_contacts[i] else '○'}" for i, n in enumerate(foot_names))

            # 更新 tkinter 窗口
            lines = []
            camera_mode_str = "自动跟随" if camera_auto_follow else "手动模式"
            view_str = view_names[camera_view_mode]
            lines.append(
                f"  仿真: {elapsed:.1f}s  实时倍率: {real_time_factor:.2f}x"
            )
            lines.append(f"  画面 FPS: {current_fps:.0f}")
            lines.append(f"  相机: {camera_mode_str} | 视角: {view_str}")
            lines.append("─" * 34)
            lines.append(f"  【速度验证】")
            lines.append(
                f"  命令: vx={vel_x:+.2f} vy={vel_y:+.2f} m/s"
            )
            lines.append(f"  命令 yaw={vel_yaw:+.2f} rad/s")
            lines.append(
                f"  命令水平合速度: {command_speed:.2f} m/s"
            )
            lines.append(f"  狗前进速度: {dog_forward_speed:+.2f} m/s")
            lines.append(f"  狗横向速度: {dog_lateral_speed:+.2f} m/s")
            lines.append(f"  世界水平合速度: {horiz_speed:.2f} m/s")
            lines.append(
                f"  狗前进滚动1s: {avg_forward_speed_1s:+.2f} m/s"
            )
            lines.append(
                f"  滚动1s水平平均: {avg_ground_speed_1s:.2f} m/s"
            )
            lines.append("─" * 34)
            lines.append(f"  姿态: roll={roll:+.1f}° pitch={pitch:+.1f}°")
            lines.append(f"  位置: x={root_pos[0]:+.2f} y={root_pos[1]:+.2f}")
            lines.append(f"  档位: {speed_gear:.1f} m/s  奖励: {rews[robot_index].item():.3f}")
            lines.append("─" * 34)
            lines.append(f"  足端: {gait}")
            lines.append(f"  FL=[{np.degrees(dof_pos[0]):+.0f},{np.degrees(dof_pos[1]):+.0f},{np.degrees(dof_pos[2]):+.0f}]°"
                         f" FR=[{np.degrees(dof_pos[3]):+.0f},{np.degrees(dof_pos[4]):+.0f},{np.degrees(dof_pos[5]):+.0f}]°")
            lines.append(f"  RL=[{np.degrees(dof_pos[6]):+.0f},{np.degrees(dof_pos[7]):+.0f},{np.degrees(dof_pos[8]):+.0f}]°"
                         f" RR=[{np.degrees(dof_pos[9]):+.0f},{np.degrees(dof_pos[10]):+.0f},{np.degrees(dof_pos[11]):+.0f}]°")
            update_info("\n".join(lines))
            info_win.update_idletasks()

        step_count += 1

        # --- FPS 计算 ---
        fps_elapsed = time.time() - fps_start
        if fps_elapsed >= 1.0:
            current_fps = fps_counter / fps_elapsed
            fps_counter = 0
            fps_start = time.time()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    args = get_args()
    play(args)
