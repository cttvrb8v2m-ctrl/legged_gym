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
#    list of conditions and the following disclaimer in the documentation
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
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        action_abs = torch.abs(actions)
        self.raw_action_max_abs = max(
            self.raw_action_max_abs, torch.max(action_abs).item()
        )
        self.action_clip_count += torch.count_nonzero(
            action_abs > clip_actions
        ).item()
        self.action_total_count += actions.numel()
        self.action_clip_ratio = (
            self.action_clip_count / self.action_total_count
        )
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            # self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            # Note: Position control
            self.target_poses = self._compute_poses(self.actions).view(self.target_poses.shape)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_poses))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # reset observation history for RMA
        if self.history_length > 0:
            self.obs_history[env_ids] = 0.
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
            if key in self.stair_reward_log_names:
                self.extras["episode"][
                    "raw_rew_" + key
                ] = torch.mean(
                    self.raw_episode_sums[key][env_ids]
                ) / self.max_episode_length_s
            self.raw_episode_sums[key][env_ids] = 0.
        if self.stair_training_enabled:
            self.extras["episode"].update(
                self._stair_episode_metrics(env_ids)
            )
            self._reset_stair_episode_trackers(env_ids)
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            if getattr(
                self.cfg.terrain,
                "stair_outcome_curriculum",
                False,
            ):
                for level in range(self.max_terrain_level):
                    self.extras["episode"][
                        f"terrain_level_{level}_ratio"
                    ] = (
                        self.terrain_levels == level
                    ).float().mean()
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            raw_rew = self.reward_functions[i]()
            rew = raw_rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            # Integrating the unscaled term with dt makes raw_rew_* an
            # interpretable per-second mean alongside the weighted rew_* log.
            self.raw_episode_sums[name] += raw_rew * self.dt
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        # update observation history for RMA
        if self.history_length > 0:
            self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
            self.obs_history[:, -1, :] = self.obs_buf

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")

        for i, p in enumerate(props):
            if i == 0: # randomize base mass
                if self.cfg.domain_rand.randomize_base_mass:
                    rng = self.cfg.domain_rand.added_mass_range
                    p.mass += np.random.uniform(rng[0], rng[1])
            else: # randomize limb mass
                if self.cfg.domain_rand.randomize_limb_mass:
                    rng = self.cfg.domain_rand.added_limb_percentage
                    p.mass *= (1 + np.random.uniform(rng[0], rng[1]) )
            # print(f"Mass of body {i}: {p.mass} (after randomization)")

        # # randomize base mass
        # if self.cfg.domain_rand.randomize_base_mass:
        #     rng = self.cfg.domain_rand.added_mass_range
        #     props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_poses(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale  # wo - current pos is better than w - current pos
        target_poses = actions_scaled + self.default_dof_pos
        target_clamped = (
            (target_poses < self.dof_pos_limits[:, 0])
            | (target_poses > self.dof_pos_limits[:, 1])
        )
        self.joint_target_clamp_count += torch.count_nonzero(
            target_clamped
        ).item()
        self.joint_target_total_count += target_poses.numel()
        self.joint_target_clamp_ratio = (
            self.joint_target_clamp_count / self.joint_target_total_count
        )
        return torch.clip(target_poses, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        if getattr(
            self.cfg.terrain, "stair_outcome_curriculum", False
        ):
            passed_steps = self.stair_max_passed_steps[env_ids]
            fell = ~self.time_out_buf[env_ids]
            success = (
                passed_steps >= self.cfg.terrain.stair_step_count
            ) & ~fell
            failure = fell | (
                passed_steps
                < self.cfg.terrain.stair_curriculum_failure_steps
            )
            current_levels = self.terrain_levels[env_ids]
            highest_level = self.max_terrain_level - 1
            highest_success = success & (
                current_levels == highest_level
            )
            next_levels = current_levels.clone()
            next_levels = torch.where(
                success & ~highest_success,
                torch.clamp(current_levels + 1, max=highest_level),
                next_levels,
            )
            next_levels = torch.where(
                failure,
                torch.clamp(current_levels - 1, min=0),
                next_levels,
            )
            random_review_levels = torch.randint(
                0,
                self.max_terrain_level,
                current_levels.shape,
                device=self.device,
                dtype=current_levels.dtype,
            )
            next_levels = torch.where(
                highest_success, random_review_levels, next_levels
            )
            self.terrain_levels[env_ids] = next_levels
            self.env_origins[env_ids] = self.terrain_origins[
                self.terrain_levels[env_ids],
                self.terrain_types[env_ids],
            ]
            return
        if getattr(
            self.cfg.terrain, "freeze_terrain_level_distribution", False
        ):
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.3, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.3, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(
            rigid_body_state
        ).view(self.num_envs, self.num_bodies, 13)

        # initialize some data used later on
        self.common_step_counter = 0
        self.raw_action_max_abs = 0.0
        self.action_clip_count = 0
        self.action_total_count = 0
        self.action_clip_ratio = 0.0
        self.joint_target_clamp_count = 0
        self.joint_target_total_count = 0
        self.joint_target_clamp_ratio = 0.0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # Note: torque get directly from gym
        # self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_poses = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_actions)

        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.stair_training_enabled = getattr(
            self.cfg.terrain, "specialized_stair_training", False
        )
        self.front_foot_slots = torch.tensor(
            [
                index for index, name in enumerate(self.feet_names)
                if name.startswith("F")
            ],
            dtype=torch.long, device=self.device
        )
        self.rear_foot_slots = torch.tensor(
            [
                index for index, name in enumerate(self.feet_names)
                if name.startswith("R")
            ],
            dtype=torch.long, device=self.device
        )
        self.stair_episode_start_x = self.root_states[:, 0].clone()
        self.stair_last_root_height = self.root_states[:, 2].clone()
        self.stair_max_passed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.stair_front_max_steps = torch.zeros_like(
            self.stair_max_passed_steps
        )
        self.stair_rear_max_steps = torch.zeros_like(
            self.stair_max_passed_steps
        )
        self.stair_foot_max_levels = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.long, device=self.device
        )
        self.stair_foot_support_levels = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.long, device=self.device
        )
        rear_shape = (self.num_envs, len(self.rear_foot_slots))
        self.stair_rear_stable_candidate_levels = torch.zeros(
            rear_shape, dtype=torch.long, device=self.device
        )
        self.stair_rear_stable_contact_steps = torch.zeros(
            rear_shape, dtype=torch.long, device=self.device
        )
        self.stair_rear_rewarded_levels = torch.zeros(
            rear_shape, dtype=torch.long, device=self.device
        )
        self.stair_rear_follow_target_levels = torch.zeros(
            rear_shape, dtype=torch.long, device=self.device
        )
        self.stair_rear_follow_active = torch.zeros(
            rear_shape, dtype=torch.bool, device=self.device
        )
        self.stair_rear_follow_elapsed_steps = torch.zeros(
            rear_shape, dtype=torch.long, device=self.device
        )
        self.stair_rear_follow_clearance_rewarded = torch.zeros(
            rear_shape, dtype=torch.bool, device=self.device
        )
        self.stair_last_rear_foot_pos = self.rigid_body_state[
            :, self.feet_indices[self.rear_foot_slots], :3
        ].clone()
        self.stair_rear_progress_initialized = torch.ones(
            rear_shape, dtype=torch.bool, device=self.device
        )
        self.stair_rear_target_best_distance = torch.full(
            rear_shape, float("inf"), dtype=torch.float,
            device=self.device
        )
        self.stair_rear_target_riser_x = torch.zeros(
            rear_shape, dtype=torch.float, device=self.device
        )
        self.stair_rear_target_riser_valid = torch.zeros(
            rear_shape, dtype=torch.bool, device=self.device
        )
        self.stair_summit_rewarded = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.stair_rear_slip_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.stair_rear_slip_by_foot = torch.zeros(
            self.num_envs, len(self.rear_foot_slots),
            dtype=torch.bool, device=self.device
        )
        self.stair_rear_contact_speed_sum = torch.zeros(
            rear_shape, dtype=torch.float, device=self.device
        )
        self.stair_rear_contact_sample_count = torch.zeros(
            rear_shape, dtype=torch.long, device=self.device
        )
        self.stair_rear_slide_distance = torch.zeros(
            rear_shape, dtype=torch.float, device=self.device
        )
        self.stair_rear_slip_steps = torch.zeros(
            rear_shape, dtype=torch.long, device=self.device
        )
        self.stair_rear_contact_speed_samples = torch.full(
            (
                self.num_envs,
                int(self.max_episode_length),
                len(self.rear_foot_slots),
            ),
            float("nan"), dtype=torch.float, device=self.device
        )
        self.stair_foot_top_landings = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.float, device=self.device
        )
        self.stair_front_collision_count = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stair_calf_collision_count = torch.zeros_like(
            self.stair_front_collision_count
        )
        limb_count = len(self.feet_indices) + 3 * len(self.calf_indices)
        body_count = len(self.base_indices) + len(self.hip_indices)
        self.stair_soft_limb_last_penetrating = torch.zeros(
            self.num_envs, limb_count, dtype=torch.bool, device=self.device
        )
        self.stair_soft_body_last_penetrating = torch.zeros(
            self.num_envs, body_count, dtype=torch.bool, device=self.device
        )
        self.stair_soft_limb_penetration_count = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stair_soft_body_penetration_count = torch.zeros_like(
            self.stair_soft_limb_penetration_count
        )
        self.stair_soft_limb_penetration_depth_sum = torch.zeros_like(
            self.stair_soft_limb_penetration_count
        )
        self.stair_soft_body_penetration_depth_sum = torch.zeros_like(
            self.stair_soft_limb_penetration_count
        )
        self.stair_soft_limb_penetration_samples = torch.zeros_like(
            self.stair_soft_limb_penetration_count
        )
        self.stair_soft_body_penetration_samples = torch.zeros_like(
            self.stair_soft_limb_penetration_count
        )
        self.stair_soft_limb_penetration_max_depth = torch.zeros_like(
            self.stair_soft_limb_penetration_count
        )
        self.stair_soft_body_penetration_max_depth = torch.zeros_like(
            self.stair_soft_limb_penetration_count
        )
        self.stair_last_foot_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.bool, device=self.device
        )
        self.stair_last_calf_contacts = torch.zeros(
            self.num_envs, len(self.calf_indices),
            dtype=torch.bool, device=self.device
        )
        self.stair_max_pitch = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stair_max_roll = torch.zeros_like(self.stair_max_pitch)
        self.stair_episode_step_height = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._stair_state_cache_step = -1
        self._stair_state_cache = None
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        
        self.history_length = getattr(self.cfg, 'rma', None)
        if self.history_length is not None:
            self.history_length = self.history_length.history_length
            self.obs_history = torch.zeros(self.num_envs, self.history_length, self.num_obs, dtype=torch.float, device=self.device, requires_grad=False)
            print(f"RMA: obs_history initialized with shape: {self.obs_history.shape}")
        else:
            self.history_length = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
            print("name", name)

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.raw_episode_sums = {
            name: torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device,
                requires_grad=False
            )
            for name in self.reward_scales.keys()
        }
        self.stair_reward_log_names = {
            "rear_step_progress",
            "rear_foot_landing",
            "rear_foot_slip",
            "front_stair_collision",
            "calf_stair_collision",
            "stair_step_progress",
            "rear_stable_step",
            "rear_target_progress",
            "stair_summit",
            "soft_limb_penetration_count",
            "soft_limb_penetration_depth",
            "soft_body_penetration_count",
            "soft_body_penetration_depth",
        }

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        calf_names = [s for s in body_names if "calf" in s]
        hip_names = [s for s in body_names if s.endswith("_hip")]
        base_names = [s for s in body_names if s == "base"]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # ! set gym's PD controller
        for i in range(self.num_dof):
            name = self.dof_names[i]
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    dof_props_asset['driveMode'][i] = gymapi.DOF_MODE_POS
                    dof_props_asset['stiffness'][i] = self.cfg.control.stiffness[dof_name] #self.Kp
                    dof_props_asset['damping'][i] = self.cfg.control.damping[dof_name] #self.Kd

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)  # Note: important to read torque !!!!
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.feet_names = feet_names
        self.calf_indices = torch.zeros(
            len(calf_names), dtype=torch.long, device=self.device,
            requires_grad=False
        )
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )
        self.hip_indices = torch.zeros(
            len(hip_names), dtype=torch.long, device=self.device,
            requires_grad=False
        )
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )
        self.base_indices = torch.zeros(
            len(base_names), dtype=torch.long, device=self.device,
            requires_grad=False
        )
        for i, name in enumerate(base_names):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            if getattr(
                self.cfg.terrain,
                "freeze_terrain_level_distribution",
                False,
            ):
                self.terrain_levels = (
                    torch.arange(self.num_envs, device=self.device)
                    % (max_init_level + 1)
                )
                permutation = torch.randperm(
                    self.num_envs, device=self.device
                )
                self.terrain_levels = self.terrain_levels[permutation]
                counts = torch.bincount(
                    self.terrain_levels, minlength=max_init_level + 1
                )
                print(
                    "Fixed terrain-level counts: "
                    f"{counts.detach().cpu().tolist()}"
                )
            else:
                self.terrain_levels = torch.randint(
                    0, max_init_level + 1,
                    (self.num_envs,), device=self.device
                )
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ stair-specialization helpers and rewards ----------------
    def _stair_env_mask(self):
        if not self.stair_training_enabled:
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        probabilities = self.cfg.terrain.terrain_proportions
        stair_start = float(sum(probabilities[:2]))
        stair_end = float(sum(probabilities[:3]))
        choices = (
            self.terrain_types.float() / self.cfg.terrain.num_cols + 0.001
        )
        return (choices >= stair_start) & (choices < stair_end)

    def _stair_step_heights(self):
        levels = self.terrain_levels.float()
        explicit_heights = getattr(
            self.cfg.terrain, "stair_curriculum_height_bins", None
        )
        if explicit_heights is not None:
            height_table = torch.tensor(
                explicit_heights, device=self.device, dtype=levels.dtype
            )
            indices = torch.clamp(
                self.terrain_levels, 0, len(explicit_heights) - 1
            )
            return height_table[indices]

        ranges = self.cfg.terrain.stair_stage_height_ranges
        heights = torch.empty_like(levels)
        stage_1 = levels <= 3
        stage_2 = (levels > 3) & (levels <= 6)
        stage_3 = levels > 6
        heights[stage_1] = (
            ranges[0][0]
            + (ranges[0][1] - ranges[0][0])
            * levels[stage_1] / 3.0
        )
        heights[stage_2] = (
            ranges[1][0]
            + (ranges[1][1] - ranges[1][0])
            * (levels[stage_2] - 3.0) / 3.0
        )
        heights[stage_3] = (
            ranges[2][0]
            + (ranges[2][1] - ranges[2][0])
            * (levels[stage_3] - 6.0) / 3.0
        )
        return heights

    def _sample_terrain_height_at(self, positions):
        points = positions.clone()
        points[..., :2] += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(
            points[..., 0], 0, self.height_samples.shape[0] - 2
        )
        py = torch.clip(
            points[..., 1], 0, self.height_samples.shape[1] - 2
        )
        height = torch.minimum(
            self.height_samples[px, py],
            self.height_samples[px + 1, py],
        )
        height = torch.minimum(height, self.height_samples[px, py + 1])
        return height * self.terrain.cfg.vertical_scale

    def _stair_riser_penetration(
        self, positions, horizontal_extent, vertical_extent
    ):
        """Horizontal penetration beyond an ideal ascending stair face."""
        scale = self.terrain.cfg.horizontal_scale
        points_x = (
            positions[..., 0] + self.terrain.cfg.border_size
        ) / scale
        points_y = (
            positions[..., 1] + self.terrain.cfg.border_size
        ) / scale
        px = torch.clamp(
            torch.floor(points_x).long(),
            1,
            self.height_samples.shape[0] - 2,
        )
        py = torch.clamp(
            torch.floor(points_y).long(),
            0,
            self.height_samples.shape[1] - 2,
        )
        vertical_scale = self.terrain.cfg.vertical_scale
        current_height = self.height_samples[px, py] * vertical_scale
        previous_height = (
            self.height_samples[px - 1, py] * vertical_scale
        )
        next_height = self.height_samples[px + 1, py] * vertical_scale
        expected_height = self._stair_step_heights().unsqueeze(1)
        previous_face = torch.abs(
            (current_height - previous_height) - expected_height
        ) <= max(vertical_scale * 1.5, 0.002)
        next_face = torch.abs(
            (next_height - current_height) - expected_height
        ) <= max(vertical_scale * 1.5, 0.002)
        cell_fraction = points_x - torch.floor(points_x)
        previous_depth = (
            horizontal_extent + cell_fraction * scale
        )
        next_depth = (
            horizontal_extent - (1.0 - cell_fraction) * scale
        )
        inside_previous_band = (
            (positions[..., 2] - vertical_extent < current_height)
            & (positions[..., 2] + vertical_extent > previous_height)
        )
        inside_next_band = (
            (positions[..., 2] - vertical_extent < next_height)
            & (positions[..., 2] + vertical_extent > current_height)
        )
        horizontal_depth = torch.maximum(
            torch.where(
                previous_face & inside_previous_band,
                previous_depth,
                torch.zeros_like(previous_depth),
            ),
            torch.where(
                next_face & inside_next_band,
                next_depth,
                torch.zeros_like(next_depth),
            ),
        )
        allowed_depth = float(getattr(
            self.cfg.terrain, "soft_stair_face_recess", 0.02
        ))
        penetrating = (
            (horizontal_depth > 1e-4)
            & self._stair_env_mask().unsqueeze(1)
        )
        depth = torch.where(
            penetrating,
            torch.clamp(horizontal_depth, max=allowed_depth),
            torch.zeros_like(horizontal_depth),
        )
        return penetrating, depth

    def _base_roll_pitch(self):
        x, y, z, w = self.base_quat.unbind(dim=-1)
        roll = torch.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x.square() + y.square()),
        )
        pitch = torch.asin(
            torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
        )
        return roll, pitch

    def _get_stair_training_state(self):
        if self._stair_state_cache_step == self.common_step_counter:
            return self._stair_state_cache

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        foot_state = self.rigid_body_state[:, self.feet_indices]
        calf_state = self.rigid_body_state[:, self.calf_indices]
        foot_pos = foot_state[..., :3]
        foot_vel = foot_state[..., 7:10]
        foot_horizontal_speed = torch.norm(foot_vel[..., :2], dim=-1)
        foot_vertical_speed = torch.abs(foot_vel[..., 2])
        foot_contact = (
            self.contact_forces[:, self.feet_indices, 2] > 1.0
        )
        calf_contact = (
            torch.norm(
                self.contact_forces[:, self.calf_indices], dim=-1
            ) > self.cfg.rewards.stair_collision_force_threshold
        )
        terrain_height = self._sample_terrain_height_at(foot_pos)
        foot_clearance = foot_pos[..., 2] - terrain_height
        stair_mask = self._stair_env_mask()

        front_force = self.contact_forces[
            :, self.feet_indices[self.front_foot_slots]
        ]
        front_horizontal_force = torch.norm(front_force[..., :2], dim=-1)
        front_vertical_force = torch.abs(front_force[..., 2])
        front_collision = (
            front_horizontal_force
            > torch.maximum(
                torch.full_like(
                    front_horizontal_force,
                    self.cfg.rewards.stair_collision_force_threshold,
                ),
                2.0 * front_vertical_force,
            )
        )

        new_foot_contact = foot_contact & ~self.stair_last_foot_contacts
        elevated_surface = (
            terrain_height
            > self.env_origins[:, 2].unsqueeze(1)
            + self.cfg.rewards.stair_top_surface_min_height
        )
        top_landing = new_foot_contact & elevated_surface
        stable_top_landing = (
            top_landing
            & (
                torch.abs(foot_pos[..., 2] - terrain_height)
                <= self.cfg.rewards.stair_top_contact_height_tolerance
            )
            & (
                foot_vertical_speed
                <= self.cfg.rewards.rear_landing_vertical_speed_threshold
            )
            & (
                foot_horizontal_speed
                <= self.cfg.rewards.rear_landing_horizontal_speed_threshold
            )
        )
        stable_top_contact = (
            foot_contact
            & elevated_surface
            & (
                torch.abs(foot_pos[..., 2] - terrain_height)
                <= self.cfg.rewards.stair_top_contact_height_tolerance
            )
            & (
                foot_vertical_speed
                <= self.cfg.rewards.rear_landing_vertical_speed_threshold
            )
            & (
                foot_horizontal_speed
                <= self.cfg.rewards.rear_landing_horizontal_speed_threshold
            )
        )
        new_front_collision = (
            front_collision
            & ~self.stair_last_foot_contacts[:, self.front_foot_slots]
        )
        new_calf_collision = (
            calf_contact & ~self.stair_last_calf_contacts
        )
        calf_offsets = torch.tensor(
            [[0.0, 0.0, -0.02],
             [0.0, 0.0, -0.1065],
             [0.0, 0.0, -0.193]],
            device=self.device,
            dtype=calf_state.dtype,
        )
        calf_quat = calf_state[..., 3:7].unsqueeze(2).expand(
            -1, -1, len(calf_offsets), -1
        )
        expanded_calf_offsets = calf_offsets.view(
            1, 1, len(calf_offsets), 3
        ).expand(self.num_envs, len(self.calf_indices), -1, -1)
        calf_sample_pos = (
            calf_state[..., :3].unsqueeze(2)
            + quat_apply(
                calf_quat.reshape(-1, 4),
                expanded_calf_offsets.reshape(-1, 3),
            ).view(
                self.num_envs, len(self.calf_indices),
                len(calf_offsets), 3
            )
        ).reshape(self.num_envs, -1, 3)
        limb_pos = torch.cat((foot_pos, calf_sample_pos), dim=1)
        limb_horizontal_extent = torch.cat((
            torch.full(
                (self.num_envs, len(self.feet_indices)),
                0.02, device=self.device
            ),
            torch.full(
                (self.num_envs, calf_sample_pos.shape[1]),
                0.008, device=self.device
            ),
        ), dim=1)
        limb_vertical_extent = limb_horizontal_extent
        body_indices = torch.cat((self.base_indices, self.hip_indices))
        body_pos = self.rigid_body_state[:, body_indices, :3]
        body_horizontal_extent = torch.cat((
            torch.full(
                (self.num_envs, len(self.base_indices)),
                0.1881, device=self.device
            ),
            torch.full(
                (self.num_envs, len(self.hip_indices)),
                0.046, device=self.device
            ),
        ), dim=1)
        body_vertical_extent = torch.cat((
            torch.full(
                (self.num_envs, len(self.base_indices)),
                0.057, device=self.device
            ),
            torch.full(
                (self.num_envs, len(self.hip_indices)),
                0.046, device=self.device
            ),
        ), dim=1)
        limb_penetrating, limb_penetration_depth = (
            self._stair_riser_penetration(
                limb_pos, limb_horizontal_extent, limb_vertical_extent
            )
        )
        body_penetrating, body_penetration_depth = (
            self._stair_riser_penetration(
                body_pos, body_horizontal_extent, body_vertical_extent
            )
        )
        new_limb_penetration = (
            limb_penetrating
            & ~self.stair_soft_limb_last_penetrating
        )
        new_body_penetration = (
            body_penetrating
            & ~self.stair_soft_body_last_penetrating
        )

        step_height = torch.clamp(
            self._stair_step_heights().unsqueeze(1), min=1e-6
        )
        terrain_level = torch.clamp(
            torch.round(
                (
                    terrain_height
                    - self.env_origins[:, 2].unsqueeze(1)
                ) / step_height
            ).long(),
            0,
            self.cfg.terrain.stair_step_count,
        )
        self.stair_foot_support_levels = torch.where(
            foot_contact,
            terrain_level,
            self.stair_foot_support_levels,
        )
        next_step_level = torch.clamp(
            self.stair_foot_support_levels + 1,
            max=self.cfg.terrain.stair_step_count,
        )
        support_top_world_z = (
            self.env_origins[:, 2].unsqueeze(1)
            + self.stair_foot_support_levels * step_height
        )
        next_step_top_world_z = (
            self.env_origins[:, 2].unsqueeze(1)
            + next_step_level * step_height
        )
        rear_levels = terrain_level[:, self.rear_foot_slots]
        rear_stable_contact = stable_top_contact[:, self.rear_foot_slots]
        same_candidate = (
            rear_levels == self.stair_rear_stable_candidate_levels
        )
        self.stair_rear_stable_contact_steps = torch.where(
            rear_stable_contact,
            torch.where(
                same_candidate,
                self.stair_rear_stable_contact_steps + 1,
                torch.ones_like(self.stair_rear_stable_contact_steps),
            ),
            torch.zeros_like(self.stair_rear_stable_contact_steps),
        )
        self.stair_rear_stable_candidate_levels = torch.where(
            rear_stable_contact,
            rear_levels,
            torch.zeros_like(self.stair_rear_stable_candidate_levels),
        )
        required_stable_steps = int(
            getattr(self.cfg.rewards, "rear_stable_contact_steps", 3)
        )
        new_rear_stable_step = (
            rear_stable_contact
            & (rear_levels > 0)
            & (
                self.stair_rear_stable_contact_steps
                >= required_stable_steps
            )
            & (rear_levels > self.stair_rear_rewarded_levels)
        )
        self.stair_rear_rewarded_levels = torch.where(
            new_rear_stable_step,
            rear_levels,
            self.stair_rear_rewarded_levels,
        )
        landed_level = torch.where(
            stable_top_landing,
            terrain_level,
            torch.zeros_like(terrain_level),
        )
        old_passed_steps = self.stair_max_passed_steps.clone()
        old_rear_steps = self.stair_rear_max_steps.clone()
        self.stair_foot_max_levels = torch.maximum(
            self.stair_foot_max_levels, landed_level
        )
        front_steps = torch.min(
            self.stair_foot_max_levels[:, self.front_foot_slots], dim=1
        ).values
        rear_steps = torch.min(
            self.stair_foot_max_levels[:, self.rear_foot_slots], dim=1
        ).values
        passed_steps = torch.minimum(front_steps, rear_steps)
        self.stair_front_max_steps = torch.maximum(
            self.stair_front_max_steps, front_steps
        )
        self.stair_rear_max_steps = torch.maximum(
            self.stair_rear_max_steps, rear_steps
        )
        self.stair_max_passed_steps = torch.maximum(
            self.stair_max_passed_steps, passed_steps
        )
        new_step_progress = (
            self.stair_max_passed_steps - old_passed_steps
        ).float()
        new_rear_step_progress = (
            self.stair_rear_max_steps - old_rear_steps
        ).float()
        new_summit = (
            (self.stair_max_passed_steps >= self.cfg.terrain.stair_step_count)
            & ~self.stair_summit_rewarded
        )
        self.stair_summit_rewarded |= new_summit

        rear_contact = foot_contact[:, self.rear_foot_slots]
        rear_horizontal_speed = torch.norm(
            foot_vel[:, self.rear_foot_slots, :2], dim=-1
        )
        rear_foot_pos = foot_pos[:, self.rear_foot_slots]
        rear_max_levels = self.stair_foot_max_levels[
            :, self.rear_foot_slots
        ]
        follow_front_target = front_steps.unsqueeze(1).expand_as(
            rear_max_levels
        )
        new_follow_target = (
            (follow_front_target > rear_max_levels)
            & (
                follow_front_target
                > self.stair_rear_follow_target_levels
            )
        )
        self.stair_rear_follow_target_levels = torch.where(
            new_follow_target,
            follow_front_target,
            self.stair_rear_follow_target_levels,
        )
        self.stair_rear_follow_active = torch.where(
            new_follow_target,
            torch.ones_like(self.stair_rear_follow_active),
            self.stair_rear_follow_active,
        )
        self.stair_rear_follow_elapsed_steps = torch.where(
            new_follow_target,
            torch.zeros_like(self.stair_rear_follow_elapsed_steps),
            self.stair_rear_follow_elapsed_steps,
        )
        self.stair_rear_follow_clearance_rewarded = torch.where(
            new_follow_target,
            torch.zeros_like(
                self.stair_rear_follow_clearance_rewarded
            ),
            self.stair_rear_follow_clearance_rewarded,
        )

        rear_target_top_world_z = (
            self.env_origins[:, 2].unsqueeze(1)
            + self.stair_rear_follow_target_levels * step_height
        )
        rear_foot_world_z = foot_pos[
            :, self.rear_foot_slots, 2
        ]
        rear_at_target_riser = (
            rear_levels >= self.stair_rear_follow_target_levels
        )
        new_rear_follow_clearance = (
            self.stair_rear_follow_active
            & ~self.stair_rear_follow_clearance_rewarded
            & ~rear_contact
            & rear_at_target_riser
            & (
                rear_foot_world_z
                >= rear_target_top_world_z
                + float(
                    getattr(
                        self.cfg.rewards,
                        "rear_follow_clearance_margin",
                        0.015,
                    )
                )
            )
        )
        self.stair_rear_follow_clearance_rewarded |= (
            new_rear_follow_clearance
        )
        self.stair_rear_follow_elapsed_steps = torch.where(
            self.stair_rear_follow_active,
            self.stair_rear_follow_elapsed_steps + 1,
            self.stair_rear_follow_elapsed_steps,
        )
        rear_follow_completed = (
            self.stair_rear_follow_active
            & new_rear_stable_step
            & (
                rear_levels
                >= self.stair_rear_follow_target_levels
            )
        )
        rear_follow_timeout_steps = int(
            getattr(
                self.cfg.rewards,
                "rear_follow_timeout_steps",
                40,
            )
        )
        new_rear_follow_timeout = (
            self.stair_rear_follow_active
            & ~rear_follow_completed
            & (
                self.stair_rear_follow_elapsed_steps
                >= rear_follow_timeout_steps
            )
        )
        self.stair_rear_follow_active &= ~(
            rear_follow_completed | new_rear_follow_timeout
        )

        # Goal-directed shaping for the rear feet.  Unlike the generic dense
        # forward/up terms below, this only rewards a new best distance to a
        # concrete point on the target tread.  Consequently, overshooting,
        # hovering, or moving back and forth cannot repeatedly earn reward.
        tread_depth = max(
            float(self.cfg.terrain.stair_tread_depth), 1e-6
        )
        search_scale = float(self.cfg.terrain.horizontal_scale)
        search_extent = float(getattr(
            self.cfg.rewards,
            "rear_target_search_extent",
            1.5 * tread_depth,
        ))
        search_offsets = torch.arange(
            0.0,
            search_extent + 0.5 * search_scale,
            search_scale,
            device=self.device,
            dtype=rear_foot_pos.dtype,
        )
        rear_search_points = rear_foot_pos.unsqueeze(2).expand(
            -1, -1, search_offsets.numel(), -1
        ).clone()
        rear_search_points[..., 0] += search_offsets.view(1, 1, -1)
        rear_search_heights = self._sample_terrain_height_at(
            rear_search_points
        )
        target_height_tolerance = max(
            float(self.cfg.terrain.vertical_scale) * 1.5,
            0.002,
        )
        reaches_target_tread = (
            rear_search_heights
            >= rear_target_top_world_z.unsqueeze(-1)
            - target_height_tolerance
        )
        target_tread_found = torch.any(reaches_target_tread, dim=-1)
        target_offset_index = torch.argmax(
            reaches_target_tread.float(), dim=-1
        )
        first_target_offset = search_offsets[target_offset_index]
        candidate_riser_x = (
            rear_foot_pos[..., 0] + first_target_offset
        )
        self.stair_rear_target_riser_x = torch.where(
            new_follow_target & target_tread_found,
            candidate_riser_x,
            self.stair_rear_target_riser_x,
        )
        self.stair_rear_target_riser_valid = torch.where(
            new_follow_target,
            target_tread_found,
            self.stair_rear_target_riser_valid,
        )
        landing_margin = float(getattr(
            self.cfg.rewards, "rear_target_landing_margin", 0.06
        ))
        clearance_margin = float(getattr(
            self.cfg.rewards, "rear_target_clearance_margin", 0.025
        ))
        two_stage_target = bool(getattr(
            self.cfg.rewards, "rear_target_two_stage", False
        ))
        if two_stage_target:
            clearance_forward_margin = float(getattr(
                self.cfg.rewards,
                "rear_target_waypoint_forward_margin",
                0.02,
            ))
            landing_height_margin = float(getattr(
                self.cfg.rewards,
                "rear_target_landing_height_margin",
                0.02,
            ))
            clearance_phase = (
                ~self.stair_rear_follow_clearance_rewarded
            )
            target_x = torch.where(
                clearance_phase,
                (
                    self.stair_rear_target_riser_x
                    + clearance_forward_margin
                ),
                self.stair_rear_target_riser_x + landing_margin,
            )
            target_z = torch.where(
                clearance_phase,
                rear_target_top_world_z + clearance_margin,
                rear_target_top_world_z + landing_height_margin,
            )
            target_valid = self.stair_rear_target_riser_valid
        else:
            # Preserve the original single-stage experiment exactly.
            target_x = (
                rear_foot_pos[..., 0]
                + first_target_offset
                + landing_margin
            )
            target_z = rear_target_top_world_z + clearance_margin
            target_valid = target_tread_found
        normalized_x_error = (
            target_x - rear_foot_pos[..., 0]
        ) / tread_depth
        normalized_z_error = (
            target_z - rear_foot_pos[..., 2]
        ) / step_height
        vertical_weight = float(getattr(
            self.cfg.rewards, "rear_target_vertical_weight", 0.50
        ))
        target_distance = torch.sqrt(
            normalized_x_error.square()
            + vertical_weight * normalized_z_error.square()
            + 1e-8
        )
        target_phase_changed = (
            new_follow_target
            | (two_stage_target & new_rear_follow_clearance)
        )
        self.stair_rear_target_best_distance = torch.where(
            target_phase_changed,
            target_distance,
            self.stair_rear_target_best_distance,
        )
        rear_target_active = (
            self.stair_rear_follow_active
            & ~rear_contact
            & target_valid
        )
        rear_target_improvement = torch.where(
            rear_target_active,
            torch.clamp(
                self.stair_rear_target_best_distance - target_distance,
                min=0.0,
                max=float(getattr(
                    self.cfg.rewards,
                    "rear_target_progress_clip_per_step",
                    0.08,
                )),
            ),
            torch.zeros_like(target_distance),
        )
        self.stair_rear_target_best_distance = torch.where(
            rear_target_active,
            torch.minimum(
                self.stair_rear_target_best_distance,
                target_distance,
            ),
            self.stair_rear_target_best_distance,
        )
        self.stair_rear_target_best_distance = torch.where(
            self.stair_rear_follow_active,
            self.stair_rear_target_best_distance,
            torch.full_like(
                self.stair_rear_target_best_distance, float("inf")
            ),
        )

        rear_position_delta = (
            rear_foot_pos - self.stair_last_rear_foot_pos
        )
        rear_forward_delta = rear_position_delta[..., 0]
        rear_up_delta = rear_position_delta[..., 2]
        rear_following = follow_front_target > rear_max_levels
        minimum_forward_delta = float(
            getattr(
                self.cfg.rewards,
                "rear_follow_min_forward_delta",
                0.001,
            )
        )
        rear_progress_mask = (
            rear_following
            & ~rear_contact
            & self.stair_rear_progress_initialized
            & (rear_forward_delta > minimum_forward_delta)
        )
        tread_depth = max(
            float(self.cfg.terrain.stair_tread_depth), 1e-6
        )
        max_forward_ratio = float(
            getattr(
                self.cfg.rewards,
                "rear_follow_forward_clip_per_step",
                0.08,
            )
        )
        max_up_ratio = float(
            getattr(
                self.cfg.rewards,
                "rear_follow_up_clip_per_step",
                0.08,
            )
        )
        rear_follow_forward_progress = torch.where(
            rear_progress_mask,
            torch.clamp(
                rear_forward_delta / tread_depth,
                min=0.0,
                max=max_forward_ratio,
            ),
            torch.zeros_like(rear_forward_delta),
        )
        rear_follow_up_progress = torch.where(
            rear_progress_mask,
            torch.clamp(
                rear_up_delta / step_height,
                min=0.0,
                max=max_up_ratio,
            ),
            torch.zeros_like(rear_up_delta),
        )
        self.stair_last_rear_foot_pos[:] = rear_foot_pos
        self.stair_rear_progress_initialized[:] = True
        rear_slip = rear_contact & (
            rear_horizontal_speed
            > self.cfg.rewards.rear_slip_velocity_threshold
        )
        for foot_slot in range(len(self.rear_foot_slots)):
            valid = (
                rear_contact[:, foot_slot]
                & (
                    self.stair_rear_contact_sample_count[:, foot_slot]
                    < self.stair_rear_contact_speed_samples.shape[1]
                )
            )
            valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            if valid_ids.numel() > 0:
                sample_indices = self.stair_rear_contact_sample_count[
                    valid_ids, foot_slot
                ]
                self.stair_rear_contact_speed_samples[
                    valid_ids, sample_indices, foot_slot
                ] = rear_horizontal_speed[valid_ids, foot_slot]
        rear_contact_float = rear_contact.float()
        self.stair_rear_contact_speed_sum += (
            rear_horizontal_speed * rear_contact_float
        )
        self.stair_rear_contact_sample_count += rear_contact.long()
        self.stair_rear_slide_distance += (
            rear_horizontal_speed * rear_contact_float * self.dt
        )
        self.stair_rear_slip_steps += rear_slip.long()
        self.stair_rear_slip_episode |= torch.any(rear_slip, dim=1)
        self.stair_rear_slip_by_foot |= rear_slip
        self.stair_foot_top_landings += stable_top_landing.float()
        self.stair_front_collision_count += torch.sum(
            new_front_collision.float(), dim=1
        )
        self.stair_calf_collision_count += torch.sum(
            new_calf_collision.float(), dim=1
        )
        self.stair_soft_limb_penetration_count += torch.sum(
            new_limb_penetration.float(), dim=1
        )
        self.stair_soft_body_penetration_count += torch.sum(
            new_body_penetration.float(), dim=1
        )
        self.stair_soft_limb_penetration_depth_sum += torch.sum(
            limb_penetration_depth, dim=1
        )
        self.stair_soft_body_penetration_depth_sum += torch.sum(
            body_penetration_depth, dim=1
        )
        self.stair_soft_limb_penetration_samples += torch.sum(
            limb_penetrating.float(), dim=1
        )
        self.stair_soft_body_penetration_samples += torch.sum(
            body_penetrating.float(), dim=1
        )
        self.stair_soft_limb_penetration_max_depth = torch.maximum(
            self.stair_soft_limb_penetration_max_depth,
            torch.max(limb_penetration_depth, dim=1).values,
        )
        self.stair_soft_body_penetration_max_depth = torch.maximum(
            self.stair_soft_body_penetration_max_depth,
            torch.max(body_penetration_depth, dim=1).values,
        )

        roll, pitch = self._base_roll_pitch()
        self.stair_max_roll = torch.maximum(
            self.stair_max_roll, roll.abs()
        )
        self.stair_max_pitch = torch.maximum(
            self.stair_max_pitch, pitch.abs()
        )
        root_height_delta = (
            self.root_states[:, 2] - self.stair_last_root_height
        )
        self.stair_last_root_height[:] = self.root_states[:, 2]
        self.stair_last_foot_contacts[:] = foot_contact
        self.stair_last_calf_contacts[:] = calf_contact
        self.stair_soft_limb_last_penetrating[:] = limb_penetrating
        self.stair_soft_body_last_penetrating[:] = body_penetrating

        self._stair_state_cache = {
            "mask": stair_mask.float(),
            "root_height_delta": root_height_delta,
            "new_step_progress": new_step_progress,
            "new_rear_step_progress": new_rear_step_progress,
            "new_rear_stable_step": new_rear_stable_step,
            "new_rear_follow_clearance": new_rear_follow_clearance,
            "new_rear_follow_timeout": new_rear_follow_timeout,
            "rear_follow_forward_progress": (
                rear_follow_forward_progress
            ),
            "rear_follow_up_progress": rear_follow_up_progress,
            "rear_target_progress": rear_target_improvement,
            "new_summit": new_summit,
            "top_landing": stable_top_landing,
            "foot_contact": foot_contact,
            "foot_clearance": foot_clearance,
            "foot_world_z": foot_pos[..., 2],
            "support_top_world_z": support_top_world_z,
            "next_step_top_world_z": next_step_top_world_z,
            "rear_horizontal_speed": rear_horizontal_speed,
            "rear_contact": rear_contact,
            "new_front_collision": new_front_collision,
            "new_calf_collision": new_calf_collision,
            "new_limb_penetration": new_limb_penetration,
            "new_body_penetration": new_body_penetration,
            "limb_penetration_depth": limb_penetration_depth,
            "body_penetration_depth": body_penetration_depth,
            "roll": roll,
            "pitch": pitch,
        }
        self._stair_state_cache_step = self.common_step_counter
        return self._stair_state_cache

    def _stair_episode_metrics(self, env_ids):
        stair_ids = env_ids[self._stair_env_mask()[env_ids]]
        zero = torch.zeros((), device=self.device)
        if stair_ids.numel() == 0:
            return {
                "stairs_mean_passed_steps": zero,
                "stairs_max_passed_steps": zero,
                "stairs_success_rate": zero,
                "stairs_max_passed_height": zero,
                "stairs_rear_leg_failure_ratio": zero,
                "stairs_rear_slip_ratio": zero,
                "stairs_rear_slip_RL": zero,
                "stairs_rear_slip_RR": zero,
                "stairs_rear_contact_speed_mean_RL": zero,
                "stairs_rear_contact_speed_mean_RR": zero,
                "stairs_rear_contact_speed_p95_RL": zero,
                "stairs_rear_contact_speed_p95_RR": zero,
                "stairs_rear_slide_distance_RL": zero,
                "stairs_rear_slide_distance_RR": zero,
                "stairs_rear_slip_duration_RL": zero,
                "stairs_rear_slip_duration_RR": zero,
                "stairs_rear_slip_time_ratio_RL": zero,
                "stairs_rear_slip_time_ratio_RR": zero,
                "stairs_top_landing_FL": zero,
                "stairs_top_landing_FR": zero,
                "stairs_top_landing_RL": zero,
                "stairs_top_landing_RR": zero,
                "stairs_front_collision_count": zero,
                "stairs_calf_collision_count": zero,
                "stairs_soft_limb_penetration_count": zero,
                "stairs_soft_body_penetration_count": zero,
                "stairs_soft_limb_penetration_mean_depth": zero,
                "stairs_soft_body_penetration_mean_depth": zero,
                "stairs_soft_limb_penetration_max_depth": zero,
                "stairs_soft_body_penetration_max_depth": zero,
                "stairs_max_pitch": zero,
                "stairs_max_roll": zero,
            }

        passed = self.stair_max_passed_steps[stair_ids]
        success = passed >= self.cfg.terrain.stair_step_count
        failed = ~success
        rear_failure = (
            self.stair_front_max_steps[stair_ids]
            > self.stair_rear_max_steps[stair_ids]
        ) & failed
        failed_count = torch.clamp(failed.float().sum(), min=1.0)
        top_ratio = torch.clamp(
            self.stair_foot_top_landings[stair_ids]
            / self.cfg.terrain.stair_step_count,
            0.0, 1.0,
        ).mean(dim=0)
        rear_counts = self.stair_rear_contact_sample_count[stair_ids]
        safe_rear_counts = torch.clamp(rear_counts, min=1)
        rear_mean_speed = (
            self.stair_rear_contact_speed_sum[stair_ids]
            / safe_rear_counts
        )
        sample_values = self.stair_rear_contact_speed_samples[
            stair_ids
        ].nan_to_num(nan=float("inf"))
        sorted_speeds = torch.sort(sample_values, dim=1).values
        p95_indices = torch.clamp(
            torch.ceil(0.95 * rear_counts.float()).long() - 1,
            min=0,
            max=sorted_speeds.shape[1] - 1,
        )
        rear_speed_p95 = torch.gather(
            sorted_speeds, 1, p95_indices.unsqueeze(1)
        ).squeeze(1)
        rear_speed_p95 = torch.where(
            rear_counts > 0,
            rear_speed_p95,
            torch.zeros_like(rear_speed_p95),
        )
        rear_slip_duration = (
            self.stair_rear_slip_steps[stair_ids].float() * self.dt
        )
        rear_contact_duration = rear_counts.float() * self.dt
        rear_slip_time_ratio = rear_slip_duration / torch.clamp(
            rear_contact_duration, min=self.dt
        )
        return {
            "stairs_mean_passed_steps": passed.float().mean(),
            "stairs_max_passed_steps": passed.max().float(),
            "stairs_success_rate": success.float().mean(),
            "stairs_max_passed_height": torch.max(
                passed.float()
                * self.stair_episode_step_height[stair_ids]
            ),
            "stairs_rear_leg_failure_ratio": (
                rear_failure.float().sum() / failed_count
            ),
            "stairs_rear_slip_ratio": (
                (
                    self.stair_rear_slip_episode[stair_ids]
                    & failed
                ).float().sum()
                / failed_count
            ),
            "stairs_rear_slip_RL": (
                self.stair_rear_slip_by_foot[stair_ids, 0]
                .float().mean()
            ),
            "stairs_rear_slip_RR": (
                self.stair_rear_slip_by_foot[stair_ids, 1]
                .float().mean()
            ),
            "stairs_rear_contact_speed_mean_RL": (
                rear_mean_speed[:, 0].mean()
            ),
            "stairs_rear_contact_speed_mean_RR": (
                rear_mean_speed[:, 1].mean()
            ),
            "stairs_rear_contact_speed_p95_RL": (
                rear_speed_p95[:, 0].mean()
            ),
            "stairs_rear_contact_speed_p95_RR": (
                rear_speed_p95[:, 1].mean()
            ),
            "stairs_rear_slide_distance_RL": (
                self.stair_rear_slide_distance[stair_ids, 0].mean()
            ),
            "stairs_rear_slide_distance_RR": (
                self.stair_rear_slide_distance[stair_ids, 1].mean()
            ),
            "stairs_rear_slip_duration_RL": (
                rear_slip_duration[:, 0].mean()
            ),
            "stairs_rear_slip_duration_RR": (
                rear_slip_duration[:, 1].mean()
            ),
            "stairs_rear_slip_time_ratio_RL": (
                rear_slip_time_ratio[:, 0].mean()
            ),
            "stairs_rear_slip_time_ratio_RR": (
                rear_slip_time_ratio[:, 1].mean()
            ),
            "stairs_top_landing_FL": top_ratio[0],
            "stairs_top_landing_FR": top_ratio[1],
            "stairs_top_landing_RL": top_ratio[2],
            "stairs_top_landing_RR": top_ratio[3],
            "stairs_front_collision_count": (
                self.stair_front_collision_count[stair_ids].mean()
            ),
            "stairs_calf_collision_count": (
                self.stair_calf_collision_count[stair_ids].mean()
            ),
            "stairs_soft_limb_penetration_count": (
                self.stair_soft_limb_penetration_count[
                    stair_ids
                ].mean()
            ),
            "stairs_soft_body_penetration_count": (
                self.stair_soft_body_penetration_count[
                    stair_ids
                ].mean()
            ),
            "stairs_soft_limb_penetration_mean_depth": (
                self.stair_soft_limb_penetration_depth_sum[stair_ids]
                / torch.clamp(
                    self.stair_soft_limb_penetration_samples[stair_ids],
                    min=1.0,
                )
            ).mean(),
            "stairs_soft_body_penetration_mean_depth": (
                self.stair_soft_body_penetration_depth_sum[stair_ids]
                / torch.clamp(
                    self.stair_soft_body_penetration_samples[stair_ids],
                    min=1.0,
                )
            ).mean(),
            "stairs_soft_limb_penetration_max_depth": (
                self.stair_soft_limb_penetration_max_depth[
                    stair_ids
                ].mean()
            ),
            "stairs_soft_body_penetration_max_depth": (
                self.stair_soft_body_penetration_max_depth[
                    stair_ids
                ].mean()
            ),
            "stairs_max_pitch": self.stair_max_pitch[stair_ids].mean(),
            "stairs_max_roll": self.stair_max_roll[stair_ids].mean(),
        }

    def _reset_stair_episode_trackers(self, env_ids):
        self.stair_episode_start_x[env_ids] = self.root_states[env_ids, 0]
        self.stair_last_root_height[env_ids] = self.root_states[env_ids, 2]
        self.stair_episode_step_height[env_ids] = (
            self._stair_step_heights()[env_ids]
        )
        self.stair_max_passed_steps[env_ids] = 0
        self.stair_front_max_steps[env_ids] = 0
        self.stair_rear_max_steps[env_ids] = 0
        self.stair_foot_max_levels[env_ids] = 0
        self.stair_foot_support_levels[env_ids] = 0
        self.stair_rear_stable_candidate_levels[env_ids] = 0
        self.stair_rear_stable_contact_steps[env_ids] = 0
        self.stair_rear_rewarded_levels[env_ids] = 0
        self.stair_rear_follow_target_levels[env_ids] = 0
        self.stair_rear_follow_active[env_ids] = False
        self.stair_rear_follow_elapsed_steps[env_ids] = 0
        self.stair_rear_follow_clearance_rewarded[env_ids] = False
        self.stair_last_rear_foot_pos[env_ids] = 0
        self.stair_rear_progress_initialized[env_ids] = False
        self.stair_rear_target_best_distance[env_ids] = float("inf")
        self.stair_rear_target_riser_x[env_ids] = 0
        self.stair_rear_target_riser_valid[env_ids] = False
        self.stair_summit_rewarded[env_ids] = False
        self.stair_rear_slip_episode[env_ids] = False
        self.stair_rear_slip_by_foot[env_ids] = False
        self.stair_rear_contact_speed_sum[env_ids] = 0
        self.stair_rear_contact_sample_count[env_ids] = 0
        self.stair_rear_slide_distance[env_ids] = 0
        self.stair_rear_slip_steps[env_ids] = 0
        self.stair_rear_contact_speed_samples[env_ids] = float("nan")
        self.stair_foot_top_landings[env_ids] = 0
        self.stair_front_collision_count[env_ids] = 0
        self.stair_calf_collision_count[env_ids] = 0
        self.stair_soft_limb_last_penetrating[env_ids] = False
        self.stair_soft_body_last_penetrating[env_ids] = False
        self.stair_soft_limb_penetration_count[env_ids] = 0
        self.stair_soft_body_penetration_count[env_ids] = 0
        self.stair_soft_limb_penetration_depth_sum[env_ids] = 0
        self.stair_soft_body_penetration_depth_sum[env_ids] = 0
        self.stair_soft_limb_penetration_samples[env_ids] = 0
        self.stair_soft_body_penetration_samples[env_ids] = 0
        self.stair_soft_limb_penetration_max_depth[env_ids] = 0
        self.stair_soft_body_penetration_max_depth[env_ids] = 0
        self.stair_last_foot_contacts[env_ids] = False
        self.stair_last_calf_contacts[env_ids] = False
        self.stair_max_pitch[env_ids] = 0
        self.stair_max_roll[env_ids] = 0
        self._stair_state_cache_step = -1

    def _reward_stair_height_progress(self):
        state = self._get_stair_training_state()
        upward_velocity = torch.clamp(
            state["root_height_delta"] / self.dt, 0.0, 0.5
        )
        return upward_velocity * state["mask"]

    def _reward_stair_step_progress(self):
        state = self._get_stair_training_state()
        return state["new_step_progress"] / self.dt * state["mask"]

    def _reward_rear_step_progress(self):
        state = self._get_stair_training_state()
        return (
            state["new_rear_step_progress"] / self.dt * state["mask"]
        )

    def _reward_rear_stable_step(self):
        state = self._get_stair_training_state()
        events = state["new_rear_stable_step"].float().sum(dim=1)
        return events / self.dt * state["mask"]

    def _reward_rear_follow_clearance(self):
        state = self._get_stair_training_state()
        events = state["new_rear_follow_clearance"].float().sum(dim=1)
        return events / self.dt * state["mask"]

    def _reward_rear_follow_timeout(self):
        state = self._get_stair_training_state()
        events = state["new_rear_follow_timeout"].float().sum(dim=1)
        return events / self.dt * state["mask"]

    def _reward_rear_follow_forward(self):
        state = self._get_stair_training_state()
        progress = state["rear_follow_forward_progress"].sum(dim=1)
        return progress / self.dt * state["mask"]

    def _reward_rear_follow_up(self):
        state = self._get_stair_training_state()
        progress = state["rear_follow_up_progress"].sum(dim=1)
        return progress / self.dt * state["mask"]

    def _reward_rear_target_progress(self):
        state = self._get_stair_training_state()
        progress = state["rear_target_progress"].sum(dim=1)
        return progress / self.dt * state["mask"]

    def _reward_stair_summit(self):
        state = self._get_stair_training_state()
        return state["new_summit"].float() / self.dt * state["mask"]

    def _reward_rear_foot_landing(self):
        state = self._get_stair_training_state()
        rear_landings = state["top_landing"][
            :, self.rear_foot_slots
        ].float().sum(dim=1)
        return rear_landings / self.dt * state["mask"]

    def _reward_swing_foot_clearance(self):
        state = self._get_stair_training_state()
        swing = ~state["foot_contact"]
        target_world_z = (
            state["next_step_top_world_z"]
            + self.cfg.rewards.swing_clearance_margin
        )
        required_lift = torch.clamp(
            target_world_z - state["support_top_world_z"],
            min=1e-6,
        )
        clearance_score = torch.clamp(
            (
                state["foot_world_z"] - state["support_top_world_z"]
            ) / required_lift,
            0.0,
            1.0,
        )
        score = torch.sum(clearance_score * swing, dim=1) / torch.clamp(
            swing.float().sum(dim=1), min=1.0
        )
        return score * state["mask"]

    def _reward_rear_foot_slip(self):
        state = self._get_stair_training_state()
        slip = (
            state["rear_horizontal_speed"].square()
            * state["rear_contact"]
        ).sum(dim=1)
        return slip * state["mask"]

    def _reward_stair_collision(self):
        state = self._get_stair_training_state()
        collision_events = (
            state["new_front_collision"].float().sum(dim=1)
            + state["new_calf_collision"].float().sum(dim=1)
        )
        return collision_events / self.dt * state["mask"]

    def _reward_front_stair_collision(self):
        state = self._get_stair_training_state()
        events = state["new_front_collision"].float().sum(dim=1)
        return events / self.dt * state["mask"]

    def _reward_calf_stair_collision(self):
        state = self._get_stair_training_state()
        events = state["new_calf_collision"].float().sum(dim=1)
        return events / self.dt * state["mask"]

    def _reward_soft_limb_penetration_count(self):
        state = self._get_stair_training_state()
        events = state["new_limb_penetration"].float().sum(dim=1)
        return events / self.dt * state["mask"]

    def _reward_soft_limb_penetration_depth(self):
        state = self._get_stair_training_state()
        allowance = float(self.cfg.terrain.soft_stair_face_recess)
        depth = state["limb_penetration_depth"].sum(dim=1)
        return depth / allowance * state["mask"]

    def _reward_soft_body_penetration_count(self):
        state = self._get_stair_training_state()
        events = state["new_body_penetration"].float().sum(dim=1)
        return events / self.dt * state["mask"]

    def _reward_soft_body_penetration_depth(self):
        state = self._get_stair_training_state()
        allowance = float(self.cfg.terrain.soft_stair_face_recess)
        depth = state["body_penetration_depth"].sum(dim=1)
        return depth / allowance * state["mask"]

    def _reward_stair_attitude(self):
        state = self._get_stair_training_state()
        pitch_excess = torch.relu(
            state["pitch"].abs()
            - self.cfg.rewards.max_stair_pitch
        )
        roll_excess = torch.relu(
            state["roll"].abs() - self.cfg.rewards.max_stair_roll
        )
        return (
            pitch_excess.square() + roll_excess.square()
        ) * state["mask"]

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    # ***************** energy disspation ***************
    def _reward_energy(self):
        # Penalize energy
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_slip(self):
        # Penalize foot sliding: contact foot with high velocity
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # get foot velocities from dof_vel (calf joints: indices 2,5,8,11 for Go1)
        foot_vel = torch.norm(self.dof_vel[:, [2, 5, 8, 11]], dim=1)
        # penalize feet that are in contact but still moving fast
        return torch.sum(contact * foot_vel.unsqueeze(-1), dim=1)

    def _reward_hip_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)


    def _reward_feet_distance(self):
        """Penalize a narrow or crossed left/right support pair.

        The Isaac Gym body order for this Go1 asset is FL, FR, RL, RR.
        Distances are measured in
        the base frame so turning the robot does not change the meaning of
        left and right.  Front and rear pairs are evaluated independently;
        otherwise a wide front pair can hide an overly narrow rear pair.
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        body_pos = rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, :, :3]
        feet_pos_world = body_pos[:, self.feet_indices, :]
        base_pos = self.root_states[:, :3].unsqueeze(1)
        feet_offset = feet_pos_world - base_pos
        batch_size = feet_offset.shape[0]
        num_feet = feet_offset.shape[1]
        feet_offset_flat = feet_offset.view(batch_size * num_feet, 3)
        base_quat_expanded = self.base_quat.unsqueeze(1).expand(batch_size, num_feet, 4).contiguous().view(batch_size * num_feet, 4)
        feet_pos_local_flat = quat_rotate_inverse(base_quat_expanded, feet_offset_flat)
        feet_pos_local = feet_pos_local_flat.view(batch_size, num_feet, 3)

        fl_y = feet_pos_local[:, 0, 1]
        fr_y = feet_pos_local[:, 1, 1]
        rl_y = feet_pos_local[:, 2, 1]
        rr_y = feet_pos_local[:, 3, 1]
        # In the base frame, left y is positive and right y is negative.
        # A correctly ordered pair therefore has a positive separation.
        front_separation = fl_y - fr_y
        rear_separation = rl_y - rr_y

        min_distance = 0.12
        narrow_penalty = (
            torch.relu(min_distance - front_separation)
            + torch.relu(min_distance - rear_separation)
        )
        crossing_penalty = 2.0 * (
            torch.relu(-front_separation)
            + torch.relu(-rear_separation)
        )
        return narrow_penalty + crossing_penalty

    def _change_cmds(self, vx, vy, vang):
        # change command_ranges with the input
        self.commands[:, 0] = vx
        self.commands[:, 1] = vy
        self.commands[:, 2] = vang
        # self.commands[:, 3] = heading

    def _reward_feet_width(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        body_pos = rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, :, :3]
        feet_pos_world = body_pos[:, self.feet_indices, :]
        base_pos = self.root_states[:, :3].unsqueeze(1)
        feet_offset = feet_pos_world - base_pos
        batch_size = feet_offset.shape[0]
        num_feet = feet_offset.shape[1]
        feet_offset_flat = feet_offset.view(batch_size * num_feet, 3)
        base_quat_expanded = self.base_quat.unsqueeze(1).expand(batch_size, num_feet, 4).contiguous().view(batch_size * num_feet, 4)
        feet_pos_local_flat = quat_rotate_inverse(base_quat_expanded, feet_offset_flat)
        feet_pos_local = feet_pos_local_flat.view(batch_size, num_feet, 3)
        fr_y = feet_pos_local[:, 0, 1]
        fl_y = feet_pos_local[:, 1, 1]
        rr_y = feet_pos_local[:, 2, 1]
        rl_y = feet_pos_local[:, 3, 1]
        front_width = torch.abs(fl_y - fr_y)
        hind_width = torch.abs(rl_y - rr_y)
        target_width = 0.3
        return (front_width - target_width)**2 + (hind_width - target_width)**2
