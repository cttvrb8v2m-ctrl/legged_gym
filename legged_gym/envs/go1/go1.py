from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import nn
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .go1_config import Go1RoughCfg

LEG_NUM = 4
LEG_DOF = 3
LEN_HIST = 5
MODEL_IN_SIZE = 2 * LEG_DOF * LEN_HIST

class UniNet(nn.Module):
    def __init__(self, model):
        super(UniNet, self).__init__()
        self.core_model = model

    def forward(self, x): # x: 4 * MODEL_IN_SIZE;
        # x: q_err_hip_1, dq_hip_1, q_err_thigh_1, dq_thigh_1, q_err_calf_1, dq_calf_1 (30)
        #    q_err_hip_2, dq_hip_2, q_err_thigh_2, dq_thigh_2, q_err_calf_2, dq_calf_2 (30)
        out = torch.tensor(()).to(x.device)
        for i in range(LEG_NUM):
            sub_in = x[:, MODEL_IN_SIZE*i:MODEL_IN_SIZE*(i+1)]
            sub_out = self.core_model(sub_in)
            out = torch.cat((out, sub_out), 1)
        return out

class Go1(LeggedRobot):
    cfg: Go1RoughCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            sub_model = torch.jit.load(actuator_network_path).to(self.device)
            self.actuator_network = UniNet(sub_model)

        # get mean and std of input and output from data
        self.pos_err_mean = torch.tile(torch.tensor([0.00036437, 0.01540757, -0.00972657]), (LEG_NUM, )).to(self.device)
        self.pos_err_std = torch.tile(torch.tensor([0.11722939, 0.19275887, 0.28700321]), (LEG_NUM, )).to(self.device)
        self.vel_mean = torch.tile(torch.tensor([-0.00017714, -0.00024455,  0.0005956 ]), (LEG_NUM, )).to(self.device)
        self.vel_std = torch.tile(torch.tensor([2.31517027, 3.84613839, 5.52599008]), (LEG_NUM, )).to(self.device)


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "rear_touchdown_last_contacts"):
            self.rear_touchdown_last_contacts[env_ids] = False

    def _init_buffers(self):
        super()._init_buffers()
        self.rear_touchdown_last_contacts = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device,
            requires_grad=False
        )
        # Additionally initialize actuator network hidden state tensors
        self.model_ins = torch.zeros(self.num_envs, MODEL_IN_SIZE * LEG_NUM, device=self.device, requires_grad=False) # [all envs * all legs, model-in-DOF]

        # init pos err and vel buffer(12 DOF)
        self.pos_err_buffs = np.zeros((self.num_envs, self.num_actions, LEN_HIST))
        self.vel_buffs = np.zeros((self.num_envs, self.num_actions, LEN_HIST))

    def _resample_commands(self, env_ids):
        """Optionally use terrain-specific forward-speed ranges.

        This is disabled for every existing task.  The dedicated high-speed
        landing-width experiment enables it so stair environments retain their
        low climbing speed while flat environments rehearse 2--3 m/s running.
        """
        super()._resample_commands(env_ids)
        if (
            len(env_ids) == 0
            or not getattr(
                self.cfg.commands, "terrain_specific_lin_vel_x", False
            )
        ):
            return

        stair_mask = self._stair_env_mask()[env_ids]
        flat_env_ids = env_ids[~stair_mask]
        if len(flat_env_ids) > 0:
            speed_range = self.cfg.commands.high_speed_flat_lin_vel_x
            self.commands[flat_env_ids, 0] = torch_rand_float(
                speed_range[0], speed_range[1],
                (len(flat_env_ids), 1), device=self.device
            ).squeeze(1)

    def _reward_rear_touchdown_width(self):
        """Penalize only narrow rear-foot touchdown events on high-speed flat.

        The term is deliberately inactive on stairs.  It enforces a minimum
        landing separation instead of a fixed stance width, and dividing by
        ``dt`` makes the configured scale the cost per touchdown event.
        """
        rear_contacts = (
            self.contact_forces[:, self.feet_indices[2:4], 2] > 1.0
        )
        touchdowns = rear_contacts & ~self.rear_touchdown_last_contacts
        self.rear_touchdown_last_contacts.copy_(rear_contacts)

        flat_high_speed = (
            ~self._stair_env_mask()
            & (
                self.commands[:, 0]
                >= self.cfg.rewards.rear_touchdown_width_speed_threshold
            )
        )
        if not torch.any(flat_high_speed):
            return torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        )
        feet_pos_world = rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, :3]
        feet_offset = feet_pos_world - self.root_states[:, :3].unsqueeze(1)
        base_quat = self.base_quat.unsqueeze(1).expand(
            -1, feet_offset.shape[1], -1
        ).reshape(-1, 4)
        feet_pos_local = quat_rotate_inverse(
            base_quat, feet_offset.reshape(-1, 3)
        ).view(self.num_envs, feet_offset.shape[1], 3)

        # Runtime foot order is FL, FR, RL, RR.
        rear_separation = (
            feet_pos_local[:, 2, 1] - feet_pos_local[:, 3, 1]
        )
        minimum = self.cfg.rewards.rear_touchdown_min_width
        normalized_deficit = torch.clamp(
            (minimum - rear_separation) / minimum, min=0.0, max=1.0
        )
        maximum = getattr(
            self.cfg.rewards, "rear_touchdown_max_width", None
        )
        if maximum is not None:
            normalized_excess = torch.clamp(
                (rear_separation - maximum) / maximum,
                min=0.0,
                max=1.0,
            )
            normalized_deficit = normalized_deficit + normalized_excess
        event_count = touchdowns.float().sum(dim=1)
        return (
            normalized_deficit
            * event_count
            * flat_high_speed.float()
            / self.dt
        )

    def _reward_rear_contact_width(self):
        """Penalize a narrow rear support pair throughout high-speed contact."""
        rear_contacts = (
            self.contact_forces[:, self.feet_indices[2:4], 2] > 1.0
        )
        rear_support = rear_contacts.any(dim=1)
        flat_high_speed = (
            ~self._stair_env_mask()
            & (
                self.commands[:, 0]
                >= self.cfg.rewards.rear_touchdown_width_speed_threshold
            )
        )
        active = rear_support & flat_high_speed
        if not torch.any(active):
            return torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        )
        feet_pos_world = rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, :3]
        feet_offset = feet_pos_world - self.root_states[:, :3].unsqueeze(1)
        base_quat = self.base_quat.unsqueeze(1).expand(
            -1, feet_offset.shape[1], -1
        ).reshape(-1, 4)
        feet_pos_local = quat_rotate_inverse(
            base_quat, feet_offset.reshape(-1, 3)
        ).view(self.num_envs, feet_offset.shape[1], 3)

        rear_separation = (
            feet_pos_local[:, 2, 1] - feet_pos_local[:, 3, 1]
        )
        minimum = self.cfg.rewards.rear_touchdown_min_width
        normalized_deficit = torch.clamp(
            (minimum - rear_separation) / minimum, min=0.0, max=1.0
        )
        maximum = getattr(
            self.cfg.rewards, "rear_touchdown_max_width", None
        )
        if maximum is not None:
            normalized_deficit = normalized_deficit + torch.clamp(
                (rear_separation - maximum) / maximum,
                min=0.0,
                max=1.0,
            )
        return normalized_deficit * active.float()

    def _compute_poses(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            dVel = self.actuator_advance(actions)

            return super()._compute_poses(actions)
        else:
            # pd controller
            return super()._compute_poses(actions)

    # TODO: actuator model buffer and forward
    def actuator_advance(self, actions):
        # scale pos_err and vel TODO: clip
        pos_err = actions - self.dof_pos
        pos_err_s = (pos_err - self.pos_err_mean) / self.pos_err_std
        vel_s = (self.dof_vel - self.vel_mean) / self.vel_std

        # TODO: note that the pos_err is of 12-dim, but real model_in is od=f 3-dim
        model_in = np.array([])
        for i in range(self.num_actions):
            # fill buffers with scaled data [t-h, ... , t-0]
            # hist can be different for each joint
            pos_err_temp = np.delete(self.pos_err_buffs[:, i, :], 0, axis=1)  # need to be numpy,
            self.pos_err_buffs[:, i, :] = np.append(pos_err_temp, pos_err_s[:, i].unsqueeze(-1).cpu().numpy(), axis=1)

            vel_temp = np.delete(self.vel_buffs[:, i, :], 0, axis=1)
            self.vel_buffs[:, i, :] = np.append(vel_temp, vel_s[:, i].unsqueeze(-1).cpu().numpy(), axis=1)

            # fill actuator model input vector
            self.model_ins[:, 2 * i * LEN_HIST:(2 * i + 1) * LEN_HIST] = torch.from_numpy(self.pos_err_buffs[:, i, :])
            self.model_ins[:, (2 * i + 1) * LEN_HIST:(2 * i + 2) * LEN_HIST] = torch.from_numpy(self.vel_buffs[:, i, :])

        with torch.inference_mode():
            # advance actuator mlp
            dVel = self.actuator_network(self.model_ins)

            # upscale mlp output(the dVel mean is counteracted)
            dVel *= self.vel_std

        return dVel



