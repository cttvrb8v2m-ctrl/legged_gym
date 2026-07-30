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
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 Unitree, Zeren Luo

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go1RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 235  # 48 base + 187 height points
        num_envs = 1500

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True
        curriculum = True
        max_init_terrain_level = 4
        # [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.15, 0.5, 0.15, 0.1, 0.1]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.32]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_joint': 30, 'thigh_joint': 50., 'calf_joint': 50. }  # [N*m/rad]
        damping = {'hip_joint': 2., 'thigh_joint': 2., 'calf_joint': 2. }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/go1_net.pt"

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-0.5, 3.]
        randomize_limb_mass = True
        added_limb_percentage = [-0.2, 0.2]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.00005
            dof_pos_limits = -1.0
            orientation = -1.5           # 提高: 身体更稳
            feet_width = -1.0
            feet_air_time = 1.0          # 提高: 鼓励更快抬腿频率
            lin_vel_z = -2.0
            collision = -0.1
            action_rate = -0.03         # 启用: 动作平滑
            feet_contact_forces = -0.01  # 新增: 软着陆更稳
            feet_distance = -1.0         # 新增: 防止左右脚交叉

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 100.0
    
    class rma:
        history_length = 10
        latent_dim = 32


class Go1RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        anchor_loss_coef = 1e-3
        film_reg_loss_coef = 1e-5

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_go1'
        max_iterations = 3000
        use_rma = True
        freeze_actor_iters = 500
        resume = False
        load_run = '1'
        checkpoint = 550
        rma_migration_checkpoint = '/home/gjt/legged_gym/logs/rough_go1/1/model_550.pt'


class Go1StairsJointCfg(Go1RoughCfg):
    """Stair-skill curriculum that preserves rough and flat locomotion."""

    class env(Go1RoughCfg.env):
        num_envs = 512

    class terrain(Go1RoughCfg.terrain):
        specialized_stair_training = True
        num_rows = 10
        num_cols = 10
        max_init_terrain_level = 3
        # flat / rough / ascending stairs / unused / unused
        terrain_proportions = [0.10, 0.20, 0.70, 0.0, 0.0]
        stair_stage_height_ranges = [
            [0.06, 0.10],
            [0.10, 0.13],
            [0.13, 0.165],
        ]
        stair_tread_depth = 0.30
        stair_step_count = 9
        rough_height = 0.05

    class commands(Go1RoughCfg.commands):
        curriculum = False

        class ranges(Go1RoughCfg.commands.ranges):
            lin_vel_x = [0.4, 0.8]
            lin_vel_y = [-0.1, 0.1]
            ang_vel_yaw = [-0.2, 0.2]
            heading = [-3.14, 3.14]

    class domain_rand(Go1RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [1.0, 1.0]
        randomize_base_mass = False
        randomize_limb_mass = False
        push_robots = False

    class rewards(Go1RoughCfg.rewards):
        stair_collision_force_threshold = 5.0
        stair_top_surface_min_height = 0.02
        stair_top_contact_height_tolerance = 0.04
        rear_landing_vertical_speed_threshold = 0.15
        rear_landing_horizontal_speed_threshold = 0.20
        rear_slip_velocity_threshold = 0.20
        swing_clearance_margin = 0.025
        max_stair_pitch = 0.55
        max_stair_roll = 0.35

        class scales(Go1RoughCfg.rewards.scales):
            # All stair-specific terms are additive and configuration-driven.
            stair_height_progress = 0.5
            stair_step_progress = 0.5
            rear_foot_landing = 0.3
            swing_foot_clearance = 0.0
            rear_foot_slip = -0.05
            stair_collision = -0.10
            stair_attitude = -1.0


class Go1StairsJointCfgPPO(Go1RoughCfgPPO):
    class algorithm(Go1RoughCfgPPO.algorithm):
        actor_learning_rate = 1e-6
        critic_learning_rate = 1e-4
        rma_learning_rate = 1e-4
        std_learning_rate = 1e-4
        anchor_loss_during_joint_training = True

    class runner(Go1RoughCfgPPO.runner):
        run_name = 'stairs_joint_smoke_from_model300'
        experiment_name = 'rough_go1'
        max_iterations = 10
        save_interval = 10
        resume = True
        load_run = 'Jul23_16-59-37_anchor_filmreg_resume100_to300_128'
        checkpoint = 300
        use_rma = True
        joint_stair_training = True
        joint_training_alpha = 0.03
        freeze_std_during_joint_training = True
        rma_stats_interval = 1


class Go1StairsRearCfg(Go1StairsJointCfg):
    """Conservative rear-leg stair curriculum resumed from model_410."""

    class env(Go1StairsJointCfg.env):
        num_envs = 512

    class terrain(Go1StairsJointCfg.terrain):
        # flat / rough / ascending stairs / unused / unused
        terrain_proportions = [0.10, 0.10, 0.80, 0.0, 0.0]
        stair_stage_height_ranges = [
            [0.09, 0.115],
            [0.105, 0.130],
            [0.120, 0.145],
        ]

    class rewards(Go1StairsJointCfg.rewards):
        stair_top_contact_height_tolerance = 0.04
        rear_landing_vertical_speed_threshold = 0.15
        rear_landing_horizontal_speed_threshold = 0.20

        class scales(Go1StairsJointCfg.rewards.scales):
            rear_step_progress = 0.15
            rear_foot_landing = 0.15
            rear_foot_slip = -0.05
            # Split the former combined stair_collision term so that front
            # feet and calves can be monitored and tuned independently.
            stair_collision = 0.0
            front_stair_collision = -0.03
            calf_stair_collision = -0.05


class Go1StairsRearCfgPPO(Go1StairsJointCfgPPO):
    class algorithm(Go1StairsJointCfgPPO.algorithm):
        actor_learning_rate = 1e-6
        critic_learning_rate = 1e-4
        rma_learning_rate = 1e-5
        std_learning_rate = 1e-4
        anchor_loss_during_joint_training = True

    class runner(Go1StairsJointCfgPPO.runner):
        run_name = "stairs_rear_smoke_from_model410"
        max_iterations = 10
        save_interval = 5
        resume = True
        load_run = "Jul26_18-44-30_stairs_joint_100_resume310"
        checkpoint = 410
        joint_stair_training = True
        joint_training_alpha = 0.03
        freeze_std_during_joint_training = True
        rma_stats_interval = 1


class Go1StairsCriticalCfg(Go1StairsRearCfg):
    """Focus training on the 0.115--0.130 m transition region."""

    class env(Go1StairsRearCfg.env):
        num_envs = 512

    class terrain(Go1StairsRearCfg.terrain):
        # flat / rough / ascending stairs / unused / unused
        terrain_proportions = [0.05, 0.10, 0.85, 0.0, 0.0]
        # Ten curriculum rows: 30% / 40% / 30% across the requested bands.
        stair_curriculum_height_bins = [
            0.105, 0.110, 0.115,
            0.115, 0.120, 0.120, 0.125,
            0.125, 0.130, 0.135,
        ]

    class rewards(Go1StairsRearCfg.rewards):
        rear_slip_velocity_threshold = 0.20

        class scales(Go1StairsRearCfg.rewards.scales):
            # A small increase only; all other rear-leg reward scales stay
            # identical to the validated model_460 training configuration.
            rear_foot_slip = -0.06


class Go1StairsCriticalCfgPPO(Go1StairsRearCfgPPO):
    class algorithm(Go1StairsRearCfgPPO.algorithm):
        actor_learning_rate = 1e-6
        critic_learning_rate = 1e-4
        rma_learning_rate = 1e-5
        std_learning_rate = 1e-4
        anchor_loss_during_joint_training = True

    class runner(Go1StairsRearCfgPPO.runner):
        run_name = "stairs_critical_smoke_from_model460"
        max_iterations = 10
        save_interval = 5
        resume = True
        load_run = "Jul26_20-12-09_stairs_rear_50_resume420"
        checkpoint = 460
        joint_stair_training = True
        joint_training_alpha = 0.03
        freeze_std_during_joint_training = True
        rma_stats_interval = 1


class Go1StairsBasicCfg(Go1RoughCfg):
    """Minimal model_550 stair-skill training on 0.09--0.105 m steps."""

    class env(Go1RoughCfg.env):
        num_envs = 512

    class terrain(Go1RoughCfg.terrain):
        specialized_stair_training = True
        num_rows = 4
        num_cols = 10
        max_init_terrain_level = 3
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
        stair_curriculum_height_bins = [0.09, 0.095, 0.100, 0.105]
        stair_tread_depth = 0.30
        stair_step_count = 9

    class commands(Go1RoughCfg.commands):
        curriculum = False

        class ranges(Go1RoughCfg.commands.ranges):
            lin_vel_x = [0.4, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-3.14, 3.14]

    class domain_rand(Go1RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [1.0, 1.0]
        randomize_base_mass = False
        randomize_limb_mass = False
        push_robots = False

    class rewards(Go1RoughCfg.rewards):
        stair_collision_force_threshold = 5.0
        stair_top_surface_min_height = 0.02
        stair_top_contact_height_tolerance = 0.04
        rear_landing_vertical_speed_threshold = 0.15
        rear_landing_horizontal_speed_threshold = 0.20
        rear_slip_velocity_threshold = 0.20
        rear_stable_contact_steps = 3

        class scales(Go1RoughCfg.rewards.scales):
            stair_step_progress = 0.5
            rear_stable_step = 0.2
            stair_summit = 0.5


class Go1StairsBasicCfgPPO(Go1RoughCfgPPO):
    class algorithm(Go1RoughCfgPPO.algorithm):
        actor_learning_rate = 1e-6
        critic_learning_rate = 1e-4
        rma_learning_rate = 1e-4
        std_learning_rate = 1e-4
        anchor_loss_during_joint_training = False

    class runner(Go1RoughCfgPPO.runner):
        run_name = "stairs_basic_model550_200"
        experiment_name = "rough_go1"
        config_source_task = "go1"
        max_iterations = 200
        save_interval = 20
        resume = False
        use_rma = True
        rma_migration_checkpoint = (
            "/home/gjt/legged_gym/logs/rough_go1/6/model_550.pt"
        )
        joint_stair_training = True
        joint_training_alpha = 0.0
        freeze_std_during_joint_training = True
        freeze_rma_during_joint_training = True
        rma_stats_interval = 20


class Go1StairsMidCfg(Go1StairsBasicCfg):
    """Fixed 20/60/20 sampling in the 0.105--0.130 m range."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 10
        max_init_terrain_level = 9
        freeze_terrain_level_distribution = True
        stair_curriculum_height_bins = [
            # 20%: 0.105--0.115 m
            0.1075, 0.1125,
            # 60%: 0.115--0.125 m
            0.1158, 0.1175, 0.1192, 0.1208, 0.1225, 0.1242,
            # 20%: 0.125--0.130 m
            0.12625, 0.12875,
        ]


class Go1StairsMidCfgPPO(Go1StairsBasicCfgPPO):
    class runner(Go1StairsBasicCfgPPO.runner):
        run_name = "stairs_mid_300_resume_basic200"
        max_iterations = 300
        save_interval = 100
        resume = True
        load_run = "Jul27_08-56-44_stairs_basic_model550_200"
        checkpoint = 200


class Go1StairsHighCfg(Go1StairsBasicCfg):
    """Fixed 20/60/20 sampling in the 0.125--0.140 m range."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 10
        max_init_terrain_level = 9
        freeze_terrain_level_distribution = True
        stair_curriculum_height_bins = [
            # 20%: 0.125--0.130 m
            0.12625, 0.12875,
            # 60%: 0.130--0.135 m
            0.1304, 0.13125, 0.1321, 0.1329, 0.13375, 0.1346,
            # 20%: 0.135--0.140 m
            0.13625, 0.13875,
        ]


class Go1StairsHighCfgPPO(Go1StairsBasicCfgPPO):
    class runner(Go1StairsBasicCfgPPO.runner):
        run_name = "stairs_high_300_resume_mid500"
        max_iterations = 300
        save_interval = 100
        resume = True
        load_run = "Jul27_09-16-38_stairs_mid_300_resume_basic200"
        checkpoint = 500


class Go1StairsHigherCfg(Go1StairsBasicCfg):
    """Fixed 20/60/20 sampling in the 0.135--0.150 m range."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 10
        max_init_terrain_level = 9
        freeze_terrain_level_distribution = True
        stair_curriculum_height_bins = [
            # 20%: 0.135--0.140 m
            0.13625, 0.13875,
            # 60%: 0.140--0.145 m
            0.1404, 0.14125, 0.1421, 0.1429, 0.14375, 0.1446,
            # 20%: 0.145--0.150 m
            0.14625, 0.14875,
        ]


class Go1StairsHigherCfgPPO(Go1StairsBasicCfgPPO):
    class runner(Go1StairsBasicCfgPPO.runner):
        run_name = "stairs_higher_300_resume_high800"
        max_iterations = 300
        save_interval = 100
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800


class Go1StairsStumbleCfg(Go1StairsHigherCfg):
    """Closed stumble experiment; the public reward remains available."""

    class rewards(Go1StairsHigherCfg.rewards):
        class scales(Go1StairsHigherCfg.rewards.scales):
            stumble = 0.0


class Go1StairsStumbleCfgPPO(Go1StairsHigherCfgPPO):
    class runner(Go1StairsHigherCfgPPO.runner):
        run_name = "stairs_stumble_m003_resume_high800_part1"
        max_iterations = 50
        save_interval = 50
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800


class Go1StairsSwingClearanceCfg(Go1StairsHigherCfg):
    """Single-variable next-step swing-clearance experiment."""

    class rewards(Go1StairsHigherCfg.rewards):
        swing_clearance_margin = 0.025

        class scales(Go1StairsHigherCfg.rewards.scales):
            swing_foot_clearance = 0.0


class Go1StairsSwingClearanceCfgPPO(Go1StairsHigherCfgPPO):
    class runner(Go1StairsHigherCfgPPO.runner):
        run_name = "stairs_swing_clearance_005_resume_high800_part1"
        max_iterations = 50
        save_interval = 50
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800


class Go1StairsHeightPushCfg(Go1StairsBasicCfg):
    """Model-800 continuation focused on 0.145--0.160 m stairs."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 20
        max_init_terrain_level = 19
        freeze_terrain_level_distribution = True
        stair_curriculum_height_bins = [
            # 25%: 0.145--0.150 m
            0.1455, 0.1465, 0.1475, 0.1485, 0.1495,
            # 55%: 0.150--0.155 m
            0.150227, 0.150682, 0.151136, 0.151591,
            0.152045, 0.1525, 0.152955, 0.153409,
            0.153864, 0.154318, 0.154773,
            # 20%: 0.155--0.160 m
            0.155625, 0.156875, 0.158125, 0.159375,
        ]

    class commands(Go1StairsBasicCfg.commands):
        class ranges(Go1StairsBasicCfg.commands.ranges):
            lin_vel_x = [0.5, 0.7]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]


class Go1StairsHeightPushCfgPPO(Go1StairsBasicCfgPPO):
    class algorithm(Go1StairsBasicCfgPPO.algorithm):
        actor_learning_rate = 5e-7
        critic_learning_rate = 1e-4

    class runner(Go1StairsBasicCfgPPO.runner):
        run_name = "stairs_height_push_resume800_part1"
        max_iterations = 50
        save_interval = 50
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800


class Go1StairsRearFollowCfg(Go1StairsBasicCfg):
    """Model-800 rear-follow training at the current height frontier."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 20
        max_init_terrain_level = 19
        freeze_terrain_level_distribution = True
        stair_curriculum_height_bins = [
            # 50%: 0.145--0.150 m
            0.145250, 0.145750, 0.146250, 0.146750,
            0.147250, 0.147750, 0.148250, 0.148750,
            0.149250, 0.149750,
            # 35%: fixed 0.155 m
            0.155000, 0.155000, 0.155000, 0.155000,
            0.155000, 0.155000, 0.155000,
            # 15%: fixed 0.160 m
            0.160000, 0.160000, 0.160000,
        ]

    class commands(Go1StairsBasicCfg.commands):
        class ranges(Go1StairsBasicCfg.commands.ranges):
            lin_vel_x = [0.5, 0.7]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(Go1StairsBasicCfg.rewards):
        rear_follow_min_forward_delta = 0.001
        rear_follow_forward_clip_per_step = 0.08
        rear_follow_up_clip_per_step = 0.08

        class scales(Go1StairsBasicCfg.rewards.scales):
            rear_follow_clearance = 0.0
            rear_follow_timeout = 0.0
            rear_follow_forward = 0.10
            rear_follow_up = 0.04
            rear_stable_step = 0.10


class Go1StairsRearFollowCfgPPO(Go1StairsBasicCfgPPO):
    class algorithm(Go1StairsBasicCfgPPO.algorithm):
        actor_learning_rate = 1e-6
        critic_learning_rate = 1e-4
        reference_anchor_enabled = True
        reference_anchor_loss_coef = 0.005

    class runner(Go1StairsBasicCfgPPO.runner):
        run_name = "stairs_rear_follow_dense5_from800"
        max_iterations = 5
        save_interval = 5
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800
        load_optimizer = True
        restore_terrain_levels = False
        reference_anchor_enabled = True
        reference_anchor_max_stair_height = 0.150
        reference_actor_checkpoint = (
            "/home/gjt/legged_gym/logs/rough_go1/"
            "Jul27_09-38-10_stairs_high_300_resume_mid500/model_800.pt"
        )
        rma_stats_interval = 1


class Go1StairsRearTargetCfg(Go1StairsBasicCfg):
    """Goal-directed rear-foot landing experiment from model 800."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 20
        max_init_terrain_level = 19
        freeze_terrain_level_distribution = True
        stair_curriculum_height_bins = [
            # Retain a strong 0.145--0.150 m rehearsal band.
            0.145250, 0.145750, 0.146250, 0.146750,
            0.147250, 0.147750, 0.148250, 0.148750,
            0.149250, 0.149750,
            # Frontier heights.
            0.155000, 0.155000, 0.155000, 0.155000,
            0.155000, 0.155000, 0.155000,
            0.160000, 0.160000, 0.160000,
        ]

    class commands(Go1StairsBasicCfg.commands):
        class ranges(Go1StairsBasicCfg.commands.ranges):
            lin_vel_x = [0.5, 0.7]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(Go1StairsBasicCfg.rewards):
        rear_target_search_extent = 0.45
        rear_target_landing_margin = 0.06
        rear_target_clearance_margin = 0.025
        rear_target_vertical_weight = 0.50
        rear_target_progress_clip_per_step = 0.08

        class scales(Go1StairsBasicCfg.rewards.scales):
            # Disable both failed rear-follow shaping variants in this task.
            rear_follow_clearance = 0.0
            rear_follow_timeout = 0.0
            rear_follow_forward = 0.0
            rear_follow_up = 0.0
            # Conservative potential shaping plus the existing one-shot
            # stable landing event.
            rear_target_progress = 0.08
            rear_stable_step = 0.10


class Go1StairsRearTargetCfgPPO(Go1StairsBasicCfgPPO):
    class algorithm(Go1StairsBasicCfgPPO.algorithm):
        actor_learning_rate = 1e-6
        critic_learning_rate = 1e-4
        reference_anchor_enabled = True
        reference_anchor_loss_coef = 0.005

    class runner(Go1StairsBasicCfgPPO.runner):
        run_name = "stairs_rear_target5_from800"
        max_iterations = 5
        save_interval = 5
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800
        load_optimizer = True
        restore_terrain_levels = False
        reference_anchor_enabled = True
        reference_anchor_max_stair_height = 0.150
        reference_actor_checkpoint = (
            "/home/gjt/legged_gym/logs/rough_go1/"
            "Jul27_09-38-10_stairs_high_300_resume_mid500/model_800.pt"
        )
        rma_stats_interval = 1


class Go1StairsRearTargetMicroCfgPPO(Go1StairsRearTargetCfgPPO):
    """Two-iteration, half-actor-LR continuation from the selected model 805."""

    class algorithm(Go1StairsRearTargetCfgPPO.algorithm):
        actor_learning_rate = 5e-7
        critic_learning_rate = 1e-4

    class runner(Go1StairsRearTargetCfgPPO.runner):
        run_name = "stairs_rear_target_micro2_from805"
        max_iterations = 2
        save_interval = 2
        resume = True
        load_run = "Jul28_12-38-31_stairs_rear_target5_from800"
        checkpoint = 805
        load_optimizer = True
        restore_terrain_levels = False


class Go1StairsRearTarget8Cfg(Go1StairsRearTargetCfg):
    """Single-variable landing-target test: 8 cm inside the tread."""

    class rewards(Go1StairsRearTargetCfg.rewards):
        rear_target_landing_margin = 0.08


class Go1StairsRearTarget8CfgPPO(Go1StairsRearTargetCfgPPO):
    class runner(Go1StairsRearTargetCfgPPO.runner):
        run_name = "stairs_rear_target8cm5_from805"
        max_iterations = 5
        save_interval = 5
        resume = True
        load_run = "Jul28_12-38-31_stairs_rear_target5_from800"
        checkpoint = 805
        load_optimizer = True
        restore_terrain_levels = False


class Go1StairsRearWaypointCfg(Go1StairsRearTargetCfg):
    """Rear foot: clear a fixed riser waypoint, then target the tread."""

    class rewards(Go1StairsRearTargetCfg.rewards):
        rear_target_two_stage = True
        rear_target_waypoint_forward_margin = 0.02
        rear_target_clearance_margin = 0.03
        rear_target_landing_margin = 0.06
        rear_target_landing_height_margin = 0.02


class Go1StairsRearWaypointCfgPPO(Go1StairsRearTargetCfgPPO):
    class runner(Go1StairsRearTargetCfgPPO.runner):
        run_name = "stairs_rear_waypoint3_from805"
        max_iterations = 3
        save_interval = 3
        resume = True
        load_run = "Jul28_12-38-31_stairs_rear_target5_from800"
        checkpoint = 805
        load_optimizer = True
        restore_terrain_levels = False


class Go1StairsRearWaypointContinueCfgPPO(Go1StairsRearWaypointCfgPPO):
    """First guarded 50-iteration continuation from model 808."""

    class runner(Go1StairsRearWaypointCfgPPO.runner):
        run_name = "stairs_rear_waypoint100_from808_part1"
        max_iterations = 50
        save_interval = 50
        resume = True
        load_run = "Jul28_14-27-15_stairs_rear_waypoint3_from805"
        checkpoint = 808
        load_optimizer = True
        restore_terrain_levels = False


class Go1StairsRearWaypointFlatCfg(Go1StairsRearWaypointCfg):
    """Rear-waypoint training with a 10% flat-terrain rehearsal share."""

    class terrain(Go1StairsRearWaypointCfg.terrain):
        # With ten curriculum columns this gives one flat column and nine
        # ascending-stair columns.  The existing stair-height row bins stay
        # unchanged.
        num_cols = 10
        terrain_proportions = [0.10, 0.0, 0.90, 0.0, 0.0]


class Go1StairsRearWaypointFlatCfgPPO(Go1StairsRearWaypointCfgPPO):
    """Train the corrected pairwise feet-distance reward for 100 iterations."""

    class runner(Go1StairsRearWaypointCfgPPO.runner):
        run_name = "stairs_rear_waypoint_flat100_widthfix_from808"
        max_iterations = 100
        save_interval = 50
        resume = True
        load_run = "Jul28_14-27-15_stairs_rear_waypoint3_from805"
        checkpoint = 808
        load_optimizer = True
        restore_terrain_levels = False


class Go1StairsRearWaypointHighSpeedFlatCfg(Go1StairsRearWaypointCfg):
    """Rehearse high-speed flat running without changing stair commands."""

    class terrain(Go1StairsRearWaypointCfg.terrain):
        # Two flat columns and eight stair columns.  The stair height bins and
        # every pre-existing terrain/reward setting remain unchanged.
        num_cols = 10
        terrain_proportions = [0.20, 0.0, 0.80, 0.0, 0.0]

    class commands(Go1StairsRearWaypointCfg.commands):
        terrain_specific_lin_vel_x = True
        high_speed_flat_lin_vel_x = [2.0, 3.0]

        class ranges(Go1StairsRearWaypointCfg.commands.ranges):
            # Used by stair environments; Go1._resample_commands replaces only
            # the flat-environment x command with the high-speed range above.
            lin_vel_x = [0.5, 0.7]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(Go1StairsRearWaypointCfg.rewards):
        rear_touchdown_min_width = 0.12
        rear_touchdown_width_speed_threshold = 2.0

        class scales(Go1StairsRearWaypointCfg.rewards.scales):
            # Conservative event cost; active only on high-speed flat terrain.
            rear_touchdown_width = -0.02


class Go1StairsRearWaypointHighSpeedFlatCfgPPO(
    Go1StairsRearWaypointCfgPPO
):
    class runner(Go1StairsRearWaypointCfgPPO.runner):
        run_name = "stairs_rear_waypoint_highspeed_flat5_from808"
        max_iterations = 5
        save_interval = 5
        resume = True
        load_run = "Jul28_14-27-15_stairs_rear_waypoint3_from805"
        checkpoint = 808
        load_optimizer = True
        restore_terrain_levels = False


class Go1HighSpeedRearWidthCfg(Go1StairsRearWaypointHighSpeedFlatCfg):
    """High-speed flat-only rear touchdown-width calibration."""

    class terrain(Go1StairsRearWaypointHighSpeedFlatCfg.terrain):
        # This experiment intentionally isolates leg-width learning from stair
        # climbing.  The source checkpoint remains the clean model 808.
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]

    class rewards(Go1StairsRearWaypointHighSpeedFlatCfg.rewards):
        # A 13 cm soft floor targets the narrow touchdown tail without imposing
        # a rigid nominal stance.
        rear_touchdown_min_width = 0.13

        class scales(
            Go1StairsRearWaypointHighSpeedFlatCfg.rewards.scales
        ):
            rear_touchdown_width = -0.04


class Go1HighSpeedRearWidthCfgPPO(
    Go1StairsRearWaypointHighSpeedFlatCfgPPO
):
    class algorithm(
        Go1StairsRearWaypointHighSpeedFlatCfgPPO.algorithm
    ):
        # Anchor to the clean source actor while allowing a small touchdown
        # adjustment.  PPO itself is unchanged.
        reference_anchor_enabled = True
        reference_anchor_loss_coef = 0.02

    class runner(Go1StairsRearWaypointHighSpeedFlatCfgPPO.runner):
        run_name = "highspeed_rear_width3_from808"
        max_iterations = 3
        save_interval = 3
        resume = True
        load_run = "Jul28_14-27-15_stairs_rear_waypoint3_from805"
        checkpoint = 808
        # New reward distribution: do not carry stair-optimizer momentum.
        load_optimizer = False
        restore_terrain_levels = False
        reference_anchor_enabled = True
        reference_actor_checkpoint = (
            "/home/gjt/legged_gym/logs/rough_go1/"
            "Jul28_14-27-15_stairs_rear_waypoint3_from805/model_808.pt"
        )


class Go1HighSpeedRearContactWidthCfg(Go1HighSpeedRearWidthCfg):
    """Keep the rear support pair wide during high-speed flat contact."""

    class rewards(Go1HighSpeedRearWidthCfg.rewards):
        class scales(Go1HighSpeedRearWidthCfg.rewards.scales):
            rear_contact_width = -0.05


class Go1HighSpeedRearContactWidthCfgPPO(Go1HighSpeedRearWidthCfgPPO):
    class runner(Go1HighSpeedRearWidthCfgPPO.runner):
        run_name = "highspeed_rear_contact_width10_from808"
        max_iterations = 10
        save_interval = 5


class Go1HighSpeedRearHipResidualCfg(Go1HighSpeedRearWidthCfg):
    """Train only a speed-gated rear-hip widening residual from model 808."""


class Go1HighSpeedRearHipResidualCfgPPO(Go1HighSpeedRearWidthCfgPPO):
    class policy(Go1HighSpeedRearWidthCfgPPO.policy):
        enable_rear_hip_residual = True
        # action_scale=0.25, so 0.08 action is at most 0.02 rad.
        rear_hip_residual_max_action = 0.08
        # Command x is observation index 9 and is scaled by lin_vel=2.0.
        rear_hip_residual_speed_threshold_obs = 4.0
        rear_hip_residual_full_speed_obs = 5.0

    class algorithm(Go1HighSpeedRearWidthCfgPPO.algorithm):
        residual_learning_rate = 1e-4
        reference_anchor_enabled = False

    class runner(Go1HighSpeedRearWidthCfgPPO.runner):
        run_name = "highspeed_rear_hip_residual10_from808"
        max_iterations = 10
        save_interval = 5
        freeze_actor_during_joint_training = True
        freeze_std_during_joint_training = True
        freeze_rma_during_joint_training = True
        reference_anchor_enabled = False


class Go1HighSpeedRearHipResidualContinueCfgPPO(
    Go1HighSpeedRearHipResidualCfgPPO
):
    class runner(Go1HighSpeedRearHipResidualCfgPPO.runner):
        run_name = "highspeed_rear_hip_residual50_from818"
        max_iterations = 40
        save_interval = 20
        load_run = "Jul28_23-28-52_highspeed_rear_hip_residual10_from808"
        checkpoint = 818
        load_optimizer = True


class Go1HighSpeedRearHipPhaseCfg(Go1HighSpeedRearWidthCfg):
    """Multi-speed flat task with a bounded rear touchdown-width band."""

    class commands(Go1HighSpeedRearWidthCfg.commands):
        high_speed_flat_lin_vel_x = [2.0, 3.5]

    class rewards(Go1HighSpeedRearWidthCfg.rewards):
        rear_touchdown_min_width = 0.12
        rear_touchdown_max_width = 0.14


class Go1HighSpeedRearHipPhaseCfgPPO(Go1HighSpeedRearWidthCfgPPO):
    class policy(Go1HighSpeedRearWidthCfgPPO.policy):
        enable_rear_hip_residual = True
        rear_hip_residual_independent = True
        rear_hip_residual_max_action = 0.08
        rear_hip_residual_speed_threshold_obs = 4.0
        rear_hip_residual_full_speed_obs = 5.0

    class algorithm(Go1HighSpeedRearWidthCfgPPO.algorithm):
        residual_learning_rate = 1e-4
        residual_reg_loss_coef = 0.01
        reference_anchor_enabled = False

    class runner(Go1HighSpeedRearWidthCfgPPO.runner):
        run_name = "highspeed_rear_hip_phase50_from808"
        max_iterations = 50
        save_interval = 10
        freeze_actor_during_joint_training = True
        freeze_std_during_joint_training = True
        freeze_rma_during_joint_training = True
        reference_anchor_enabled = False


class Go1HighSpeedRearHipContactBandCfg(Go1HighSpeedRearHipPhaseCfg):
    """Symmetric rear-hip residual with continuous support-width feedback."""

    class rewards(Go1HighSpeedRearHipPhaseCfg.rewards):
        class scales(Go1HighSpeedRearHipPhaseCfg.rewards.scales):
            rear_contact_width = -0.05


class Go1HighSpeedRearHipContactBandCfgPPO(Go1HighSpeedRearWidthCfgPPO):
    class policy(Go1HighSpeedRearWidthCfgPPO.policy):
        enable_rear_hip_residual = True
        rear_hip_residual_independent = False
        rear_hip_residual_max_action = 0.12
        rear_hip_residual_speed_threshold_obs = 4.0
        rear_hip_residual_full_speed_obs = 5.0

    class algorithm(Go1HighSpeedRearWidthCfgPPO.algorithm):
        residual_learning_rate = 1e-4
        residual_reg_loss_coef = 0.005
        reference_anchor_enabled = False

    class runner(Go1HighSpeedRearWidthCfgPPO.runner):
        run_name = "highspeed_rear_hip_contact_band50_from808"
        max_iterations = 50
        save_interval = 10
        freeze_actor_during_joint_training = True
        freeze_std_during_joint_training = True
        freeze_rma_during_joint_training = True
        reference_anchor_enabled = False


class Go1HighSpeedRearHipContactBandContinueCfgPPO(
    Go1HighSpeedRearHipContactBandCfgPPO
):
    class runner(Go1HighSpeedRearHipContactBandCfgPPO.runner):
        run_name = "highspeed_rear_hip_contact_band100_from858"
        max_iterations = 100
        save_interval = 20
        load_run = "Jul29_11-34-45_highspeed_rear_hip_contact_band50_from808"
        checkpoint = 858
        load_optimizer = True


class Go1RMA808RearWidth200Cfg(
    Go1StairsRearWaypointHighSpeedFlatCfg
):
    """RMA-only adaptation from model 808 on stairs plus high-speed flat."""

    class rewards(Go1StairsRearWaypointHighSpeedFlatCfg.rewards):
        rear_touchdown_min_width = 0.13

        class scales(
            Go1StairsRearWaypointHighSpeedFlatCfg.rewards.scales
        ):
            rear_touchdown_width = -0.04


class Go1RMA808RearWidth200CfgPPO(
    Go1StairsRearWaypointHighSpeedFlatCfgPPO
):
    class algorithm(
        Go1StairsRearWaypointHighSpeedFlatCfgPPO.algorithm
    ):
        actor_learning_rate = 1e-6
        critic_learning_rate = 1e-4
        rma_learning_rate = 1e-5
        std_learning_rate = 1e-4
        # Actor is hard-frozen for this RMA-only run, so an additional
        # reference-actor loss is unnecessary.
        reference_anchor_enabled = False

    class runner(Go1StairsRearWaypointHighSpeedFlatCfgPPO.runner):
        run_name = "rma_rear_width200_from808"
        max_iterations = 200
        save_interval = 50
        resume = True
        load_run = "Jul28_14-27-15_stairs_rear_waypoint3_from805"
        checkpoint = 808
        load_optimizer = True
        restore_terrain_levels = False

        # Use the existing migration-stage freeze logic: the absolute
        # checkpoint iteration is 808, so 2000 keeps Actor/std unchanged for
        # this entire 200-iteration experiment while Critic/RMA train.
        joint_stair_training = False
        freeze_actor_iters = 2000
        rma_warmup_iters = 808
        rma_alpha_ramp_iters = 20
        rma_max_alpha = 0.03
        reference_anchor_enabled = False
        rma_stats_interval = 10


class Go1RMAAlpha03RearWidth200CfgPPO(
    Go1RMA808RearWidth200CfgPPO
):
    """Continue model 1008 with the requested stronger alpha=0.3."""

    class runner(Go1RMA808RearWidth200CfgPPO.runner):
        run_name = "rma_alpha03_rear_width200_from1008"
        max_iterations = 200
        save_interval = 50
        resume = True
        load_run = "Jul28_16-42-28_rma_rear_width200_from808"
        checkpoint = 1008
        load_optimizer = True
        restore_terrain_levels = False

        # At absolute iteration 1008 this schedule evaluates to 0.3
        # immediately and remains there. Actor/std stay frozen because the
        # freeze boundary is above the final iteration 1208.
        joint_stair_training = False
        freeze_actor_iters = 2000
        rma_warmup_iters = 988
        rma_alpha_ramp_iters = 20
        rma_max_alpha = 0.3
        reference_anchor_enabled = False
        rma_stats_interval = 10


class Go1SoftStairsCfg(Go1StairsBasicCfg):
    """Independent stair task with only the vertical risers recessed."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 3
        num_cols = 1
        max_init_terrain_level = 2
        freeze_terrain_level_distribution = True
        stair_curriculum_height_bins = [0.150, 0.155, 0.160]
        soft_stair_vertical_faces = True
        soft_stair_face_recess = 0.02

    class commands(Go1StairsBasicCfg.commands):
        class ranges(Go1StairsBasicCfg.commands.ranges):
            lin_vel_x = [0.5, 0.7]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(Go1StairsBasicCfg.rewards):
        class scales(Go1StairsBasicCfg.rewards.scales):
            # New soft-riser penalties only; inherited scales are unchanged.
            soft_limb_penetration_count = -0.01
            soft_limb_penetration_depth = -0.02
            soft_body_penetration_count = -0.05
            soft_body_penetration_depth = -0.10


class Go1SoftStairsCfgPPO(Go1StairsBasicCfgPPO):
    class runner(Go1StairsBasicCfgPPO.runner):
        run_name = "soft_stairs_resume800_50"
        max_iterations = 50
        save_interval = 50
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800


class Go1StairsPeakCfg(Go1StairsBasicCfg):
    """Fixed 40/50/10 sampling in the 0.140--0.155 m range."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 10
        max_init_terrain_level = 9
        freeze_terrain_level_distribution = True
        stair_curriculum_height_bins = [
            # 40%: 0.140--0.145 m
            0.140625, 0.141875, 0.143125, 0.144375,
            # 50%: 0.145--0.150 m
            0.1455, 0.1465, 0.1475, 0.1485, 0.1495,
            # 10%: 0.150--0.155 m
            0.1525,
        ]


class Go1StairsPeakCfgPPO(Go1StairsBasicCfgPPO):
    class algorithm(Go1StairsBasicCfgPPO.algorithm):
        actor_learning_rate = 5e-7
        critic_learning_rate = 5e-5
        rma_learning_rate = 5e-5
        std_learning_rate = 5e-5

    class runner(Go1StairsBasicCfgPPO.runner):
        run_name = "stairs_peak_150_resume_high800_halflr"
        max_iterations = 150
        save_interval = 50
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800


class Go1StairsAnchoredCfg(Go1StairsBasicCfg):
    """High-stair fine-tuning with a 50/35/15 fixed distribution."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 20
        max_init_terrain_level = 19
        freeze_terrain_level_distribution = True
        stair_curriculum_height_bins = [
            # 50%: 0.135--0.145 m
            0.1355, 0.1365, 0.1375, 0.1385, 0.1395,
            0.1405, 0.1415, 0.1425, 0.1435, 0.1445,
            # 35%: 0.145--0.150 m
            0.145357, 0.146071, 0.146786, 0.1475,
            0.148214, 0.148929, 0.149643,
            # 15%: 0.150--0.155 m
            0.150833, 0.1525, 0.154167,
        ]


class Go1StairsAnchoredCfgPPO(Go1StairsBasicCfgPPO):
    class algorithm(Go1StairsBasicCfgPPO.algorithm):
        actor_learning_rate = 1.25e-7
        critic_learning_rate = 1.25e-5
        rma_learning_rate = 1.25e-5
        std_learning_rate = 1.25e-5
        reference_anchor_enabled = True
        reference_anchor_loss_coef = 0.02

    class runner(Go1StairsBasicCfgPPO.runner):
        run_name = "stairs_anchor_ref800_100_freshopt_quarterlr"
        max_iterations = 100
        save_interval = 50
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800
        load_optimizer = False
        reference_anchor_enabled = True
        reference_anchor_max_stair_height = 0.145


class Go1StairsOutcomeCurriculumCfg(Go1StairsBasicCfg):
    """Five-level stair curriculum driven by complete episode outcomes."""

    class terrain(Go1StairsBasicCfg.terrain):
        num_rows = 5
        max_init_terrain_level = 4
        freeze_terrain_level_distribution = False
        stair_outcome_curriculum = True
        stair_curriculum_failure_steps = 6
        stair_curriculum_height_bins = [
            0.135, 0.140, 0.145, 0.150, 0.155,
        ]


class Go1StairsOutcomeCurriculumCfgPPO(Go1StairsAnchoredCfgPPO):
    class runner(Go1StairsAnchoredCfgPPO.runner):
        run_name = "stairs_outcome_curriculum_ref800_part1"
        max_iterations = 100
        save_interval = 100
        init_at_random_ep_len = False
        resume = True
        load_run = "Jul27_09-38-10_stairs_high_300_resume_mid500"
        checkpoint = 800
        load_optimizer = False
        reference_actor_checkpoint = (
            "/home/gjt/legged_gym/logs/rough_go1/"
            "Jul27_09-38-10_stairs_high_300_resume_mid500/model_800.pt"
        )


class Go1StairsOutcomeCurriculumContinueCfgPPO(
    Go1StairsOutcomeCurriculumCfgPPO
):
    class runner(Go1StairsOutcomeCurriculumCfgPPO.runner):
        run_name = "stairs_outcome_curriculum_ref800_part2"
        max_iterations = 100
        save_interval = 100
        init_at_random_ep_len = False
        resume = True
        load_optimizer = True
