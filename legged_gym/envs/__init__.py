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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1_src.a1_src_config import A1SrcRoughCfgPPO, A1SrcRoughCfg
from .go1.go1_config import (
    Go1RoughCfgPPO,
    Go1RoughCfg,
    Go1StairsJointCfg,
    Go1StairsJointCfgPPO,
    Go1StairsRearCfg,
    Go1StairsRearCfgPPO,
    Go1StairsCriticalCfg,
    Go1StairsCriticalCfgPPO,
    Go1StairsBasicCfg,
    Go1StairsBasicCfgPPO,
    Go1StairsMidCfg,
    Go1StairsMidCfgPPO,
    Go1StairsHighCfg,
    Go1StairsHighCfgPPO,
    Go1StairsHigherCfg,
    Go1StairsHigherCfgPPO,
    Go1StairsStumbleCfg,
    Go1StairsStumbleCfgPPO,
    Go1StairsSwingClearanceCfg,
    Go1StairsSwingClearanceCfgPPO,
    Go1StairsHeightPushCfg,
    Go1StairsHeightPushCfgPPO,
    Go1StairsRearFollowCfg,
    Go1StairsRearFollowCfgPPO,
    Go1StairsRearTargetCfg,
    Go1StairsRearTargetCfgPPO,
    Go1StairsRearTargetMicroCfgPPO,
    Go1StairsRearTarget8Cfg,
    Go1StairsRearTarget8CfgPPO,
    Go1StairsRearWaypointCfg,
    Go1StairsRearWaypointCfgPPO,
    Go1StairsRearWaypointContinueCfgPPO,
    Go1StairsRearWaypointFlatCfg,
    Go1StairsRearWaypointFlatCfgPPO,
    Go1StairsRearWaypointHighSpeedFlatCfg,
    Go1StairsRearWaypointHighSpeedFlatCfgPPO,
    Go1HighSpeedRearWidthCfg,
    Go1HighSpeedRearWidthCfgPPO,
    Go1HighSpeedRearContactWidthCfg,
    Go1HighSpeedRearContactWidthCfgPPO,
    Go1HighSpeedRearHipResidualCfg,
    Go1HighSpeedRearHipResidualCfgPPO,
    Go1HighSpeedRearHipResidualContinueCfgPPO,
    Go1HighSpeedRearHipPhaseCfg,
    Go1HighSpeedRearHipPhaseCfgPPO,
    Go1HighSpeedRearHipContactBandCfg,
    Go1HighSpeedRearHipContactBandCfgPPO,
    Go1HighSpeedRearHipContactBandContinueCfgPPO,
    Go1RMA808RearWidth200Cfg,
    Go1RMA808RearWidth200CfgPPO,
    Go1RMAAlpha03RearWidth200CfgPPO,
    Go1SoftStairsCfg,
    Go1SoftStairsCfgPPO,
    Go1StairsPeakCfg,
    Go1StairsPeakCfgPPO,
    Go1StairsAnchoredCfg,
    Go1StairsAnchoredCfgPPO,
    Go1StairsOutcomeCurriculumCfg,
    Go1StairsOutcomeCurriculumCfgPPO,
    Go1StairsOutcomeCurriculumContinueCfgPPO,
)
from .go1.go1 import Go1
from .aliengo.aliengo_config import AliengoRoughCfgPPO, AliengoRoughCfg
from .aliengo.aliengo import Aliengo


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "a1_src", LeggedRobot, A1SrcRoughCfg(), A1SrcRoughCfgPPO() )
task_registry.register( "go1", Go1, Go1RoughCfg(), Go1RoughCfgPPO() )
task_registry.register(
    "go1_stairs_joint",
    Go1,
    Go1StairsJointCfg(),
    Go1StairsJointCfgPPO(),
)
task_registry.register(
    "go1_stairs_rear",
    Go1,
    Go1StairsRearCfg(),
    Go1StairsRearCfgPPO(),
)
task_registry.register(
    "go1_stairs_critical",
    Go1,
    Go1StairsCriticalCfg(),
    Go1StairsCriticalCfgPPO(),
)
task_registry.register(
    "go1_stairs_basic",
    Go1,
    Go1StairsBasicCfg(),
    Go1StairsBasicCfgPPO(),
)
task_registry.register(
    "go1_stairs_mid",
    Go1,
    Go1StairsMidCfg(),
    Go1StairsMidCfgPPO(),
)
task_registry.register(
    "go1_stairs_high",
    Go1,
    Go1StairsHighCfg(),
    Go1StairsHighCfgPPO(),
)
task_registry.register(
    "go1_stairs_higher",
    Go1,
    Go1StairsHigherCfg(),
    Go1StairsHigherCfgPPO(),
)
task_registry.register(
    "go1_stairs_stumble",
    Go1,
    Go1StairsStumbleCfg(),
    Go1StairsStumbleCfgPPO(),
)
task_registry.register(
    "go1_stairs_swing_clearance",
    Go1,
    Go1StairsSwingClearanceCfg(),
    Go1StairsSwingClearanceCfgPPO(),
)
task_registry.register(
    "go1_stairs_height_push",
    Go1,
    Go1StairsHeightPushCfg(),
    Go1StairsHeightPushCfgPPO(),
)
task_registry.register(
    "go1_stairs_rear_follow",
    Go1,
    Go1StairsRearFollowCfg(),
    Go1StairsRearFollowCfgPPO(),
)
task_registry.register(
    "go1_stairs_rear_target",
    Go1,
    Go1StairsRearTargetCfg(),
    Go1StairsRearTargetCfgPPO(),
)
task_registry.register(
    "go1_stairs_rear_target_micro",
    Go1,
    Go1StairsRearTargetCfg(),
    Go1StairsRearTargetMicroCfgPPO(),
)
task_registry.register(
    "go1_stairs_rear_target8",
    Go1,
    Go1StairsRearTarget8Cfg(),
    Go1StairsRearTarget8CfgPPO(),
)
task_registry.register(
    "go1_stairs_rear_waypoint",
    Go1,
    Go1StairsRearWaypointCfg(),
    Go1StairsRearWaypointCfgPPO(),
)
task_registry.register(
    "go1_stairs_rear_waypoint_continue",
    Go1,
    Go1StairsRearWaypointCfg(),
    Go1StairsRearWaypointContinueCfgPPO(),
)
task_registry.register(
    "go1_stairs_rear_waypoint_flat",
    Go1,
    Go1StairsRearWaypointFlatCfg(),
    Go1StairsRearWaypointFlatCfgPPO(),
)
task_registry.register(
    "go1_stairs_rear_waypoint_highspeed_flat",
    Go1,
    Go1StairsRearWaypointHighSpeedFlatCfg(),
    Go1StairsRearWaypointHighSpeedFlatCfgPPO(),
)
task_registry.register(
    "go1_highspeed_rear_width",
    Go1,
    Go1HighSpeedRearWidthCfg(),
    Go1HighSpeedRearWidthCfgPPO(),
)
task_registry.register(
    "go1_highspeed_rear_contact_width",
    Go1,
    Go1HighSpeedRearContactWidthCfg(),
    Go1HighSpeedRearContactWidthCfgPPO(),
)
task_registry.register(
    "go1_highspeed_rear_hip_residual",
    Go1,
    Go1HighSpeedRearHipResidualCfg(),
    Go1HighSpeedRearHipResidualCfgPPO(),
)
task_registry.register(
    "go1_highspeed_rear_hip_residual_continue",
    Go1,
    Go1HighSpeedRearHipResidualCfg(),
    Go1HighSpeedRearHipResidualContinueCfgPPO(),
)
task_registry.register(
    "go1_highspeed_rear_hip_phase",
    Go1,
    Go1HighSpeedRearHipPhaseCfg(),
    Go1HighSpeedRearHipPhaseCfgPPO(),
)
task_registry.register(
    "go1_highspeed_rear_hip_contact_band",
    Go1,
    Go1HighSpeedRearHipContactBandCfg(),
    Go1HighSpeedRearHipContactBandCfgPPO(),
)
task_registry.register(
    "go1_highspeed_rear_hip_contact_band_continue",
    Go1,
    Go1HighSpeedRearHipContactBandCfg(),
    Go1HighSpeedRearHipContactBandContinueCfgPPO(),
)
task_registry.register(
    "go1_rma808_rear_width200",
    Go1,
    Go1RMA808RearWidth200Cfg(),
    Go1RMA808RearWidth200CfgPPO(),
)
task_registry.register(
    "go1_rma_alpha03_rear_width200",
    Go1,
    Go1RMA808RearWidth200Cfg(),
    Go1RMAAlpha03RearWidth200CfgPPO(),
)
task_registry.register(
    "go1_soft_stairs",
    Go1,
    Go1SoftStairsCfg(),
    Go1SoftStairsCfgPPO(),
)
task_registry.register(
    "go1_stairs_peak",
    Go1,
    Go1StairsPeakCfg(),
    Go1StairsPeakCfgPPO(),
)
task_registry.register(
    "go1_stairs_anchored",
    Go1,
    Go1StairsAnchoredCfg(),
    Go1StairsAnchoredCfgPPO(),
)
task_registry.register(
    "go1_stairs_outcome_curriculum",
    Go1,
    Go1StairsOutcomeCurriculumCfg(),
    Go1StairsOutcomeCurriculumCfgPPO(),
)
task_registry.register(
    "go1_stairs_outcome_curriculum_continue",
    Go1,
    Go1StairsOutcomeCurriculumCfg(),
    Go1StairsOutcomeCurriculumContinueCfgPPO(),
)
task_registry.register( "aliengo", Aliengo, AliengoRoughCfg(), AliengoRoughCfgPPO() )
