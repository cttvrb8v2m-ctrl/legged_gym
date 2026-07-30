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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.specialized_stair_training = getattr(
            cfg, "specialized_stair_training", False
        )

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
            if getattr(self.cfg, "soft_stair_vertical_faces", False):
                self.vertices, self.triangles = self._recess_stair_risers(
                    self.vertices,
                    self.triangles,
                    float(getattr(
                        self.cfg, "soft_stair_face_recess", 0.02
                    )),
                )
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 1.0])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        if getattr(self.cfg, "pure_rough_terrain", False):
            rough_h = getattr(self.cfg, "rough_height", 0.05)
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-rough_h / 2,
                max_height=rough_h / 2,
                step=0.005,
                downsampled_scale=0.2,
            )
            return terrain
        if self.specialized_stair_training:
            if choice < self.proportions[0]:
                # The 10% preservation terrain is genuinely flat.
                return terrain
            if choice < self.proportions[1]:
                rough_h = getattr(self.cfg, "rough_height", 0.05)
                terrain_utils.random_uniform_terrain(
                    terrain,
                    min_height=-rough_h / 2,
                    max_height=rough_h / 2,
                    step=0.005,
                    downsampled_scale=0.2,
                )
                return terrain

            step_height = self._stair_curriculum_height(difficulty)
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=getattr(self.cfg, "stair_tread_depth", 0.30),
                # Negative height is the ascending direction used by this
                # project when the robot moves along +x.
                step_height=-step_height,
                platform_size=3.0,
            )
            return terrain

        slope = difficulty * 0.4
        # 从配置读取台阶高度范围
        step_min = getattr(self.cfg, 'step_height_min', 0.03)
        step_max = getattr(self.cfg, 'step_height_max', 0.20)
        step_height = step_min + (step_max - step_min) * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # rough terrain: 从配置读取高低差
            rough_h = getattr(self.cfg, 'rough_height', 0.05)
            terrain_utils.random_uniform_terrain(terrain, min_height=-rough_h/2, max_height=rough_h/2, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def _stair_curriculum_height(self, difficulty):
        """Map terrain rows to the three configured stair-height stages."""
        explicit_heights = getattr(
            self.cfg, "stair_curriculum_height_bins", None
        )
        if explicit_heights is not None:
            index = min(
                int(round(difficulty * len(explicit_heights))),
                len(explicit_heights) - 1,
            )
            return explicit_heights[index]

        ranges = getattr(
            self.cfg,
            "stair_stage_height_ranges",
            [[0.06, 0.10], [0.10, 0.13], [0.13, 0.165]],
        )
        # Ten terrain rows use difficulties 0.0 ... 0.9. Boundaries are
        # shared so the curriculum changes continuously between stages.
        if difficulty <= 0.3:
            progress = difficulty / 0.3
            low, high = ranges[0]
        elif difficulty <= 0.6:
            progress = (difficulty - 0.3) / 0.3
            low, high = ranges[1]
        else:
            progress = (difficulty - 0.6) / 0.3
            low, high = ranges[2]
        return low + (high - low) * np.clip(progress, 0.0, 1.0)

    def _recess_stair_risers(self, vertices, triangles, recess):
        """Recess only steep stair-face triangles; keep tread vertices intact."""
        triangle_vertices = vertices[triangles]
        edge_1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
        edge_2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
        normals = np.cross(edge_1, edge_2)
        horizontal_normal = np.linalg.norm(normals[:, :2], axis=1)
        z_span = np.ptp(triangle_vertices[:, :, 2], axis=1)
        configured_heights = np.asarray(
            getattr(
                self.cfg,
                "stair_curriculum_height_bins",
                [0.15, 0.155, 0.16],
            ),
            dtype=np.float32,
        )
        min_height = float(np.min(configured_heights)) - 0.01
        max_height = float(np.max(configured_heights)) + 0.01
        riser_mask = (
            (np.abs(normals[:, 2]) <= 0.05 * np.maximum(
                horizontal_normal, 1e-9
            ))
            & (z_span >= min_height)
            & (z_span <= max_height)
        )
        riser_ids = np.flatnonzero(riser_mask)
        if riser_ids.size == 0:
            raise RuntimeError(
                "soft-stairs requested but no vertical risers were found"
            )

        riser_vertices = triangle_vertices[riser_ids].copy()
        z_values = riser_vertices[:, :, 2]
        z_min = np.min(z_values, axis=1, keepdims=True)
        z_max = np.max(z_values, axis=1, keepdims=True)
        lower_mask = np.isclose(z_values, z_min, atol=1e-6)
        upper_mask = np.isclose(z_values, z_max, atol=1e-6)
        lower_xy = np.sum(
            riser_vertices[:, :, :2] * lower_mask[:, :, None], axis=1
        ) / np.maximum(
            np.sum(lower_mask, axis=1, keepdims=True), 1
        )
        upper_xy = np.sum(
            riser_vertices[:, :, :2] * upper_mask[:, :, None], axis=1
        ) / np.maximum(
            np.sum(upper_mask, axis=1, keepdims=True), 1
        )
        toward_upper = upper_xy - lower_xy
        toward_upper /= np.maximum(
            np.linalg.norm(toward_upper, axis=1, keepdims=True), 1e-9
        )
        riser_vertices[:, :, :2] += (
            toward_upper[:, None, :] * recess
        )

        first_new_vertex = vertices.shape[0]
        new_vertices = riser_vertices.reshape(-1, 3)
        new_indices = np.arange(
            first_new_vertex,
            first_new_vertex + new_vertices.shape[0],
            dtype=triangles.dtype,
        ).reshape(-1, 3)
        soft_triangles = triangles.copy()
        soft_triangles[riser_ids] = new_indices
        self.soft_stair_riser_triangle_count = int(riser_ids.size)
        return (
            np.concatenate((vertices, new_vertices), axis=0),
            soft_triangles,
        )

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
