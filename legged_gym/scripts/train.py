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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def print_cfg(cfg, name="Config", indent=0):
    """递归打印配置类的所有参数"""
    prefix = "  " * indent
    if hasattr(cfg, '__dict__'):
        print(f"{prefix}{name}:")
        for key, value in cfg.__dict__.items():
            if not key.startswith('_'):
                if hasattr(value, '__dict__') and not isinstance(value, (int, float, str, list, dict, np.ndarray, torch.Tensor)):
                    print_cfg(value, key, indent + 1)
                else:
                    print(f"{prefix}  {key} = {value}")
    elif isinstance(cfg, dict):
        print(f"{prefix}{name}:")
        for key, value in cfg.items():
            if hasattr(value, '__dict__') and not isinstance(value, (int, float, str, list, dict, np.ndarray, torch.Tensor)):
                print_cfg(value, key, indent + 1)
            else:
                print(f"{prefix}  {key} = {value}")

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # 打印环境配置
    print("\n" + "=" * 80)
    print("Environment Configuration:")
    print("=" * 80)
    print_cfg(env_cfg.terrain, "Terrain")
    print_cfg(env_cfg.commands, "Commands")
    print_cfg(env_cfg.rewards.scales, "Reward Scales")
    print(f"  num_envs = {env_cfg.env.num_envs}")
    print(f"  num_observations = {env_cfg.env.num_observations}")
    print(f"  num_actions = {env_cfg.env.num_actions}")
    
    # 打印训练配置
    print("\n" + "=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print_cfg(train_cfg.policy, "Policy")
    print_cfg(train_cfg.algorithm, "Algorithm")
    print_cfg(train_cfg.runner, "Runner")
    print("=" * 80 + "\n")
    
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=getattr(
            train_cfg.runner, "init_at_random_ep_len", True
        ),
    )

if __name__ == '__main__':
    args = get_args()
    train(args)
