# legged_gym - Unitree Go1 强化学习训练框架

基于 IsaacGym + RSL-RL 的四足机器人强化学习训练项目，支持平地高速奔跑、台阶攀爬、崎岖地形稳定行走，并集成 RMA（Rapid Motor Adaptation）自适应模块。

---

## 功能特性

| 模块 | 功能 |
|------|------|
| 🏃 平地奔跑 | 最大速度 3.5+ m/s |
| 🪜 台阶攀爬 | 5~10 cm 高度台阶上下 |
| ⛰️ 崎岖地形 | rough terrain 稳定行走/奔跑 |
| 🧠 RMA 自适应 | History Encoder + FiLM 残差调制 |
| 📊 训练可视化 | TensorBoard + matplotlib 曲线 |
| 🎬 回放演示 | play.py 模型效果演示 |
| 📐 批量评测 | 速度/台阶/alpha 扫描自动化评测 |

---

## 项目结构

```
legged_gym/
├── legged_gym/
│   ├── envs/
│   │   ├── base/                    # 基础环境 (legged_robot.py)
│   │   └── go1/
│   │       ├── go1.py              # Go1 环境实现
│   │       └── go1_config.py       # 配置 (terrain/reward/PPO/RMA)
│   ├── algorithms/
│   │   ├── rma_actor_critic.py     # RMA Actor-Critic + FiLM 调制
│   │   └── rma.py                  # RMA 核心模块
│   └── scripts/
│       ├── train.py                # 训练入口
│       ├── play.py                 # 回放演示
│       ├── eval_go1_speed.py       # 平地速度评测
│       ├── eval_stairs_joint.py    # 台阶评测
│       └── eval_rma_alpha.py       # RMA alpha 扫描
├── resources/robots/go1/meshes/    # Go1 mesh (分片，clone后运行 assemble_trunk.sh)
└── logs/rough_go1/7/               # 训练 checkpoint + 评测数据 (⭐ model_880.pt 推荐)
```

> **注意**：`resources/robots/go1/meshes/trunk.dae` 因大小限制分片存储，clone 后执行：
> ```bash
> bash resources/robots/go1/meshes/assemble_trunk.sh
> ```

---

## 环境准备

```bash
# 激活 conda 环境
conda activate py38t230cu121

# IsaacGym 需要手动安装 (参考官方文档)
```

依赖：`torch`, `isaacgym`, `rsl_rl`, `numpy`, `matplotlib`

---

## 快速开始

### 训练

```bash
cd legged_gym
python legged_gym/scripts/train.py --task=go1 --num_envs=1000 --headless
```

- `--headless`：无画面训练
- `--resume`：从 checkpoint 继续训练
- `--experiment_name`：实验名称

### 回放演示

```bash
python legged_gym/scripts/play.py --task=go1 --load_run=7 --checkpoint=880
```

初始相机位置已设置，避免黑屏。

### 速度评测

```bash
# 评测平地 3.5 m/s 命令速度下的稳定性
python legged_gym/scripts/eval_go1_speed.py \
  --load_run=7 --checkpoint=880 \
  --command_x=3.5 --num_envs=100
```

---

## 地形配置 (go1_config.py)

| 地形类型 | 比例 | 参数 |
|----------|------|------|
| Flat | 50% | 平坦路面 |
| Rough | 30% | 高度扰动 ±5→±10 cm (课程学习) |
| Stairs | 20% | 台阶高 3→10 cm (课程学习) |

Terrain curriculum learning 已启用，随训练迭代增加难度。

---

## 奖励函数

| 奖励项 | 说明 |
|--------|------|
| 速度跟踪 | 命令速度误差最小化 |
| 身体姿态稳定 | roll/pitch 平稳 |
| 身体高度保持 | base height 稳定 |
| 足端抬脚 | 足端步高 |
| 足端横向距离 | 防止左右腿交叉步 |
| 能耗惩罚 | 减少电机扭矩消耗 |

---

## RMA 模块 (Rapid Motor Adaptation)

### 结构

```
最近10帧 obs_history
       ↓
History Encoder (Conv1d)
       ↓
Adaptation Module (MLP + LayerNorm)
       ↓
latent vector (32 维, LayerNorm 归一化)
       ↓
FiLM: gamma_raw, beta_raw  (tanh → [-0.1, 0.1])
       ↓
残差调制 Actor 隐藏层:
h = h * (1 + alpha * gamma_raw) + alpha * beta_raw
```

### 稳定性措施

| 措施 | 说明 |
|------|------|
| Latent LayerNorm | 防止 latent 爆炸 |
| FiLM tanh 边界 | gamma/beta 限制在 ±0.1 |
| FiLM 零初始化 | 最后一层 weight/bias = 0，初始等价原 PPO |
| 参数组学习率 | Actor 1e-5 / Critic 1e-4 / RMA 1e-4 / std 1e-4 |
| 梯度裁剪 | RMA grad ≤0.5 / Total grad ≤1.0 |
| Alpha schedule | 0-20 iter: 0 → 20-520 iter: 线性增加到 0.2 |
| 异常保护 | NaN/Inf 或 latent_max_abs>20 → 立即停止并保存诊断 |

### alpha=0 等价性验证

```bash
python test_rma_equivalence.py
```

RMA with alpha=0 与原 PPO 动作差异 < 1e-5 ✅

---

## 训练数据可视化

```bash
# 绘制训练曲线 (奖励、损失、步态)
python plot_training.py --log_dir logs/rough_go1/7
```

自动生成：总奖励、步态奖励、稳定性、速度、地形、episode length、探索率、损失 8 张图。

---

## 训练记录 (logs/rough_go1/7/)

| Checkpoint | 说明 |
|------------|------|
| model_860.pt | 中期训练点 |
| **model_880.pt** | **⭐ 最推荐模型（综合表现最佳）** |
| model_900.pt / 920.pt / 940.pt | 后期微调点 |
| model_958.pt | 最终训练点 |

包含：
- 完整 TensorBoard tfevents 日志
- 各速度/台阶评测 JSON 结果
- 初始配对状态 paired_initial_states

---

## 常用命令速查

```bash
# 检查 RMA 等价性
python test_rma_equivalence.py

# 短训练测试 RMA 是否接入
python legged_gym/scripts/train.py --task=go1 --headless --max_iterations=20

# 打印当前配置 (terrain/reward/PPO)
# train.py 已内置递归配置打印

# 模型加载迁移 (旧 PPO → RMA)
# on_policy_runner.py 支持 strict=False，自动打印加载率
```

---

## 关键文件引用

- RMA Actor-Critic: [rma_actor_critic.py](legged_gym/algorithms/rma_actor_critic.py)
- PPO 参数组 + 梯度裁剪: `rsl_rl/algorithms/ppo.py`
- Alpha schedule + 稳定性检查: `rsl_rl/runners/on_policy_runner.py`
- Go1 环境 + 奖励: [go1.py](legged_gym/envs/go1/go1.py)
- 配置: [go1_config.py](legged_gym/envs/go1/go1_config.py)
