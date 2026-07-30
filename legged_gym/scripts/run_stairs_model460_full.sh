#!/bin/bash
set -euo pipefail

export PYTHONPATH=/home/gjt/isaacgym/python:/home/gjt/legged_gym
python_bin=/home/gjt/miniconda3/envs/py38t230cu121/bin/python
eval_script=/home/gjt/legged_gym/legged_gym/scripts/eval_stairs_joint.py
checkpoint=/home/gjt/legged_gym/logs/rough_go1/Jul26_20-12-09_stairs_rear_50_resume420/model_460.pt
output_dir=/tmp/go1_stairs_rear_full/model460

mkdir -p "$output_dir"
for height in 0.115 0.130 0.145 0.165; do
    for seed in 20260723 20260724 20260725; do
        echo "START height=${height} seed=${seed}"
        env \
            GO1_STAIRS_JOINT_CHECKPOINT="$checkpoint" \
            GO1_STAIRS_JOINT_EVAL_OUTPUT="$output_dir" \
            GO1_STAIRS_EVAL_EPISODES=100 \
            GO1_STAIRS_EVAL_HEIGHT="$height" \
            GO1_STAIRS_EVAL_SEED="$seed" \
            "$python_bin" "$eval_script" \
            --task go1_stairs_joint --headless
        echo "DONE height=${height} seed=${seed}"
    done
done

echo "STAIRS_MODEL460_FULL_DONE"
