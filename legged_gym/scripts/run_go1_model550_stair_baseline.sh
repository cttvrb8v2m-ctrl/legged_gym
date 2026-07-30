#!/bin/bash
set -euo pipefail

export PYTHONPATH=/home/gjt/isaacgym/python:/home/gjt/legged_gym
python_bin=/home/gjt/miniconda3/envs/py38t230cu121/bin/python
eval_script=/home/gjt/legged_gym/legged_gym/scripts/eval_go1_model550_baseline.py
output_dir=/home/gjt/legged_gym/logs/baselines/go1_model550_stairs

mkdir -p "$output_dir"
for seed in 20260723 20260724 20260725; do
    for height in 0.060 0.090 0.105 0.115 0.125 0.130 0.145 0.165; do
        echo "START seed=${seed} height=${height}"
        env \
            GO1_STAIRS_JOINT_EVAL_OUTPUT="$output_dir" \
            GO1_STAIRS_EVAL_EPISODES=100 \
            GO1_STAIRS_EVAL_HEIGHT="$height" \
            GO1_STAIRS_EVAL_SEED="$seed" \
            "$python_bin" "$eval_script" \
            --task go1_stairs_joint --headless
        echo "DONE seed=${seed} height=${height}"
    done
done

echo "GO1_MODEL550_STAIR_BASELINE_DONE"
