#!/bin/bash
set -euo pipefail

export PYTHONPATH=/home/gjt/isaacgym/python:/home/gjt/legged_gym
python_bin=/home/gjt/miniconda3/envs/py38t230cu121/bin/python
eval_script=/home/gjt/legged_gym/legged_gym/scripts/eval_stairs_joint.py
model510=/home/gjt/legged_gym/logs/rough_go1/Jul26_20-55-31_stairs_critical_50_resume470/model_510.pt
model520=/home/gjt/legged_gym/logs/rough_go1/Jul26_20-55-31_stairs_critical_50_resume470/model_520.pt
output_dir=/tmp/go1_stairs_model510_model520_b_full

mkdir -p "$output_dir"
for seed in 20260723 20260724 20260725; do
    for height in 0.115 0.125 0.130; do
        echo "START seed=${seed} height=${height}"
        env \
            GO1_STAIRS_JOINT_CHECKPOINT="$model510" \
            GO1_STAIRS_PAIR_CHECKPOINT="$model520" \
            GO1_STAIRS_JOINT_EVAL_OUTPUT="$output_dir" \
            GO1_STAIRS_EVAL_EPISODES=100 \
            GO1_STAIRS_EVAL_HEIGHT="$height" \
            GO1_STAIRS_EVAL_SEED="$seed" \
            "$python_bin" "$eval_script" \
            --task go1_stairs_joint --headless
        echo "DONE seed=${seed} height=${height}"
    done
done

echo "STAIRS_MODEL510_MODEL520_B_FULL_DONE"
