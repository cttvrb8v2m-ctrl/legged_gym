#!/bin/bash
set -euo pipefail

export PYTHONPATH=/home/gjt/isaacgym/python:/home/gjt/legged_gym
python_bin=/home/gjt/miniconda3/envs/py38t230cu121/bin/python
eval_script=/home/gjt/legged_gym/legged_gym/scripts/eval_stairs_joint.py
seed=20260723

run_eval() {
    local tag="$1"
    local checkpoint="$2"
    local output_dir="/tmp/go1_stairs_rear_quick/${tag}"
    mkdir -p "$output_dir"
    for height in 0.115 0.130 0.145 0.165; do
        echo "START checkpoint=${tag} height=${height}"
        env \
            GO1_STAIRS_JOINT_CHECKPOINT="$checkpoint" \
            GO1_STAIRS_JOINT_EVAL_OUTPUT="$output_dir" \
            GO1_STAIRS_EVAL_EPISODES=30 \
            GO1_STAIRS_EVAL_HEIGHT="$height" \
            GO1_STAIRS_EVAL_SEED="$seed" \
            "$python_bin" "$eval_script" \
            --task go1_stairs_joint --headless
        echo "DONE checkpoint=${tag} height=${height}"
    done
}

run_eval \
    model420 \
    /home/gjt/legged_gym/logs/rough_go1/Jul26_20-10-27_stairs_rear_smoke_from_model410/model_420.pt
run_eval \
    model435 \
    /home/gjt/legged_gym/logs/rough_go1/Jul26_20-12-09_stairs_rear_50_resume420/model_435.pt
run_eval \
    model460 \
    /home/gjt/legged_gym/logs/rough_go1/Jul26_20-12-09_stairs_rear_50_resume420/model_460.pt
run_eval \
    model470 \
    /home/gjt/legged_gym/logs/rough_go1/Jul26_20-12-09_stairs_rear_50_resume420/model_470.pt

echo "STAIRS_REAR_QUICK_SCREEN_DONE"
