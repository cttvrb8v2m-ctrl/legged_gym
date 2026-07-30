#!/bin/bash
set -euo pipefail

export PYTHONPATH=/home/gjt/isaacgym/python:/home/gjt/legged_gym
python_bin=/home/gjt/miniconda3/envs/py38t230cu121/bin/python
eval_script=/home/gjt/legged_gym/legged_gym/scripts/eval_go1_speed.py
output_dir=/home/gjt/legged_gym/logs/baselines/go1_model550_speed

mkdir -p "$output_dir"
for seed in 20260723 20260724 20260725; do
    for terrain in flat rough; do
        for speed in 0.5 1.0 1.5 2.0 2.5 3.0; do
            echo "START seed=${seed} terrain=${terrain} speed=${speed}"
            env \
                GO1_SPEED_EVAL_OUTPUT="$output_dir" \
                GO1_SPEED_EVAL_EPISODES=100 \
                GO1_SPEED_EVAL_TERRAIN="$terrain" \
                GO1_SPEED_EVAL_COMMAND_X="$speed" \
                GO1_SPEED_EVAL_SEED="$seed" \
                "$python_bin" "$eval_script" \
                --task go1 --headless
            echo "DONE seed=${seed} terrain=${terrain} speed=${speed}"
        done
    done
done

echo "GO1_MODEL550_SPEED_BASELINE_DONE"
