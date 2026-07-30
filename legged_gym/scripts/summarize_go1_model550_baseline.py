"""Aggregate the immutable model_550 stair and speed baseline JSON files."""

import csv
import glob
import json
import math
import os


ROOT = "/home/gjt/legged_gym/logs/baselines"
STAIR_DIR = os.path.join(ROOT, "go1_model550_stairs")
SPEED_DIR = os.path.join(ROOT, "go1_model550_speed")
EXPECTED_SHA = (
    "03eb303363cc7e260d497a52887cdf719c42ce47b50bec1b713d3b8801fd19c4"
)


def mean(values):
    return sum(values) / len(values) if values else None


def stair_summary(records):
    summit = [
        row["summit_time_s"] for row in records
        if row["summit_time_s"] is not None
    ]
    return {
        "episodes": len(records),
        "success_rate": mean([row["success"] for row in records]),
        "mean_passed_steps": mean(
            [row["passed_steps"] for row in records]
        ),
        "max_passed_steps": max(
            row["passed_steps"] for row in records
        ),
        "mean_summit_time_s_success_only": mean(summit),
        "fall_rate": mean([row["fall"] for row in records]),
        "top_landing_RL": mean(
            [row["foot_top_landing_ratio"][2] for row in records]
        ),
        "top_landing_RR": mean(
            [row["foot_top_landing_ratio"][3] for row in records]
        ),
        "contact_speed_RL": mean(
            [row["rear_contact_mean_speed"][0] for row in records]
        ),
        "contact_speed_RR": mean(
            [row["rear_contact_mean_speed"][1] for row in records]
        ),
        "slide_distance_RL": mean(
            [row["rear_slide_distance"][0] for row in records]
        ),
        "slide_distance_RR": mean(
            [row["rear_slide_distance"][1] for row in records]
        ),
        "slip_time_ratio_RL": mean(
            [row["rear_slip_time_ratio"][0] for row in records]
        ),
        "slip_time_ratio_RR": mean(
            [row["rear_slip_time_ratio"][1] for row in records]
        ),
        "front_collision": mean(
            [row["front_collision_count"] for row in records]
        ),
        "calf_collision": mean(
            [row["calf_collision_count"] for row in records]
        ),
        "max_pitch_rad": mean(
            [row["max_pitch_rad"] for row in records]
        ),
        "max_roll_rad": mean(
            [row["max_roll_rad"] for row in records]
        ),
        "episode_action_p99_mean": mean(
            [row["action_p99"] for row in records]
        ),
        "action_max": max(row["action_max"] for row in records),
    }


def speed_summary(records):
    return {
        "episodes": len(records),
        "mean_x_velocity": mean(
            [row["mean_x_velocity"] for row in records]
        ),
        "mean_abs_tracking_error": mean(
            [row["mean_abs_tracking_error"] for row in records]
        ),
        "tracking_lin_vel": mean(
            [row["tracking_lin_vel"] for row in records]
        ),
        "fall_rate": mean([row["fall"] for row in records]),
        "episode_length": mean(
            [row["episode_length"] for row in records]
        ),
        "reward": mean([row["reward"] for row in records]),
        "episode_action_p99_mean": mean(
            [row["action_p99"] for row in records]
        ),
        "action_max": max(row["action_max"] for row in records),
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    stair_files = sorted(glob.glob(os.path.join(STAIR_DIR, "*.json")))
    speed_files = sorted(glob.glob(os.path.join(SPEED_DIR, "*.json")))
    if len(stair_files) != 24:
        raise RuntimeError(f"Expected 24 stair files, got {len(stair_files)}")
    if len(speed_files) != 36:
        raise RuntimeError(f"Expected 36 speed files, got {len(speed_files)}")

    stair_raw = [json.load(open(path)) for path in stair_files]
    speed_raw = [json.load(open(path)) for path in speed_files]
    for data in stair_raw:
        if data["ppo_checkpoint_sha256"] != EXPECTED_SHA:
            raise RuntimeError("Unexpected stair checkpoint SHA")
        if data["training_updates"]:
            raise RuntimeError("Training update detected in stair baseline")
    for data in speed_raw:
        if data["checkpoint_sha256"] != EXPECTED_SHA:
            raise RuntimeError("Unexpected speed checkpoint SHA")
        if data["training_updates"]:
            raise RuntimeError("Training update detected in speed baseline")

    stair_rows = []
    for height in sorted({data["step_height_m"] for data in stair_raw}):
        height_data = [
            data for data in stair_raw
            if data["step_height_m"] == height
        ]
        if len({data["terrain_hash"] for data in height_data}) != 1:
            raise RuntimeError(f"Terrain hash mismatch at height {height}")
        pooled = []
        for data in sorted(height_data, key=lambda item: item["seed"]):
            summary = stair_summary(data["records"])
            stair_rows.append({
                "height_m": height,
                "seed": data["seed"],
                **summary,
            })
            pooled.extend(data["records"])
        stair_rows.append({
            "height_m": height,
            "seed": "pooled",
            **stair_summary(pooled),
        })

    speed_rows = []
    for terrain in ("flat", "rough"):
        for speed in sorted({
            data["command_x"] for data in speed_raw
            if data["terrain"] == terrain
        }):
            condition = [
                data for data in speed_raw
                if data["terrain"] == terrain
                and data["command_x"] == speed
            ]
            pooled = []
            for data in sorted(condition, key=lambda item: item["seed"]):
                summary = speed_summary(data["records"])
                speed_rows.append({
                    "terrain": terrain,
                    "command_x": speed,
                    "seed": data["seed"],
                    **summary,
                })
                pooled.extend(data["records"])
            speed_rows.append({
                "terrain": terrain,
                "command_x": speed,
                "seed": "pooled",
                **speed_summary(pooled),
            })

    paired_state_paths = {}
    for terrain in ("flat", "rough"):
        for seed in (20260723, 20260724, 20260725):
            condition = [
                data for data in speed_raw
                if data["terrain"] == terrain and data["seed"] == seed
            ]
            if len({
                data["physical_initial_state_hash"] for data in condition
            }) != 1:
                raise RuntimeError(
                    f"Physical initial-state hash mismatch: {terrain}/{seed}"
                )
            if len({data["terrain_hash"] for data in condition}) != 1:
                raise RuntimeError(
                    f"Terrain hash mismatch: {terrain}/{seed}"
                )
            state_paths = {
                data["paired_initial_state_path"] for data in condition
            }
            if len(state_paths) != 1:
                raise RuntimeError(
                    f"Paired-state path mismatch: {terrain}/{seed}"
                )
            paired_state_paths[f"{terrain}/{seed}"] = state_paths.pop()

    stair_csv = os.path.join(ROOT, "go1_model550_stairs_summary.csv")
    speed_csv = os.path.join(ROOT, "go1_model550_speed_summary.csv")
    summary_json = os.path.join(ROOT, "go1_model550_baseline_summary.json")
    write_csv(stair_csv, stair_rows)
    write_csv(speed_csv, speed_rows)
    with open(summary_json, "w") as stream:
        json.dump({
            "checkpoint": (
                "/home/gjt/legged_gym/logs/rough_go1/6/model_550.pt"
            ),
            "checkpoint_sha256": EXPECTED_SHA,
            "stair_raw_dir": STAIR_DIR,
            "speed_raw_dir": SPEED_DIR,
            "stair_files": len(stair_files),
            "speed_files": len(speed_files),
            "stair_rows": stair_rows,
            "speed_rows": speed_rows,
            "paired_initial_state_paths": paired_state_paths,
            "hash_validation_passed": True,
        }, stream, indent=2, sort_keys=True)
    print(json.dumps({
        "stair_csv": stair_csv,
        "speed_csv": speed_csv,
        "summary_json": summary_json,
        "hash_validation_passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
