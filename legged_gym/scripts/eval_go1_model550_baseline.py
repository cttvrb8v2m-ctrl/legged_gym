"""Pure-PPO deterministic stair baseline for rough_go1/6/model_550.pt."""

import os

os.environ["GO1_STAIRS_PPO_ONLY"] = "1"
os.environ.pop("GO1_STAIRS_PAIR_CHECKPOINT", None)

from eval_stairs_joint import main


if __name__ == "__main__":
    main()
