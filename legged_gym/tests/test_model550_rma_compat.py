"""Fail-fast compatibility test for model_550 and the RMA wrapper."""

import hashlib
import json
import os

import torch

from legged_gym.algorithms.rma_actor_critic import RMAActorCritic
from rsl_rl.modules import ActorCritic


CHECKPOINT = os.path.realpath(
    "/home/gjt/legged_gym/logs/rough_go1/6/model_550.pt"
)
OUTPUT = (
    "/home/gjt/legged_gym/logs/baselines/"
    "model550_rma_compatibility.json"
)


def main():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state = checkpoint["model_state_dict"]
    ppo = ActorCritic(
        235, 235, 12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    ppo.load_state_dict(state, strict=True)
    ppo.eval()

    rma = RMAActorCritic(
        235, 235, 12,
        history_len=10,
        latent_dim=32,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    compatible = {
        key: value for key, value in state.items()
        if key in rma.state_dict()
        and rma.state_dict()[key].shape == value.shape
    }
    result = rma.load_state_dict(compatible, strict=False)
    loaded_actor = [
        key for key in state if key.startswith("actor.")
        and key in compatible
    ]
    loaded_critic = [
        key for key in state if key.startswith("critic.")
        and key in compatible
    ]
    std_loaded = "std" in compatible

    film_zero_max = max(
        rma.actor_film.weight.abs().max().item(),
        rma.actor_film.bias.abs().max().item(),
        rma.critic_film.weight.abs().max().item(),
        rma.critic_film.bias.abs().max().item(),
    )
    for name, parameter in rma.named_parameters():
        if (
            name.startswith("history_encoder.")
            or name.startswith("adaptation_module.")
            or name.startswith("latent_norm.")
            or name.startswith("actor_film.")
            or name.startswith("critic_film.")
            or name == "std"
        ):
            parameter.requires_grad_(False)
    rma.set_rma_alpha(0.0)
    rma.eval()

    generator = torch.Generator().manual_seed(20260723)
    obs = torch.randn(256, 235, generator=generator)
    history = torch.randn(256, 10, 235, generator=generator)
    with torch.inference_mode():
        ppo_action = ppo.act_inference(obs)
        rma_action = rma.act_inference(obs, obs_history=history)
        ppo_value = ppo.evaluate(obs)
        rma_value = rma.evaluate(obs, obs_history=history)
    action_diff = (ppo_action - rma_action).abs().max().item()
    value_diff = (ppo_value - rma_value).abs().max().item()
    actor_diff = max(
        (
            ppo.state_dict()[key] - rma.state_dict()[key]
        ).abs().max().item()
        for key in state if key.startswith("actor.")
    )
    critic_diff = max(
        (
            ppo.state_dict()[key] - rma.state_dict()[key]
        ).abs().max().item()
        for key in state if key.startswith("critic.")
    )
    rma_trainable = [
        name for name, parameter in rma.named_parameters()
        if (
            name.startswith("history_encoder.")
            or name.startswith("adaptation_module.")
            or name.startswith("latent_norm.")
            or name.startswith("actor_film.")
            or name.startswith("critic_film.")
        )
        and parameter.requires_grad
    ]

    with open(CHECKPOINT, "rb") as stream:
        checkpoint_sha = hashlib.sha256(stream.read()).hexdigest()
    report = {
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": checkpoint_sha,
        "actor_loaded": f"{len(loaded_actor)}/8",
        "critic_loaded": f"{len(loaded_critic)}/8",
        "std_loaded": std_loaded,
        "std_min": state["std"].min().item(),
        "std_max": state["std"].max().item(),
        "new_rma_missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
        "film_zero_init_max_abs": film_zero_max,
        "actor_param_max_abs_diff": actor_diff,
        "critic_param_max_abs_diff": critic_diff,
        "alpha_zero_action_max_abs_diff": action_diff,
        "alpha_zero_value_max_abs_diff": value_diff,
        "rma_trainable_parameters": rma_trainable,
        "rma_grad_norm_expected": 0.0,
        "passed": (
            len(loaded_actor) == 8
            and len(loaded_critic) == 8
            and std_loaded
            and film_zero_max == 0.0
            and actor_diff == 0.0
            and critic_diff == 0.0
            and action_diff < 1e-7
            and value_diff < 1e-7
            and not rma_trainable
        ),
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError("model_550 RMA compatibility test failed")


if __name__ == "__main__":
    main()
