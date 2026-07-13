"""Render matched stale-state and learned-state Pi0.5 LIBERO training configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def make_config(
    train_episodes: list[int],
    mode: str,
    project_root: Path,
    steps: int,
    save_checkpoint: bool,
) -> dict:
    name = f"pi05_libero_{mode}_{steps}"
    config = {
        "policy": {
            "type": "pi05",
            "pretrained_path": str(project_root / "models/pi05_base"),
            "push_to_hub": False,
            "dtype": "bfloat16",
            "device": "cuda",
            "state_cond": True,
            "empty_cameras": 1,
        },
        "dataset": {
            "repo_id": "lerobot/libero",
            "root": str(project_root / "datasets/libero"),
            "episodes": train_episodes,
            "video_backend": "torchcodec",
        },
        "output_dir": str(project_root / f"outputs/{name}"),
        "job_name": name,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "steps": steps,
        "num_workers": 1,
        "seed": 1000,
        "use_policy_training_preset": False,
        "optimizer": {
            "type": "adamw", "lr": 5.0e-5, "betas": [0.9, 0.95], "weight_decay": 1.0e-10,
        },
        "scheduler": {
            "type": "cosine_decay_with_warmup",
            "num_warmup_steps": max(1, steps // 10),
            "peak_lr": 5.0e-5,
            "decay_lr": 2.5e-6,
            "num_decay_steps": steps,
        },
        "save_checkpoint": save_checkpoint,
        "save_freq": steps,
        "log_freq": 20 if steps >= 20 else 1,
        "wandb": {"enable": False},
        "max_delay_steps": 2,
        "shared_observation": True,
        "future_state_mode": mode,
        "lora": {
            "enable": True,
            "backend": "peft",
            "extra_trainable_modules": [
                "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
                "state_proj", "state_mlp_in", "state_mlp_out", "embeddings",
                "input_layernorm", "post_attention_layernorm",
            ],
            "r": 16,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                "down_proj", "out_proj", "fc1", "fc2",
            ],
        },
    }
    if mode == "learned":
        config["future_state_predictor_path"] = str(
            project_root / "vlash_reproduction/ablation/libero_state_predictor/best.pt"
        )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    train_episodes = [int(value) for value in split["train_episode_ids"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("stale", "learned"):
        config = make_config(train_episodes, mode, args.project_root, args.steps, args.save_checkpoint)
        output = args.output_dir / f"pi05_libero_{mode}_{args.steps}.yaml"
        output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
