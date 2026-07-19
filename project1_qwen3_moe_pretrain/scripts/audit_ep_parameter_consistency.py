from __future__ import annotations

import argparse

import torch

import nanotron.distributed as dist
from nanotron.parallel.parameters import NanotronParameter
from nanotron.trainer import DistributedTrainer
from run_train import get_dataloader


def unwrap(model):
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def audit_replicated_parameters(trainer: DistributedTrainer, label: str) -> None:
    model = unwrap(trainer.model)
    ep_pg = trainer.parallel_context.ep_pg
    ep_size = dist.get_world_size(ep_pg)
    rows = []

    for name, param in model.named_parameters():
        expert_sharded = False
        if isinstance(param, NanotronParameter) and param.is_sharded:
            expert_sharded = param.get_sharded_info().is_expert_sharded(trainer.parallel_context)
        if expert_sharded or ".experts." in name:
            continue

        value = param.detach().float().contiguous()
        gathered = [torch.empty_like(value) for _ in range(ep_size)]
        dist.all_gather(gathered, value, group=ep_pg)
        max_diff = max((other - gathered[0]).abs().max().item() for other in gathered[1:])
        rows.append((max_diff, name))

    rows.sort(reverse=True)
    if dist.get_rank(ep_pg) == 0:
        print(
            f"EP_AUDIT label={label} params={len(rows)} "
            f"nonzero={sum(diff > 0 for diff, _ in rows)} "
            f"max_abs_diff={(rows[0][0] if rows else 0):.9g}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    args = parser.parse_args()

    trainer = DistributedTrainer(args.config_file)
    audit_replicated_parameters(trainer, "before_train")
    trainer.train(get_dataloader(trainer))
    audit_replicated_parameters(trainer, "after_train")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
