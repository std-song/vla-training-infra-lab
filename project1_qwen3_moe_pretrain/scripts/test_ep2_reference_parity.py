from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import nanotron.distributed as dist
from nanotron.nn.moe import Qwen2MoELayer
from nanotron.trainer import DistributedTrainer


def gather_experts(local: torch.Tensor, group) -> torch.Tensor:
    shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, local.detach().contiguous(), group=group)
    return torch.cat(shards, dim=0).requires_grad_()


def dense_reference(hidden, router_weight, gate_up, down, top_k):
    probs = F.softmax(F.linear(hidden.float(), router_weight), dim=-1, dtype=torch.float32)
    routing_weights, routing_indices = torch.topk(probs, k=top_k, dim=-1)
    output = torch.zeros_like(hidden)
    for expert_id in range(gate_up.shape[0]):
        token_ids, k_ids = (routing_indices == expert_id).nonzero(as_tuple=True)
        if token_ids.numel() == 0:
            continue
        merged = hidden.index_select(0, token_ids) @ gate_up[expert_id]
        gate, up = merged.chunk(2, dim=-1)
        expert_output = (F.silu(gate) * up) @ down[expert_id]
        expert_output = expert_output * routing_weights[token_ids, k_ids].to(expert_output.dtype).unsqueeze(-1)
        output.index_add_(0, token_ids, expert_output)
    return output


def errors(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().max().clamp_min(1e-8)
    return delta.max().item(), (delta.max() / scale).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    args = parser.parse_args()

    trainer = DistributedTrainer(args.config_file)
    ep_pg = trainer.parallel_context.ep_pg
    ep_rank = dist.get_rank(ep_pg)
    layer = next(module for module in trainer.unwrapped_model.modules() if isinstance(module, Qwen2MoELayer))

    torch.manual_seed(1234)
    hidden_ep = torch.randn(32, layer.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    hidden_ref = hidden_ep.detach().clone().requires_grad_()
    router_ref = layer.router.weight.detach().clone().requires_grad_()
    gate_up_ref = gather_experts(layer.experts.merged_gate_up_proj, ep_pg)
    down_ref = gather_experts(layer.experts.merged_down_proj, ep_pg)

    weights, indices, _ = layer.router(hidden_ep)
    output_ep = layer._core_forward_ep(hidden_ep, weights, indices)
    output_ref = dense_reference(
        hidden_ref,
        router_ref,
        gate_up_ref,
        down_ref,
        layer.num_experts_per_token,
    )

    probe = torch.randn_like(output_ep)
    grads_ep = torch.autograd.grad(
        (output_ep.float() * probe.float()).sum(),
        [
            hidden_ep,
            layer.router.weight,
            layer.experts.merged_gate_up_proj,
            layer.experts.merged_down_proj,
        ],
    )
    grads_ref = torch.autograd.grad(
        (output_ref.float() * probe.float()).sum(),
        [hidden_ref, router_ref, gate_up_ref, down_ref],
    )

    start = ep_rank * layer.num_local_experts
    stop = start + layer.num_local_experts
    comparisons = {
        "output": errors(output_ep, output_ref),
        "hidden_grad": errors(grads_ep[0], grads_ref[0]),
        "router_grad": errors(grads_ep[1], grads_ref[1]),
        "gate_up_grad": errors(grads_ep[2], grads_ref[2][start:stop]),
        "down_grad": errors(grads_ep[3], grads_ref[3][start:stop]),
    }
    for name, (max_abs, max_rel) in comparisons.items():
        print(f"EP_PARITY rank={ep_rank} name={name} max_abs={max_abs:.9g} max_rel={max_rel:.9g}", flush=True)
        assert max_rel < 0.04 or max_abs < 5e-3, (name, max_abs, max_rel)

    if ep_rank == 0:
        print("EP2_REFERENCE_PARITY_OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
