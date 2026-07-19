import torch

import nanotron.distributed as dist
from nanotron.nn.moe import _AllToAllSingle, _ReduceFromExpertParallel, _ShardReplicatedTokens


def main():
    dist.init_process_group(backend="nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    assert dist.get_world_size(group) == 2
    torch.cuda.set_device(rank)

    replicated = torch.arange(8, device="cuda", dtype=torch.float32).view(4, 2).requires_grad_()
    token_ids = torch.arange(rank, 4, 2, device="cuda")
    shard = _ShardReplicatedTokens.apply(replicated, token_ids, group)
    shard.sum().backward()
    torch.testing.assert_close(replicated.grad, torch.ones_like(replicated))

    payload = (torch.arange(4, device="cuda", dtype=torch.float32) + 10 * rank).requires_grad_()
    exchanged = _AllToAllSingle.apply(payload, (2, 2), (2, 2), group)
    exchanged.sum().backward()
    torch.testing.assert_close(payload.grad, torch.ones_like(payload))

    owner_output = torch.full((2, 2), rank + 1.0, device="cuda", requires_grad=True)
    replicated_output = _ReduceFromExpertParallel.apply(owner_output, group)
    torch.testing.assert_close(replicated_output, torch.full_like(replicated_output, 3.0))
    replicated_output.sum().backward()
    torch.testing.assert_close(owner_output.grad, torch.ones_like(owner_output))

    if rank == 0:
        print("EP_AUTOGRAD_PRIMITIVES_OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
