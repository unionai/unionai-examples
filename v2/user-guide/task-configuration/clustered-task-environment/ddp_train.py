# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.18",
#    "torch",
#    "numpy",
# ]
# main = "train_ddp"
# params = "steps=50"
# ///

# {{docs-fragment imports}}
from __future__ import annotations

import flyte
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

# {{/docs-fragment imports}}

# --- Knobs -----------------------------------------------------------------
# Flip to True to run on GPUs (NCCL). The default runs on CPU (gloo) so the
# example can be smoke-tested without a GPU cluster.
USE_GPU = False
REPLICAS = 2  # pods (== nodes)
NPROC_PER_NODE = 1  # processes per pod => world_size = REPLICAS * NPROC_PER_NODE

_BACKEND = "nccl" if USE_GPU else "gloo"

# {{docs-fragment env}}
# The torch wheel from PyPI bundles CUDA + NCCL, so the same image runs on CPU
# and GPU nodes. `flyte` itself provides the `clustered` runtime entrypoint that
# bootstraps torchrun inside each pod — no extra runtime library is needed.
image = flyte.Image.from_debian_base().with_pip_packages("torch", "numpy")

resources = (
    flyte.Resources(cpu=(2, 4), memory=("4Gi", "8Gi"), gpu="L4:1")
    if USE_GPU
    else flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi"))
)

env = ClusteredTaskEnvironment(
    name="ddp_env",
    image=image,
    resources=resources,
    replicas=REPLICAS,  # number of pods (nodes) in the JobSet
    nproc_per_node=NPROC_PER_NODE,  # processes (one per GPU) per pod
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=1),
)
# {{/docs-fragment env}}


# {{docs-fragment task}}
@env.task
async def train_ddp(steps: int = 50, lr: float = 0.05) -> float:
    """Run DDP training across the cluster and return the final (rank-0) loss."""
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    ctx = flyte.ctx()

    # Bind this rank to its local GPU BEFORE init_process_group so NCCL binds
    # the right device.
    if _BACKEND == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(ctx.local_rank or 0)
        device = torch.device(f"cuda:{ctx.local_rank or 0}")
    else:
        device = torch.device("cpu")

    # torchrun has already populated RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.
    dist.init_process_group(backend=_BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        f"[rank {rank}/{world_size}] device={device} "
        f"node_rank={ctx.node_rank} nnodes={ctx.nnodes} master_addr={ctx.master_addr}",
        flush=True,
    )

    # Tiny model the workers train cooperatively: learn y = x · [1,1,1,1].
    torch.manual_seed(0)
    model = nn.Linear(4, 1).to(device)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Each rank trains on its own shard of synthetic data.
    g = torch.Generator().manual_seed(rank)
    x = torch.randn(64, 4, generator=g).to(device)
    y = x.sum(dim=1, keepdim=True)

    last_loss = 0.0
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        if rank == 0 and step % 10 == 0:
            print(f"[rank 0] step {step:3d}  loss {last_loss:.5f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] done — final loss {last_loss:.5f}", flush=True)
    return last_loss

# {{/docs-fragment task}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_ddp, steps=50)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
# {{/docs-fragment run}}
