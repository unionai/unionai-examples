# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.10",
#    "flyteplugins-nsight @ git+https://github.com/flyteorg/flyte-sdk@6d3d72b8198d0444ad0471836ca118d32344b268#subdirectory=plugins/nsight",
# ]
# main = "train_ddp"
# params = ""
# ///

import flyte
from flyte.clustered import ClusteredTaskEnvironment
from flyteplugins.nsight import nsys_profile, nvtx

image = (
    flyte.Image.from_base("nvcr.io/nvidia/pytorch:26.09-py3")
    .clone(extendable=True, name="nsight", python_version=(3, 12))
    .with_pip_packages(
        "flyte",
        "flyteplugins-nsight @ git+https://github.com/flyteorg/flyte-sdk@6d3d72b8198d0444ad0471836ca118d32344b268#subdirectory=plugins/nsight",
    )
    .with_commands(
        ["sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' /opt/venv/pyvenv.cfg"]
    )
)

# {{docs-fragment env}}
env = ClusteredTaskEnvironment(
    name="nsight_ddp",
    image=image,
    resources=flyte.Resources(cpu="8", memory="32Gi", gpu="T4:2"),
    replicas=2,  # two pods
    nproc_per_node=2,  # one process per GPU, four ranks in total
)


@nsys_profile(trace=["cuda", "nvtx"])
@env.task
async def train_ddp(steps: int = 30) -> str:
    import os

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = DDP(nn.Linear(4096, 4096).cuda(), device_ids=[local_rank])
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    for i in range(steps):
        with nvtx.range(f"step_{i}"):
            loss = model(torch.randn(512, 4096, device="cuda")).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()  # DDP all-reduces gradients here
            opt.step()
    torch.cuda.synchronize()

    rank = dist.get_rank()
    dist.destroy_process_group()
    return f"rank {rank} done, loss {loss.item():.4f}"
# {{/docs-fragment env}}


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(train_ddp).url)
