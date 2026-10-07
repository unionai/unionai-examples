# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.10",
#    "flyteplugins-nsight @ git+https://github.com/flyteorg/flyte-sdk@6d3d72b8198d0444ad0471836ca118d32344b268#subdirectory=plugins/nsight",
# ]
# main = "train_regions"
# params = ""
# ///

import flyte
from flyteplugins.nsight import nsys, nsys_profile, nvtx

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

env = flyte.TaskEnvironment(
    name="nsight_regions",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1"),
)


def build_model():
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 4096)).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, opt


def train_step(model, opt):
    import torch

    loss = model(torch.randn(512, 4096, device="cuda")).pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss


def evaluate(model):
    import torch

    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            with nvtx.range("eval_step"):
                model(torch.randn(512, 4096, device="cuda"))


# {{docs-fragment async}}
@nsys_profile(capture="manual", trace=["cuda", "nvtx"])
@env.task
async def train_regions(steps: int = 30) -> float:
    import torch

    model, opt = build_model()
    for _ in range(3):  # warmup, not profiled
        train_step(model, opt)
    torch.cuda.synchronize()

    async with nsys.range("training"):
        for i in range(steps):
            with nvtx.range(f"step_{i}"):
                loss = train_step(model, opt)
        torch.cuda.synchronize()

    async with nsys.range("evaluation"):
        evaluate(model)
        torch.cuda.synchronize()

    return loss.item()
# {{/docs-fragment async}}


# {{docs-fragment sync}}
@nsys_profile(capture="manual", trace=["cuda", "nvtx"])
@env.task
def train_regions_sync(steps: int = 30) -> float:
    import torch

    model, opt = build_model()
    for _ in range(3):
        train_step(model, opt)
    torch.cuda.synchronize()

    with nsys.range("training"):
        for i in range(steps):
            with nvtx.range(f"step_{i}"):
                loss = train_step(model, opt)
        torch.cuda.synchronize()

    return loss.item()
# {{/docs-fragment sync}}


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(train_regions).url)
    print(flyte.run(train_regions_sync).url)
