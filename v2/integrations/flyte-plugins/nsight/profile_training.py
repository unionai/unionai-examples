# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.10",
#    "kubernetes",
#    "flyteplugins-nsight @ git+https://github.com/flyteorg/flyte-sdk#subdirectory=plugins/nsight",
# ]
# main = "train"
# params = ""
# ///

# {{docs-fragment image}}
import flyte
from flyteplugins.nsight import nsys_profile, nvtx

image = (
    flyte.Image.from_base("nvcr.io/nvidia/pytorch:26.09-py3")
    # python_version must match the base image's Python (3.12 in NGC 26.xx images).
    .clone(extendable=True, name="nsight", python_version=(3, 12))
    .with_pip_packages(
        "flyte",
        "kubernetes",  # imported by allow_nested_sandboxing() when the task module loads
        "flyteplugins-nsight @ git+https://github.com/flyteorg/flyte-sdk#subdirectory=plugins/nsight",
    )
    # NGC installs torch into the system Python, but Flyte runs tasks in /opt/venv.
    # Let that venv see system site-packages so `import torch` resolves to NGC's build.
    .with_commands(
        ["sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' /opt/venv/pyvenv.cfg"]
    )
)
# {{/docs-fragment image}}

# {{docs-fragment env}}
env = flyte.TaskEnvironment(
    name="nsight_train",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1"),
    # Required for the osrt trace domain. See "Permissions" on the docs page.
    pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
)
# {{/docs-fragment env}}


# {{docs-fragment task}}
@nsys_profile(trace=["cuda", "nvtx", "osrt"])
@env.task
async def train(steps: int = 20) -> float:
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 4096)).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    def step() -> torch.Tensor:
        loss = model(torch.randn(512, 4096, device="cuda")).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    # Unlabeled warmup keeps one-time CUDA startup costs out of the NVTX summary.
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    for i in range(steps):
        with nvtx.range(f"step_{i}"):
            loss = step()

    torch.cuda.synchronize()
    return loss.item()
# {{/docs-fragment task}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train)
    print(run.url)
# {{/docs-fragment run}}
