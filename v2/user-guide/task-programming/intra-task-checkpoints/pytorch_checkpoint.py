# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.0",
#    "torch>=2.0",
# ]
# main = "train_linear"
# params = "epochs=10"
# ///

"""Resume a PyTorch training loop across task retries with `flyte.Checkpoint`."""

# {{docs-fragment env}}
import pathlib

import torch
import torch.nn as nn

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_pytorch",
    image=flyte.Image.from_debian_base().with_pip_packages("torch"),
)

RETRIES = 3
# {{/docs-fragment env}}


# {{docs-fragment task}}
@env.task(retries=RETRIES)
async def train_linear(epochs: int = 10) -> float:
    checkpoint = flyte.ctx().checkpoint

    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # Resume model, optimizer, and epoch from the previous attempt, if any.
    prev = await checkpoint.load()
    if prev:
        blob = torch.load(prev, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        start = int(blob["epoch"]) + 1
    else:
        start = 0

    wpath = pathlib.Path("pytorch_linear") / "training.pt"
    wpath.parent.mkdir(parents=True, exist_ok=True)

    failure_interval = epochs // RETRIES
    for epoch in range(start, epochs):
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch > start and epoch % failure_interval == 0:
            # Simulate a failure so the next attempt resumes from the checkpoint
            raise RuntimeError(f"Simulated failure at epoch {epoch}")

        # Save model, optimizer, and epoch state to object storage.
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch},
            wpath,
        )
        await checkpoint.save(wpath)

    with torch.no_grad():
        return float(model(torch.ones(1, 4)).squeeze().item())
# {{/docs-fragment task}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_linear, epochs=10)
    print(run.url)
