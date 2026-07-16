# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "torch",
# ]
# main = "main"
# params = "epochs=5"
# ///

# {{docs-fragment all}}
import flyte
import torch
import torch.nn as nn

# GPU type and count go in a single "T4:1"-style string. For multi-node
# distributed training, wrap the training task with the torch elastic plugin.
env = flyte.TaskEnvironment(
    name="train_deep_learning",
    image=flyte.Image.from_debian_base().with_pip_packages("torch"),
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="T4:1"),
)


@env.task
async def train(epochs: int) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Linear(10, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    X = torch.randn(128, 10, device=device)
    y = torch.randn(128, 1, device=device)

    loss = torch.tensor(0.0)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    return float(loss.item())


@env.task
async def main(epochs: int) -> float:
    return await train(epochs)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, epochs=5)
    print(r.name)
    print(r.url)
    r.wait()
