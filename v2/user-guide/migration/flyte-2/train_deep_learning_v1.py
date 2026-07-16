from flytekit import task, workflow, ImageSpec, Resources
from flytekit.extras.accelerators import T4
import torch
import torch.nn as nn

image = ImageSpec(
    name="dl-image",
    packages=["torch"],
)


@task(
    container_image=image,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    accelerator=T4,
)
def train(epochs: int) -> float:
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


@workflow
def main(epochs: int) -> float:
    return train(epochs=epochs)
