from flytekit import task, workflow, ImageSpec, Resources
import flytekit
import torch
import torch.nn as nn

image = ImageSpec(
    name="dl-image",
    packages=["torch"],
)


@task(
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
    enable_deck=True,
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
        flytekit.current_context().default_deck.append(f"loss: {float(loss.item())}")
        flytekit.Deck.publish()

    return float(loss.item())


@workflow
def main(epochs: int) -> float:
    return train(epochs=epochs)
