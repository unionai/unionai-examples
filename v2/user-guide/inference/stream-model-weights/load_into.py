# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "torch",
# ]
# main = "predict"
# params = "model=s3://your-bucket/models/classifier features='[0.1, 0.2, 0.3, 0.4]'"
# ///

"""Stream a checkpoint's weights into a PyTorch module inside a task."""

# {{docs-fragment load-into}}
import flyte
from flyte.io import Dir

env = flyte.TaskEnvironment(
    name="streamed-classifier",
    image=flyte.Image.from_debian_base().with_pip_packages("torch"),
    resources=flyte.Resources(cpu=4, memory="8Gi", gpu="L4:1"),
)


@env.task
async def predict(model: Dir, features: list[float]) -> list[float]:
    import torch
    from flyte.extras.model_streamer import ModelStreamer, empty_weights

    # The parameters stay on the meta device until their tensors arrive.
    with empty_weights():
        classifier = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))

    await ModelStreamer(model.path).load_into(classifier, device="cuda")

    with torch.inference_mode():
        return classifier(torch.tensor([features], device="cuda")).softmax(dim=-1)[0].tolist()


# {{/docs-fragment load-into}}

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        predict,
        model=Dir.from_existing_remote("s3://your-bucket/models/classifier"),
        features=[0.1, 0.2, 0.3, 0.4],
    )
    print(run.url)
