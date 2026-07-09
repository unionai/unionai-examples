# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.0",
# ]
# main = "use_checkpoint"
# params = "n_iterations=10"
# ///

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(name="checkpoint_generic_sync")

RETRIES = 3


@env.task(retries=RETRIES)
def use_checkpoint(n_iterations: int = 10) -> int:
    checkpoint = flyte.ctx().checkpoint

    # Load the previous attempt's checkpoint, if any.
    # On the first attempt there is none, so load_sync() returns None.
    path = checkpoint.load_sync()
    start = int(path.read_bytes()) if path else 0

    failure_interval = n_iterations // RETRIES
    index = start
    for index in range(start, n_iterations):
        if index > start and index % failure_interval == 0:
            # Simulate a failure so the next attempt resumes from the checkpoint
            raise RuntimeError(f"Simulated failure at iteration {index}")
        # Persist progress to object storage.
        checkpoint.save_sync(f"{index + 1}".encode())
    return index
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(use_checkpoint, n_iterations=10)
    print(run.url)
