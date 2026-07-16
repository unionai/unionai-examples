# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "max_depth=0"
# ///

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(name="error_handling")


@env.task
async def train_fold(max_depth: int) -> float:
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    return 0.90 + 0.001 * max_depth


# Failure handling is ordinary Python try/except -- no on_failure handler.
@env.task
async def main(max_depth: int) -> float:
    try:
        return await train_fold(max_depth)
    except ValueError as e:
        print(f"invalid hyperparameter ({e}); falling back to a safe default")
        # Recover with a safe default instead of failing the whole run.
        return await train_fold(max_depth=6)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, max_depth=0)
    print(r.name)
    print(r.url)
    r.wait()
