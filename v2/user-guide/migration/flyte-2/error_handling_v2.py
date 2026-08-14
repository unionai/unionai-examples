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
import flyte.errors

env = flyte.TaskEnvironment(name="error_handling", resources=flyte.Resources(cpu=1, memory="250Mi"))


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


# Infrastructure failures are catchable too: recover from an out-of-memory
# kill by retrying the same task with a larger memory request.
# {{docs-fragment oom}}
@env.task
async def oomer(x: int) -> float:
    large_list = [x] * 100000000
    return sum(large_list) / len(large_list)


@env.task
async def main_with_oom_retry(x: int) -> float:
    try:
        return await oomer(x)
    except flyte.errors.OOMError:
        # Retry the same task with a larger memory request.
        return await oomer.override(
            resources=flyte.Resources(memory="16Gi")
        )(x)
# {{/docs-fragment oom}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, max_depth=0)
    print(r.name)
    print(r.url)
    r.wait()
