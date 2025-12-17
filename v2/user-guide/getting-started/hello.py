# hello.py

import flyte

# The `hello_env` TaskEnvironment is assgned to the variable `env`.
# It is then used in the `@env.task` decorator to define tasks.
# The environment groups configuration for all tasks defined within it.
env = flyte.TaskEnvironment(name="hello_env")

# We use the `@env.task` decorator to define a task called `fn`.
@env.task
def fn(x: int) -> int: # Type annotations are required
    slope, intercept = 2, 5
    return slope * x + intercept

# We also use the `@env.task` decorator to define another task called `main`.
# This is the is the entrypoint task of the workflow.
# It calls the `fn` task defined above multiple times using `flyte.map`.
@env.task
def main(x_list: list[int] = list(range(10))) -> float:
    y_list = list(flyte.map(fn, x_list)) # flyte.map is like Python map, but runs in parallel.
    y_mean = sum(y_list) / len(y_list)
    return y_mean
