import flyte.remote

env = flyte.TaskEnvironment(name="root")

# get remote tasks that were previously deployed
torch_task = flyte.remote.Task.get("torch_env.torch_task", auto_version="latest")
spark_task = flyte.remote.Task.get("spark_env.spark_task", auto_version="latest")

@env.task
def main() -> flyte.File:
    dataset = await spark_task(value)
    model = await torch_task(dataset)
    return model