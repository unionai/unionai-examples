from flytekit import task, FlyteRemote


# Define your task
@task def hello_world(name: str) -> str:
    return f"Hello {name}"


# Create a FlyteRemote object
remote = FlyteRemote.from_config()

# Register your task
remote.register_task(
    entity=hello_world,
    project="my_project",
    domain="development",
    name="hello_world_task",
    version="v1"
)