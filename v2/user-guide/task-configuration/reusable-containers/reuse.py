# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = "n=500"
# ///

import flyte
from datetime import timedelta

# {{docs-fragment env}}
# Currently required to enable resuable containers
reusable_image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse>=0.1.3")

env = flyte.TaskEnvironment(
    name="reusable-env",
    resources=flyte.Resources(memory="1Gi", cpu="500m"),
    reusable=flyte.ReusePolicy(
        replicas=2,                           # Create 2 container instances
        concurrency=1,                        # Process 1 task per container at a time
        scaledown_ttl=timedelta(minutes=10),  # Individual containers shut down after 5 minutes of inactivity
        idle_ttl=timedelta(hours=1)           # Entire environment shuts down after 30 minutes of no tasks
    ),
    image=reusable_image  # Use the container image augmented with the unionai-reuse library.
)
# {{/docs-fragment env}}

@env.task
async def compute_task(x: int) -> int:
    return x * x

@env.task
async def main() -> list[int]:
    # These tasks will reuse containers from the pool
    results = []
    for i in range(10):
        result = await compute_task(i)
        results.append(result)
    return results

if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
