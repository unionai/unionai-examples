# {{docs-fragment first-example}}
import flyte

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
# {{/docs-fragment first-example}}

# {{docs-fragment policy-params}}
flyte.ReusePolicy(
    replicas: typing.Union[int, typing.Tuple[int, int]],
    concurrency: int,
    scaledown_ttl: typing.Union[int, datetime.timedelta],
    idle_ttl: typing.Union[int, datetime.timedelta]
)
# {{/docs-fragment policy-params}}

# {{docs-fragment replicas}}
# Fixed pool size
reuse_policy = flyte.ReusePolicy(
    replicas=3,
    concurrency=1,
    scaledown_ttl=timedelta(minutes=10),
    idle_ttl=timedelta(hours=1)
)

# Auto-scaling pool
reuse_policy = flyte.ReusePolicy(
    replicas=(1, 10),
    concurrency=1,
    scaledown_ttl=timedelta(minutes=10),
    idle_ttl=timedelta(hours=1)
)
# {{/docs-fragment replicas}}

# {{docs-fragment concurrency}}
# Sequential processing (default)
sequential_policy = flyte.ReusePolicy(
    replicas=2,
    concurrency=1,  # One task per container
    scaledown_ttl=timedelta(minutes=10),
    idle_ttl=timedelta(hours=1)
)

# Concurrent processing
concurrent_policy = flyte.ReusePolicy(
    replicas=2,
    concurrency=5,  # 5 tasks per container = 10 total concurrent tasks
    scaledown_ttl=timedelta(minutes=10),
    idle_ttl=timedelta(hours=1)
)
# {{/docs-fragment concurrency}}

# {{docs-fragment ttl}}
from datetime import timedelta

lifecycle_policy = flyte.ReusePolicy(
    replicas=3,
    concurrency=2,
    scaledown_ttl=timedelta(minutes=10),  # Individual containers shut down after 10 minutes of inactivity
    idle_ttl=timedelta(hours=1)         # Entire environment shuts down after 1 hour of no tasks
)
# {{/docs-fragment ttl}}

# {{docs-fragment param-relationship}}
reuse_policy = flyte.ReusePolicy(
    replicas=4,                           # Infrastructure: How many containers?
    concurrency=3,                        # Throughput: How many tasks per container?
    scaledown_ttl=timedelta(minutes=10),  # Individual: When do idle containers shut down?
    idle_ttl=timedelta(hours=1)           # Environment: When does the whole pool shut down?
)
# Total capacity: 4 × 3 = 12 concurrent tasks
# Individual containers shut down after 10 minutes of inactivity
# Entire environment shuts down after 1 hour of no tasks
# {{/docs-fragment param-relationship}}
