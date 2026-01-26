# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# main = "main"
# params = "n=500"
# ///

# {{docs-fragment import}}
import asyncio
import logging

import flyte

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# {{/docs-fragment import}}


# {{docs-fragment env}}
env = flyte.TaskEnvironment(
    name="reuse_concurrency",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        idle_ttl=60,
        concurrency=100,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse>=0.1.10"),
)
# {{/docs-fragment env}}

# {{docs-fragment tasks}}
@env.task
async def noop(x: int) -> int:
    logger.debug(f"Task noop: {x}")
    return x


@env.task
async def main(n: int = 50) -> int:
    coros = [noop(i) for i in range(n)]
    results = await asyncio.gather(*coros)
    return sum(results)
# {{/docs-fragment tasks}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, n=500)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment run}}
