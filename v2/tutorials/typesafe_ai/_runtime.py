"""Shared Flyte environments for the System 1 + System 2 tutorial.

Two environments, differing only in size: `env` runs one unit of work at a time;
`driver_env` holds every result in memory at once and renders the report, so its
footprint grows with the size of the matrix rather than with any single unit.
"""

import flyte

# {{docs-fragment env}}
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyteplugins-typesafe-ai",
    "anthropic",
)

# System 1 reads TYPESAFE_API_KEY; System 2 reads ANTHROPIC_API_KEY. One
# environment carries both, so a single task can interleave the two models.
env = flyte.TaskEnvironment(
    name="typesafe-ai-tutorial",
    image=image,
    secrets=[
        flyte.Secret(key="TYPESAFE_API_KEY", as_env_var="TYPESAFE_API_KEY"),
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
    ],
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)

# The aggregator. It fans out into `env`, and a cross-environment call has to
# declare the dependency or the worker environment is missing from the image
# cache at runtime.
driver_env = flyte.TaskEnvironment(
    name="typesafe-ai-tutorial-driver",
    image=image,
    secrets=env.secrets,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[env],
)
# {{/docs-fragment env}}
