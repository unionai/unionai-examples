"""Shared task environment for the `flyteplugins-typesafe-ai` examples.

Every example in this folder imports `env` from here, so the image and the
secret are declared once. Run any of them from this directory:

    flyte run triage.py handle
"""

import flyte

# {{docs-fragment env}}
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyteplugins-typesafe-ai",
)

# The TypeSafe SDK reads the key from TYPESAFE_API_KEY, so mount the secret as
# that env var. `as_env_var` is spelled out on purpose: it is the string you
# will grep for when a task cannot find the key.
env = flyte.TaskEnvironment(
    name="typesafe-ai",
    image=image,
    secrets=[flyte.Secret(key="TYPESAFE_API_KEY", as_env_var="TYPESAFE_API_KEY")],
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)
# {{/docs-fragment env}}
