# {{docs-fragment pod-template}}
# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte>=2.0.0b25",
#    "kubernetes"
# ]
# ///

import flyte
from kubernetes.client import (
    V1Container,
    V1EnvVar,
    V1LocalObjectReference,
    V1PodSpec,
)


# Create a custom pod template
pod_template = flyte.PodTemplate(
    primary_container_name="primary",           # Name of the main container
    labels={"lKeyA": "lValA"},                 # Custom pod labels
    annotations={"aKeyA": "aValA"},            # Custom pod annotations
    pod_spec=V1PodSpec(                        # Kubernetes pod specification
        containers=[
            V1Container(
                name="primary",
                env=[V1EnvVar(name="hello", value="world")]  # Environment variables
            )
        ],
        image_pull_secrets=[                   # Access to private registries
            V1LocalObjectReference(name="regcred-test")
        ],
    ),
)

# Use the pod template in a TaskEnvironment
env = flyte.TaskEnvironment(
    name="hello_world",
    pod_template=pod_template,                 # Apply the custom pod template
    image=flyte.Image.from_uv_script(__file__, name="flyte", pre=True),
)

@env.task
async def say_hello(data: str) -> str:
    return f"Hello {data}"

@env.task
async def say_hello_nested(data: str = "default string") -> str:
    return await say_hello(data=data)

if __name__ == "__main__":
    flyte.init_from_config()
    result = flyte.run(say_hello_nested, data="hello world")
    print(result.url)
# {{/docs-fragment pod-template}}
