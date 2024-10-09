# # Serve with vLLM on Union Actors for Named Entity Recognition
#
# This tutorial demonstrates how to deploy an automated, low-latency, named entity recognition workflow. Given some
# unstructured text, this workflow will locate and classify named entities using an LLM. In order to minimize
# latency, we will use Union actors which allow for container and environment reuse between tasks that need to
# maintain state. To further maximize inference efficiency, we will serve a Hugging Face model on our actor using
# vLLM. vLLM achieves state-of-the-art throughput with dynamic batching and caching mechanisms that maximize GPU
# utilization and reduces redundant computations.

# ## Creating Secrets to Pull Hugging Face Models
#
# In this example we will use the `google/gemma-7b-it` as it was fine-tuned on instruction-following datasets,
# allowing it to handle conversational contexts. For our workflow to download the `google/gemma-7b-it` model,
# we need to sign in to Hugging Face, accept the license for [google/gemma-7b-it](
# https://huggingface.co/google/gemma-7b-it), and then generate an [access token](
# https://huggingface.co/settings/tokens).
#
# Then, we securely store our access token using the `union` CLI tool:
# ```bash
# union create secret HF_TOKEN
# ```
# and paste the access token when prompted.
#
# With our Hugging Face credentials set up and saved on Union, we can import our dependencies and set some constants
# that we will use in our vLLM server:

import json
import random
import re
from dataclasses import dataclass
from typing import Annotated, Tuple

import requests
import vllm
from flytekit import ImageSpec, task, workflow, Secret, PodTemplate, Artifact, kwtypes
from flytekitplugins.awssagemaker_inference import BotoConfig, BotoTask
from kubernetes.client import V1Toleration
from kubernetes.client.models import (
    V1Container,
    V1ContainerPort,
    V1HTTPGetAction,
    V1PodSpec,
    V1Probe,
    V1ResourceRequirements,
)
from union.actor import ActorEnvironment

# Define the host and port
HOST = "0.0.0.0"
PORT = 8000
VLLM_HOST = f"http://{HOST}:{PORT}"
URL = f"{VLLM_HOST}/v1/chat/completions"


@dataclass
class Text:
    id: str
    body: str


TextSample = Artifact(name="text_sample")


def extract_entities_from_response(response_text):
    # Define regex patterns to capture persons, organizations, locations, and dates
    persons_pattern = r'"persons":\s*\[([^\]]*)\]'
    organizations_pattern = r'"organizations":\s*\[([^\]]*)\]'
    locations_pattern = r'"locations":\s*\[([^\]]*)\]'
    dates_pattern = r'"dates":\s*\[([^\]]*)\]'

    # Function to clean and split the extracted entity lists
    def clean_list(match):
        # Remove extra quotes and split by commas
        return [item.strip().strip('"') for item in match.split(",") if item.strip()]

    # Extract the entities using regex
    persons_match = re.search(persons_pattern, response_text)
    organizations_match = re.search(organizations_pattern, response_text)
    locations_match = re.search(locations_pattern, response_text)
    dates_match = re.search(dates_pattern, response_text)

    # Prepare the extracted dictionary
    entities_dict = {
        "persons": clean_list(persons_match.group(1)) if persons_match else [],
        "organizations": (
            clean_list(organizations_match.group(1)) if organizations_match else []
        ),
        "locations": clean_list(locations_match.group(1)) if locations_match else [],
        "dates": clean_list(dates_match.group(1)) if dates_match else [],
    }

    return entities_dict


def ner_request(text_body: str) -> str:
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "google/gemma-7b-it",
        "messages": [
            {
                "role": "user",
                "content": f"""
                    Extract all named entities such as persons, organizations, locations, and dates from the following text: {text_body}.
                    Return the results in this exact JSON format:
                    {{
                        "persons": ["..."],
                        "organizations": ["..."],
                        "locations": ["..."],
                        "dates": ["..."]
                    }}""",
            }
        ],
        "max_tokens": 150,
        "temperature": 0,
    }

    # Make the request
    response = requests.post(URL, headers=headers, data=json.dumps(data))
    entities_dict = extract_entities_from_response(
        response.json()["choices"][0]["message"]["content"]
    )
    return json.dumps(entities_dict)


image = ImageSpec(
    name="test-image",
    registry="ghcr.io/dansola",
    packages=[
        "vllm==0.6.2",
        "fastapi==0.115.0",
        "union==0.1.82",
        "kubernetes==31.0.0",
        "flytekitplugins-awssagemaker==1.13.8",
    ],
    apt_packages=["build-essential"],
)

pod_template = PodTemplate(
    pod_spec=V1PodSpec(
        containers=[],
        init_containers=[
            V1Container(
                name="vllm",
                image="ghcr.io/dansola/test-image:spzyICY_wI5b6DbFuoo4qA",
                ports=[V1ContainerPort(container_port=PORT)],
                resources=V1ResourceRequirements(
                    requests={
                        "cpu": "2",
                        "memory": "10Gi",
                        "nvidia.com/gpu": "1",
                    },
                    limits={
                        "cpu": "2",
                        "memory": "10Gi",
                        "nvidia.com/gpu": "1",
                    },
                ),
                command=[
                    "sh",
                    "-c",
                    "export HF_TOKEN=$_UNION_HF_TOKEN && python -m vllm.entrypoints.openai.api_server --model=google/gemma-7b-it --dtype=half --max-model-len=2000",
                ],
                restart_policy="Always",
                startup_probe=V1Probe(
                    http_get=V1HTTPGetAction(port=PORT, path="/health"),
                    failure_threshold=100,
                ),
            ),
        ],
        node_selector={"k8s.amazonaws.com/accelerator": "nvidia-tesla-l4"},
        tolerations=[
            V1Toleration(
                effect="NoSchedule",
                key="k8s.amazonaws.com/accelerator",
                operator="Equal",
                value="nvidia-l4",
            ),
            V1Toleration(
                effect="NoSchedule",
                key="nvidia.com/gpu",
                operator="Equal",
                value="present",
            ),
        ],
    ),
)

actor_env = ActorEnvironment(
    name="vllm-v8",
    replica_count=1,
    pod_template=pod_template,
    container_image=image,
    ttl_seconds=300,
    secret_requests=[Secret(key="HF_TOKEN", mount_requirement=Secret.MountType.ENV_VAR)],
)


@actor_env.task
def vllm(text: Text) -> Tuple[str, str]:
    return f"daniel_testing/text_directory/{text.id}/named_entities.json", ner_request(
        text_body=text.body
    )


put_object_config = BotoConfig(
    service="s3",
    method="put_object",
    config={
        "Bucket": "{inputs.bucket}",
        "Key": "{inputs.key}",
        "Body": "{inputs.body}",
    },
    images={"primary_container_image": image},
    region="us-east-2",
)
put_object_task = BotoTask(
    name="put_object",
    task_config=put_object_config,
    inputs=kwtypes(bucket=str, key=str, body=str),
)


@workflow
def ner_wf(text: Text = TextSample.query()):
    file_name, named_entities = vllm(text=text)
    put_object_task(bucket="union-oc-production-demo", key=file_name, body=named_entities)


@task
def get_text() -> Text:
    example_texts = {
        "1": "Elon Musk, the CEO of Tesla, announced a partnership with SpaceX to launch satellites from Cape Canaveral in 2024.",
        "2": "On September 15th, 2023, Serena Williams won the U.S. Open at Arthur Ashe Stadium in New York City.",
        "3": "President Joe Biden met with leaders from NATO in Brussels to discuss the conflict in Ukraine on July 10th, 2022.",
        "4": "Sundar Pichai, the CEO of Google, gave the keynote speech at the Google I/O conference held in Mountain View on May 11th, 2023.",
        "5": "J.K. Rowling, author of the Harry Potter series, gave a talk at Oxford University in December 2019.",
    }
    id = random.choice(list(example_texts.keys()))
    return Text(id=id, body=example_texts[id])


@workflow
def upstream_wf() -> Annotated[Text, TextSample]:
    return get_text()
