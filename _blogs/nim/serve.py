from flytekit import ImageSpec, Resources, Secret, task
from flytekit.extras.accelerators import A10G
from flytekitplugins.inference import NIM, NIMSecrets
from openai import OpenAI

from constants import BUILDER, HF_KEY, HF_REPO_ID, NGC_IMAGE_SECRET, NGC_KEY, REGISTRY

image = ImageSpec(
    name="nim_serve",
    registry=REGISTRY,
    apt_packages=["git"],
    packages=["flytekitplugins-inference>=1.13.1a5"],
    builder=BUILDER,
)

nim_instance = NIM(
    image="nvcr.io/nim/meta/llama3-8b-instruct:1.0.0",
    secrets=NIMSecrets(
        ngc_image_secret=NGC_IMAGE_SECRET,
        ngc_secret_key=NGC_KEY,
        secrets_prefix="_UNION_",
        hf_token_key=HF_KEY,
    ),
    hf_repo_ids=[HF_REPO_ID],
    lora_adapter_mem="500Mi",
    env={"NIM_PEFT_SOURCE": "/home/nvs/loras"},
)


@task(
    container_image=image,
    pod_template=nim_instance.pod_template,
    secret_requests=[
        Secret(key=HF_KEY, mount_requirement=Secret.MountType.ENV_VAR),
        Secret(
            key=NGC_KEY, mount_requirement=Secret.MountType.ENV_VAR
        ),  # must be mounted as env vars
    ],
    accelerator=A10G,
    requests=Resources(gpu="0"),
)
def model_serving(questions: list[str], repo_id: str) -> list[str]:
    responses = []
    client = OpenAI(
        base_url=f"{nim_instance.base_url}/v1", api_key="nim"
    )  # api key required but ignored

    for question in questions:
        completion = client.chat.completions.create(
            model=repo_id.split("/")[1],
            messages=[
                {"role": "system", "content": "You are a knowledgeable AI assistant."},
                {"role": "user", "content": question},
            ],
            max_tokens=256,
        )
        responses.append(completion.choices[0].message.content)

    return responses
