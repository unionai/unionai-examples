from flytekit import ImageSpec, Resources, Secret, task
from flytekit.core.inference import NIM, NIMSecrets
from flytekit.extras.accelerators import A10G
from openai import OpenAI

from .constants import BUILDER, HF_REPO_ID, NGC_KEY, REGISTRY, HF_KEY

image = ImageSpec(
    name="nim_serve",
    registry=REGISTRY,
    apt_packages=["git"],
    packages=[
        "git+https://github.com/flyteorg/flytekit.git@56d53f7b042a725767cb112d12c0d6ea22d284b4",
        "kubernetes",
        "openai",
    ],
    builder=BUILDER,
)

nim_instance = NIM(
    image="nvcr.io/nim/meta/llama3-8b-instruct:1.0.0",
    secrets=NIMSecrets(
        ngc_image_secret="nvcrio-cred", ngc_secret_key=NGC_KEY, hf_token_key=HF_KEY
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
