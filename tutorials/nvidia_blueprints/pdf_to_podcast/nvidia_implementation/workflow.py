import os
import time

import flytekit as fl
import requests
from flytekit.types.file import FlyteFile
from kubernetes.client.models import (
    V1Container,
    V1ContainerPort,
    V1EnvVar,
    V1ExecAction,
    V1HTTPGetAction,
    V1PodSpec,
    V1Probe,
    V1Volume,
    V1VolumeMount,
)
from union.actor import ActorEnvironment

from .utils import generate_podcast, wait_for_completion

REGISTRY = os.getenv("REGISTRY", "ghcr.io/unionai-oss")

BASE_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "redis",
    "httpx",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-instrumentation-fastapi",
    "opentelemetry-instrumentation-requests",
    "opentelemetry-instrumentation-redis",
    "opentelemetry-exporter-otlp-proto-grpc",
    "opentelemetry-instrumentation-httpx",
    "opentelemetry-instrumentation-urllib3",
    "ujson",
]

SHARED_COPY = ["shared/"]

DEFAULT_COMMANDS = ["pip install /root/shared"]

service_specs = [
    {
        "name": "agent",
        "python_version": "3.9",
        "apt_packages": ["build-essential"],
        "packages": BASE_PACKAGES + ["python-dotenv", "jinja2", "minio"],
        "copy": SHARED_COPY
        + [
            "services/AgentService/main.py",
            "services/AgentService/podcast_prompts.py",
            "services/AgentService/monologue_prompts.py",
            "services/AgentService/podcast_flow.py",
            "services/AgentService/monologue_flow.py",
            "models.json",
        ],
        "commands": DEFAULT_COMMANDS,
    },
    {
        "name": "api",
        "python_version": "3.9",
        "apt_packages": ["build-essential", "curl"],
        "packages": BASE_PACKAGES
        + [
            "uvicorn[standard]",
            "python-multipart",
            "requests",
            "websockets",
            "asyncio",
            "minio",
        ],
        "copy": SHARED_COPY + ["services/APIService/main.py"],
        "commands": DEFAULT_COMMANDS,
    },
    {
        "name": "pdf",
        "python_version": "3.11",
        "apt_packages": ["curl"],
        "packages": BASE_PACKAGES + ["python-multipart", "asyncio", "requests"],
        "copy": SHARED_COPY + ["services/PDFService/main.py"],
        "commands": DEFAULT_COMMANDS,
    },
    {
        "name": "tts",
        "python_version": "3.11",
        "apt_packages": [
            "wget",
            "curl",
            "man",
            "git",
            "less",
            "openssl",
            "libssl-dev",
            "unzip",
            "unar",
            "build-essential",
            "aria2",
            "tmux",
            "vim",
            "openssh-server",
            "sox",
            "libsox-fmt-all",
            "libsox-fmt-mp3",
            "libsndfile1-dev",
            "ffmpeg",
        ],
        "packages": BASE_PACKAGES + ["edge-tts", "elevenlabs"],
        "copy": SHARED_COPY + ["services/TTSService/main.py"],
        "commands": DEFAULT_COMMANDS,
    },
    {
        "name": "pdf-api",
        "apt_packages": ["curl", "git", "procps"],
        "copy": [
            "services/PDFService/PDFModelService/main.py",
            "services/PDFService/PDFModelService/tasks.py",
            "services/PDFService/PDFModelService/requirements.api.txt",
        ],
        "python_version": "3.12",
        "packages": ["torch==2.4.0"],
        "commands": [
            "mkdir -p /tmp/pdf_conversions",
            "pip install -r /root/requirements.api.txt",
        ],
    },
    {
        "name": "celery-worker",
        "apt_packages": ["libgl1", "libglib2.0-0", "curl", "wget", "git", "procps"],
        "copy": [
            "services/PDFService/PDFModelService/tasks.py",
            "services/PDFService/PDFModelService/requirements.worker.txt",
            "services/PDFService/PDFModelService/download_models.py",
        ],
        "python_version": "3.12",
        "packages": ["torch==2.4.0"],
        "commands": [
            "mkdir -p /tmp/pdf_conversions",
            "pip install -r /root/requirements.worker.txt",
            "python /root/download_models.py",
        ],
    },
]


urls_to_check = [
    "http://localhost:8003/health",
    "http://localhost:8964/health",
    "http://localhost:8889/health",
]
check_urls_script = " && ".join(
    [
        f"until curl -sSf {url}; do echo 'Waiting for {url}...'; sleep 5; done"
        for url in urls_to_check
    ]
)
check_pdf_api_script = "until curl -sSf http://localhost:8004/health; do echo 'Waiting for http://localhost:8004/health...'; sleep 5; done"


image_specs = [
    fl.ImageSpec(
        name=spec["name"],
        builder="union",
        python_version=spec["python_version"],
        apt_packages=spec.get("apt_packages", []),
        packages=spec["packages"],
        copy=spec["copy"],
        registry=REGISTRY,
    ).with_commands(spec["commands"])
    for spec in service_specs
]

actor_pod_template = fl.PodTemplate(
    pod_spec=V1PodSpec(
        init_containers=[
            V1Container(
                name="redis",
                image="redis:latest",
                ports=[V1ContainerPort(container_port=6379)],
                command=["redis-server", "--appendonly", "no"],
                restart_policy="Always",
                startup_probe=V1Probe(
                    _exec=V1ExecAction(command=["redis-cli", "PING"]),
                    initial_delay_seconds=5,
                    period_seconds=3,
                    failure_threshold=10,
                ),
            ),
            V1Container(
                name="jaeger",
                image="jaegertracing/all-in-one:latest",
                ports=[
                    V1ContainerPort(container_port=16686),  # UI
                    V1ContainerPort(container_port=4317),  # OTLP GRPC
                    V1ContainerPort(container_port=4318),  # OTLP HTTP
                ],
                env=[V1EnvVar(name="COLLECTOR_OTLP_ENABLED", value="true")],
                startup_probe=V1Probe(
                    http_get=V1HTTPGetAction(path="/", port=16686),
                    initial_delay_seconds=5,
                    period_seconds=3,
                    failure_threshold=10,
                ),
                restart_policy="Always",
            ),
            V1Container(
                name="minio",
                image="minio/minio:latest",
                ports=[
                    V1ContainerPort(container_port=9000),
                    V1ContainerPort(container_port=9001),
                ],
                env=[
                    V1EnvVar(name="MINIO_ROOT_USER", value="minioadmin"),
                    V1EnvVar(name="MINIO_ROOT_PASSWORD", value="minioadmin"),
                ],
                command=["minio", "server", "/data", "--console-address", ":9001"],
                volume_mounts=[V1VolumeMount(name="minio-data", mount_path="/data")],
                startup_probe=V1Probe(
                    http_get=V1HTTPGetAction(path="/minio/health/live", port=9000),
                    initial_delay_seconds=10,
                    period_seconds=5,
                    failure_threshold=12,
                ),
                restart_policy="Always",
            ),
        ],
        containers=[
            V1Container(
                name="celery-worker",
                image=image_specs[5],
                env=[
                    V1EnvVar(name="CELERY_BROKER_URL", value="redis://localhost:6379/0"),
                    V1EnvVar(name="CELERY_RESULT_BACKEND", value="redis://localhost:6379/0"),
                    V1EnvVar(name="TEMP_FILE_DIR", value="/tmp/pdf_conversions"),
                ],
                volume_mounts=[
                    V1VolumeMount(name="pdf-temp", mount_path="/tmp/pdf_conversions")
                ],
                command=["celery", "-A", "tasks", "worker", "--loglevel=info"],
            ),
            V1Container(
                name="pdf-api",
                image=image_specs[4],
                ports=[V1ContainerPort(container_port=8004)],
                env=[
                    V1EnvVar(name="CELERY_BROKER_URL", value="redis://localhost:6379/0"),
                    V1EnvVar(name="CELERY_RESULT_BACKEND", value="redis://localhost:6379/0"),
                    V1EnvVar(name="REDIS_HOST", value="localhost"),
                    V1EnvVar(name="REDIS_PORT", value="6379"),
                    V1EnvVar(name="TEMP_FILE_DIR", value="/tmp/pdf_conversions"),
                ],
                volume_mounts=[
                    V1VolumeMount(name="pdf-temp", mount_path="/tmp/pdf_conversions")
                ],
                command=[
                    "/bin/bash",
                    "-c",
                    """
                    until celery -A tasks inspect ping; do
                    echo "Waiting for Celery worker to be ready..."
                    sleep 2
                    done
                    uvicorn main:app --host 0.0.0.0 --port 8004
                    """,
                ],
            ),
            V1Container(
                name="agent-service",
                image=image_specs[0],
                ports=[V1ContainerPort(container_port=8964)],
                env=[
                    V1EnvVar(
                        name="NVIDIA_API_KEY",
                        value="$(_UNION_SAMHITA-NVIDIA-BUILD-API-KEY)",
                    ),
                    V1EnvVar(name="REDIS_URL", value="redis://localhost:6379"),
                    V1EnvVar(name="MODEL_CONFIG_PATH", value="/root/models.json"),
                    V1EnvVar(name="MINIO_ENDPOINT", value="localhost:9000"),
                ],
                command=["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8964"],
            ),
            V1Container(
                name="pdf-service",
                image=image_specs[2],
                ports=[V1ContainerPort(container_port=8003)],
                env=[
                    V1EnvVar(name="REDIS_URL", value="redis://localhost:6379"),
                    V1EnvVar(
                        name="MODEL_API_URL",
                        value="http://localhost:8004",
                    ),
                ],
                command=[
                    "/bin/sh",
                    "-c",
                    f"{check_pdf_api_script} && uvicorn main:app --host 0.0.0.0 --port 8003",
                ],
            ),
            V1Container(
                name="tts-service",
                image=image_specs[3],
                ports=[V1ContainerPort(container_port=8889)],
                env=[
                    V1EnvVar(
                        name="MAX_CONCURRENT_REQUESTS",
                        value="1",
                    ),
                    V1EnvVar(
                        name="ELEVENLABS_API_KEY",
                        value="$(_UNION_SAMHITA-ELEVENLABS-API-KEY)",
                    ),
                    V1EnvVar(name="REDIS_URL", value="redis://localhost:6379"),
                ],
                command=["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8889"],
            ),
            V1Container(
                name="api-service",
                image=image_specs[1],
                ports=[V1ContainerPort(container_port=8002)],
                env=[
                    V1EnvVar(name="PDF_SERVICE_URL", value="http://localhost:8003"),
                    V1EnvVar(name="AGENT_SERVICE_URL", value="http://localhost:8964"),
                    V1EnvVar(name="TTS_SERVICE_URL", value="http://localhost:8889"),
                    V1EnvVar(name="REDIS_URL", value="redis://localhost:6379"),
                    V1EnvVar(name="MINIO_ENDPOINT", value="localhost:9000"),
                ],
                command=[
                    "/bin/sh",
                    "-c",
                    f"{check_urls_script} && uvicorn main:app --host 0.0.0.0 --port 8002 --ws auto",
                ],
            ),
        ],
        volumes=[
            V1Volume(name="minio-data", empty_dir={}),
            V1Volume(name="pdf-temp", empty_dir={}),
        ],
    )
)


actor = ActorEnvironment(
    name="blueprints-pdf-to-podcast-actor",
    replica_count=1,
    container_image=fl.ImageSpec(
        name="blueprints-pdf-to-podcast",
        packages=[
            "websockets",
            "fastapi",
            "uvicorn[standard]",
            "python-multipart",
            "pydantic",
            "redis",
            "asyncio",
            "minio",
            "httpx",
            "jinja2",
            "ruff",
            "ujson",
            "union",
            "kubernetes",
        ],
        registry=REGISTRY,
        builder="union",
    ),
    ttl_seconds=300,
    secret_requests=[
        fl.Secret(key="SAMHITA-NVIDIA-BUILD-API-KEY"),
        fl.Secret(key="SAMHITA-ELEVENLABS-API-KEY"),
    ],
    pod_template=actor_pod_template,
    requests=fl.Resources(cpu="8", mem="100Gi", gpu="1"),  # GPU unnecessary.
)


def wait_for_service(
    url: str,
    interval: int = 15,
    timeout: int = 1200,
):
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Service at {url} did not become available within {timeout} seconds"
            )

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"Service at {url} is ready.")
                return
        except requests.RequestException:
            print(f"Waiting for service at {url}")

        time.sleep(interval)


@actor.task
def pdf_to_podcast(
    target_pdfs: list[FlyteFile] = ["samples/investorpres-main.pdf"],
    context_pdfs: list[FlyteFile] = [
        "samples/bofa-context.pdf",
        "samples/citi-context.pdf",
    ],
) -> FlyteFile:
    wait_for_service("http://localhost:8002/health")

    job_id = generate_podcast(
        target_pdf_paths=[target_pdf.download() for target_pdf in target_pdfs],
        context_pdf_paths=[context_pdf.download() for context_pdf in context_pdfs],
        name="NVIDIA Earnings Analysis",
        duration=15,
        speaker_1_name="Alex",
        is_monologue=True,
        guide="Focus on NVIDIA's earnings and the key points driving it's growth",
    )
    wait_for_completion(job_id)

    url = f"http://localhost:8002/output/{job_id}?userId=test-userid"
    output_file = "nvidia_earnings_analysis.mp3"

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Audio file saved as {output_file}")
    else:
        print(f"Failed to download audio. Status code: {response.status_code}")

    return FlyteFile(path=output_file)
