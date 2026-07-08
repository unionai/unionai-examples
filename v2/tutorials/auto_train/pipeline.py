"""
AutoML pipeline — task environments, tasks, and pipeline definition.

Imported lazily by app.py so that flyte serve only builds the web image.
Task images (cpu_image, gpu_image) are built when the pipeline is first run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import flyte
from flyte.io import Dir


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

_base_packages = [
    "anthropic>=0.40.0",
    "claude-agent-sdk",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
    "scikit-learn>=1.4.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "datasets>=2.18.0",
    "numpy>=1.26.0",
    "Pillow>=10.0.0",
    "requests>=2.31.0",
    "gitpython>=3.1.0",
    "PyGithub>=2.5.0",
    "plotly>=5.0.0",
]

_gpu_packages = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "timm>=0.9.0",
    "transformers>=4.40.0",
    "accelerate>=0.30.0",
    "einops>=0.7.0",
]

cpu_image = (
    flyte.Image.from_debian_base(name="automl-cpu")
    .with_apt_packages("git")
    .with_pip_packages(*_base_packages)
    .with_source_folder(Path(__file__).parent, copy_contents_only=True)
)

gpu_image = (
    flyte.Image.from_debian_base(name="automl-gpu")
    .with_apt_packages("git")
    .with_pip_packages(*_base_packages, *_gpu_packages)
    .with_source_folder(Path(__file__).parent, copy_contents_only=True)
)

_secrets = [
    flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    flyte.Secret(key="github-token", as_env_var="GITHUB_TOKEN"),
]


# ---------------------------------------------------------------------------
# Task environments
# ---------------------------------------------------------------------------

data_env = flyte.TaskEnvironment(
    name="automl-data",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi", disk="20Gi"),
    secrets=_secrets,
)

design_env = flyte.TaskEnvironment(
    name="automl-design",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=_secrets,
)

research_env = flyte.TaskEnvironment(
    name="automl-research",
    image=gpu_image,
    resources=flyte.Resources(cpu=8, memory="32Gi", gpu="T4:1", disk="100Gi"),
    secrets=_secrets,
)

pipeline_env = flyte.TaskEnvironment(
    name="automl-pipeline",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=_secrets,
    depends_on=[data_env, design_env, research_env],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resolve_data_path(cleaned: Path) -> str:
    if not cleaned.exists():
        return str(cleaned)
    subdirs = [p for p in cleaned.iterdir() if p.is_dir()]
    if subdirs:
        return str(cleaned)
    files = [p for p in cleaned.iterdir() if p.is_file()]
    return str(files[0]) if files else str(cleaned)


# ---------------------------------------------------------------------------
# Task 1: Data Agent
# ---------------------------------------------------------------------------

@data_env.task
async def run_data_agent(
    dataset_link: str,
    target_column: str,
    domain: str = "auto",
    max_samples: int = 0,
) -> Dir:
    import shutil
    from agents.data_agent import DataAgent

    out = Path("/tmp/automl_data_output")
    out.mkdir(parents=True, exist_ok=True)

    agent = DataAgent(work_dir=str(out / "work"))
    profile = agent.run(dataset_link, target_column, domain, max_samples=max_samples)

    (out / "profile.json").write_text(json.dumps(profile.to_dict()))

    cleaned = Path(profile.local_data_path)
    dest    = out / "cleaned"
    if cleaned.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(cleaned, dest / cleaned.name)
    elif cleaned.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(cleaned), str(dest))

    return await Dir.from_local(str(out))


# ---------------------------------------------------------------------------
# Task 2: Design Agent
# ---------------------------------------------------------------------------

class ArchResult(NamedTuple):
    branch_name:         str
    experiment_folder:   str
    compute_config_json: str
    extra_packages_json: str


@design_env.task
async def run_design_agent(
    data_dir: Dir,
    github_repo: str,
    time_budget_per_experiment_seconds: float = 9000.0,
    max_experiments: int = 20,
) -> ArchResult:
    from agents.data_agent import DataProfile
    from agents.design_agent import DesignAgent

    local = Path("/tmp/automl_arch_input")
    local.mkdir(parents=True, exist_ok=True)
    await data_dir.download(local_path=str(local))

    profile = DataProfile.from_dict(json.loads((local / "profile.json").read_text()))
    profile.local_data_path = _resolve_data_path(local / "cleaned")

    agent = DesignAgent(github_repo=github_repo)
    branch_name, folder_name, compute_config, packages = agent.run(
        profile=profile,
        time_budget_per_experiment_seconds=time_budget_per_experiment_seconds,
        max_experiments=max_experiments,
    )

    return ArchResult(
        branch_name=branch_name,
        experiment_folder=folder_name,
        compute_config_json=json.dumps(compute_config),
        extra_packages_json=json.dumps(packages),
    )


# ---------------------------------------------------------------------------
# Task 3: Research Agent
# ---------------------------------------------------------------------------

@research_env.task
async def run_research(
    data_dir: Dir,
    branch_name: str,
    experiment_folder: str,
    github_repo: str,
    extra_packages_json: str,
    job_id: str = "",
    webapp_endpoint: str = "",
    max_experiments: int = 20,
    time_budget_per_experiment_seconds: float = 9000.0,
) -> str:
    import asyncio as _asyncio
    import functools
    import subprocess
    import urllib.request
    from agents.research_agent import ResearchAgent

    extra_packages = json.loads(extra_packages_json)
    if extra_packages:
        print(f"Pre-installing packages: {extra_packages}", flush=True)
        subprocess.run(["pip", "install", "--quiet"] + extra_packages, check=False)

    data_local = Path("/tmp/automl_research_data")
    data_local.mkdir(parents=True, exist_ok=True)
    await data_dir.download(local_path=str(data_local))
    local_data_path = _resolve_data_path(data_local / "cleaned")

    agent = ResearchAgent(github_repo=github_repo)

    # Run blocking training loop in a thread so the event loop stays alive.
    result = await _asyncio.get_running_loop().run_in_executor(
        None,
        functools.partial(
            agent.run,
            branch_name=branch_name,
            experiment_folder=experiment_folder,
            local_data_path=local_data_path,
            max_experiments=max_experiments,
            time_budget_per_experiment_seconds=time_budget_per_experiment_seconds,
        ),
    )

    if job_id and webapp_endpoint:
        try:
            callback_url = f"{webapp_endpoint}/result/{job_id}"
            payload = json.dumps({"result": result}).encode()
            req = urllib.request.Request(
                callback_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=30)
            print(f"Result callback sent to {callback_url}", flush=True)
        except Exception as exc:
            print(f"Result callback failed (non-fatal): {exc}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@pipeline_env.task
async def automl_pipeline(
    dataset_link: str  = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    target_column: str = "Survived",
    domain: str        = "auto",
    github_repo: str   = "unionai-oss/autoresearch-experiments",
    job_id: str        = "",
    webapp_endpoint: str = "",
    max_experiments: int = 20,
    time_budget_per_experiment_seconds: float = 9000.0,
    max_samples: int = 0,
) -> str:
    """Full AutoML pipeline: data → architecture → research loop."""
    data_dir = await run_data_agent(dataset_link, target_column, domain, max_samples=max_samples)

    branch_name, experiment_folder, _, extra_packages_json = await run_design_agent(
        data_dir=data_dir,
        github_repo=github_repo,
        time_budget_per_experiment_seconds=time_budget_per_experiment_seconds,
        max_experiments=max_experiments,
    )

    return await run_research(
        data_dir=data_dir,
        branch_name=branch_name,
        experiment_folder=experiment_folder,
        github_repo=github_repo,
        extra_packages_json=extra_packages_json,
        job_id=job_id,
        webapp_endpoint=webapp_endpoint,
        max_experiments=max_experiments,
        time_budget_per_experiment_seconds=time_budget_per_experiment_seconds,
    )
