from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import NamedTuple

import flyte
from flyte.io import Dir

GITHUB_USERNAME = "parnianz"
GITHUB_EMAIL    = "parnianzargham@gmail.com"

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

# ---------------------------------------------------------------------------
# Secrets
#   union create secret github_token
#   union create secret internal-anthropic-api-key
# ---------------------------------------------------------------------------

_secrets = [
    flyte.Secret(key="github_token",               as_env_var="GITHUB_TOKEN"),
    flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
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

arch_env = flyte.TaskEnvironment(
    name="automl-arch",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=_secrets,
)

# Fixed GPU spec — extra packages detected by ArchitectureAgent are pip-installed
# at the start of the task body rather than baked into the image.
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
    depends_on=[data_env, arch_env, research_env],
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resolve_data_path(cleaned: Path) -> str:
    """
    Return the right local_data_path from the downloaded data_dir/cleaned folder.
    - ImageFolder: cleaned/ has class subdirectories → return cleaned/ itself
    - Tabular: cleaned/ has a single file → return that file
    """
    if not cleaned.exists():
        return str(cleaned)
    subdirs = [p for p in cleaned.iterdir() if p.is_dir()]
    if subdirs:
        # ImageFolder layout — DATA_PATH must point to the directory, not a file
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
    """
    Download, profile, and clean the dataset.
    Returns a Dir containing profile.json + cleaned data.
    Cached — re-running with the same inputs skips the download.
    max_samples: if > 0, cap the dataset at this many rows (stratified).
    """
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
# Task 2: Architecture Agent
# ---------------------------------------------------------------------------

class ArchResult(NamedTuple):
    branch_name:         str  # GitHub branch created by ArchitectureAgent
    experiment_folder:   str  # subfolder inside the branch (Claude-generated + timestamp)
    compute_config_json: str  # {"gpu","memory","cpu","disk"} — informational
    extra_packages_json: str  # JSON list of pip packages detected from train.py


@arch_env.task
async def run_architecture_agent(
    data_dir: Dir,
    github_repo: str,
    time_budget_per_experiment_seconds: float = 9000.0,
    max_experiments: int = 20,
) -> ArchResult:
    """
    Read DataProfile, generate train.py + program.md via Claude,
    push to a new GitHub branch under a Claude-generated folder name.
    """
    from agents.data_agent import DataProfile
    from agents.architecture_agent import ArchitectureAgent

    local = Path("/tmp/automl_arch_input")
    local.mkdir(parents=True, exist_ok=True)
    await data_dir.download(local_path=str(local))

    profile = DataProfile.from_dict(json.loads((local / "profile.json").read_text()))
    profile.local_data_path = _resolve_data_path(local / "cleaned")

    agent = ArchitectureAgent(
        github_repo=github_repo,
        github_token=os.environ["GITHUB_TOKEN"],
    )
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

@research_env.task(report=True)
async def run_research(
    data_dir: Dir,
    branch_name: str,
    experiment_folder: str,
    github_repo: str,
    extra_packages_json: str,
    max_experiments: int = 20,
    time_budget_per_experiment_seconds: float = 9000.0,
) -> str:
    """
    Clone the GitHub branch pushed by ArchitectureAgent and launch Claude Code CLI
    to run the full autoresearch loop (train → parse → keep/revert → commit → repeat).
    Extra packages detected by ArchitectureAgent are pre-installed so the first
    train.py run doesn't fail on missing imports.
    """
    import subprocess
    from agents.research_agent import ResearchAgent

    # Pre-install packages that ArchitectureAgent detected train.py needs
    extra_packages = json.loads(extra_packages_json)
    if extra_packages:
        print(f"Pre-installing packages: {extra_packages}", flush=True)
        subprocess.run(["pip", "install", "--quiet"] + extra_packages, check=False)

    data_local = Path("/tmp/automl_research_data")
    data_local.mkdir(parents=True, exist_ok=True)
    await data_dir.download(local_path=str(data_local))
    local_data_path = _resolve_data_path(data_local / "cleaned")

    agent = ResearchAgent(
        github_repo=github_repo,
        github_token=os.environ["GITHUB_TOKEN"],
    )
    return agent.run(
        branch_name=branch_name,
        experiment_folder=experiment_folder,
        local_data_path=local_data_path,
        max_experiments=max_experiments,
        time_budget_per_experiment_seconds=time_budget_per_experiment_seconds,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@pipeline_env.task
async def automl_pipeline(
    dataset_link: str  = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    target_column: str = "Survived",
    domain: str        = "auto",
    github_repo: str   = "unionai-oss/autoresearch_exps",
    max_experiments: int = 20,
    time_budget_per_experiment_seconds: float = 9000.0,
    max_samples: int = 0,
) -> str:
    """Full AutoML pipeline: data → architecture → research loop."""
    data_dir = await run_data_agent(dataset_link, target_column, domain, max_samples=max_samples)

    branch_name, experiment_folder, _, extra_packages_json = (
        await run_architecture_agent(
            data_dir=data_dir,
            github_repo=github_repo,
            time_budget_per_experiment_seconds=time_budget_per_experiment_seconds,
            max_experiments=max_experiments,
        )
    )

    return await run_research(
        data_dir=data_dir,
        branch_name=branch_name,
        experiment_folder=experiment_folder,
        github_repo=github_repo,
        extra_packages_json=extra_packages_json,
        max_experiments=max_experiments,
        time_budget_per_experiment_seconds=time_budget_per_experiment_seconds,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoML Platform — submits to Union cloud",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_link",    type=str, default=None)
    parser.add_argument("--target_column",   type=str, default=None)
    parser.add_argument("--domain",          type=str, default="auto")
    parser.add_argument("--github_repo",     type=str, default="unionai-oss/autoresearch_exps",
                        help="GitHub repo where experiments are pushed (owner/repo)")
    parser.add_argument("--max_experiments", type=int, default=20)
    parser.add_argument("--max_samples",     type=int, default=0,
                        help="Cap dataset at N rows (0 = no limit, stratified for classification)")
    parser.add_argument("--time_budget",     type=float, default=1800.0,
                        help="Seconds per experiment")
    args = parser.parse_args()

    if not args.dataset_link:
        args.dataset_link = input("Dataset link (URL / local path / HuggingFace ID): ").strip()
    if not args.target_column:
        args.target_column = input("Target column / objective: ").strip()

    import pathlib
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.with_runcontext().run(
        automl_pipeline,
        dataset_link=args.dataset_link,
        target_column=args.target_column,
        domain=args.domain,
        github_repo=args.github_repo,
        max_experiments=args.max_experiments,
        time_budget_per_experiment_seconds=args.time_budget,
        max_samples=args.max_samples,
    )
    print(f"\nPipeline submitted!")
    print(f"Run URL: {run.url}")
