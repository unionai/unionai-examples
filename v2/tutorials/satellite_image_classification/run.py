"""
Flyte/Union pipeline for satellite image classification.

Three-task pipeline:
1. load_dataset  — download EuroSAT from torchvision, preprocess, cache as Dir (CPU)
2. train_model   — two-phase EfficientNet-B0 training with W&B logging (GPU)
3. create_report — render training curves in the Union UI (CPU)
"""

import json

import flyte
from flyte.io import Dir
from flyteplugins.wandb import wandb_config, wandb_init

from config import TrainingConfig, dataset_env, pipeline_env, report_env, training_env
from dataset import download_eurosat
from training import train_satellite_classifier

TRAINING_CONFIG = TrainingConfig(
    phase1_epochs=7,
    phase2_epochs=10,
    phase1_lr=2e-3,
    phase2_lr=1e-4,
    batch_size=64,
    num_workers=0,
    log_interval=50,
    tsne_interval=3,
)


@dataset_env.task
async def load_dataset() -> Dir:
    """
    Download raw EuroSAT JPEG files and cache as flyte.io.Dir.
    Runs once — result is reused on subsequent pipeline runs (cache="auto").
    """
    return await download_eurosat()


@wandb_init
@training_env.task
async def train_model(dataset_dir: Dir, config_json: str) -> Dir:
    """
    Download the raw dataset Dir, run two-phase training,
    and return training metrics as a Dir for the report task.
    """
    from pathlib import Path

    local_dir = Path("/tmp/eurosat_local")
    local_dir.mkdir(parents=True, exist_ok=True)
    await dataset_dir.download(local_path=str(local_dir))

    config = TrainingConfig(**json.loads(config_json))
    result = train_satellite_classifier(config=config, dataset_path=str(local_dir))

    output_dir = Path("/tmp/training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result["metrics"]))

    return await Dir.from_local(str(output_dir))


@report_env.task(report=True)
async def create_report(results_dir: Dir) -> None:
    """
    Download training metrics and render loss/accuracy curves
    in the Union UI report panel.
    """
    import plotly.graph_objects as go
    from pathlib import Path

    local_dir = Path("/tmp/training_report")
    local_dir.mkdir(parents=True, exist_ok=True)
    await results_dir.download(local_path=str(local_dir))

    matches = list(local_dir.glob("**/metrics.json"))
    if not matches:
        raise RuntimeError(f"metrics.json not found under {local_dir}")
    local_path = matches[0].parent

    history = json.loads((local_path / "metrics.json").read_text())

    epochs = [e["epoch"] for e in history]
    val_acc = [e["val_acc"] for e in history]
    val_loss = [e["val_loss"] for e in history]
    train_loss = [e["train_loss"] for e in history]
    # phase_boundary: first epoch where phase 2 begins (frozen → fine-tune transition)
    phase_boundary = next((e["epoch"] for e in history if e["phase"] == 2), None)

    def add_phase_line(fig):
        if phase_boundary is not None:
            fig.add_vline(
                x=phase_boundary,
                line_dash="dash",
                line_color="gray",
                annotation_text="Phase 2 start",
            )

    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers", name="Val Accuracy"))
    acc_fig.update_layout(title="Validation Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
    add_phase_line(acc_fig)

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Train Loss"))
    loss_fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Val Loss"))
    loss_fig.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss")
    add_phase_line(loss_fig)

    combined_html = (
        acc_fig.to_html(include_plotlyjs=True, full_html=False)
        + loss_fig.to_html(include_plotlyjs=False, full_html=False)
    )
    flyte.report.log(combined_html, do_flush=True)


@pipeline_env.task
async def satellite_classification_pipeline() -> None:
    """Orchestrate dataset loading, GPU training, and report generation."""
    dataset_dir = await load_dataset()
    results_dir = await train_model(
        dataset_dir=dataset_dir,
        config_json=json.dumps(TRAINING_CONFIG.to_dict()),
    )
    await create_report(results_dir=results_dir)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.with_runcontext(
        custom_context=wandb_config(
            project=TRAINING_CONFIG.wandb_project,
            entity=TRAINING_CONFIG.wandb_entity,
        ),
    ).run(satellite_classification_pipeline)

    print(f"\n✓ Pipeline submitted!")
    print(f"Run URL: {run.url}")
