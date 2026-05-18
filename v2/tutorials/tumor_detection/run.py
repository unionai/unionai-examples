"""
Flyte/Union pipeline for brain tumor MRI classification.

Three-task pipeline:
1. load_dataset  — download Brain Tumor MRI from Hugging Face, cache as Dir (CPU)
2. train_model   — two-phase EfficientNet-B4 training with focal loss (GPU)
3. create_report — render training curves and confusion matrix in the Union UI (CPU)
"""

import json

import flyte
from flyte.io import Dir

from config import TrainingConfig, dataset_env, pipeline_env, report_env, training_env
from dataset import download_tumor_dataset

TRAINING_CONFIG = TrainingConfig(
    phase1_epochs=8,
    phase2_epochs=25,
    phase1_lr=1e-3,
    phase2_lr=5e-5,
    batch_size=16,
    num_workers=0,
    log_interval=50,
    mixup_alpha=0.0,
    image_size=380,
    focal_gamma=3.0,
)

@dataset_env.task
async def load_dataset() -> Dir:
    """
    Download raw Brain Tumor MRI JPEG files from Hugging Face and cache as flyte.io.Dir.
    Runs once — result is reused on subsequent pipeline runs (cache="auto").
    """
    return await download_tumor_dataset()


@training_env.task
async def train_model(dataset_dir: Dir, config_json: str) -> Dir:
    """
    Download the raw dataset Dir, run two-phase training,
    and return training metrics and final predictions as a Dir for the report task.
    """
    from pathlib import Path

    local_dir = Path("/tmp/tumor_local")
    local_dir.mkdir(parents=True, exist_ok=True)
    await dataset_dir.download(local_path=str(local_dir))

    from training import train_tumor_classifier
    config = TrainingConfig(**json.loads(config_json))
    result = train_tumor_classifier(config=config, dataset_path=str(local_dir))

    output_dir = Path("/tmp/training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result["metrics"]))
    (output_dir / "predictions.json").write_text(json.dumps({
        "preds": result["final_preds"],
        "targets": result["final_targets"],
    }))

    return await Dir.from_local(str(output_dir))


@report_env.task(report=True)
async def create_report(results_dir: Dir) -> None:
    """
    Download training metrics and render loss/accuracy curves, confusion matrix,
    and per-class F1 chart in the Union UI report panel.
    """
    import numpy as np
    from pathlib import Path

    from utils import create_confusion_matrix_plot, create_metrics_plots, create_per_class_f1_plot

    local_dir = Path("/tmp/tumor_report")
    local_dir.mkdir(parents=True, exist_ok=True)
    await results_dir.download(local_path=str(local_dir))

    matches = list(local_dir.glob("**/metrics.json"))
    if not matches:
        raise RuntimeError(f"metrics.json not found under {local_dir}")
    local_path = matches[0].parent

    history = json.loads((local_path / "metrics.json").read_text())
    predictions = json.loads((local_path / "predictions.json").read_text())

    preds = np.array(predictions["preds"])
    targets = np.array(predictions["targets"])

    loss_fig, acc_fig = create_metrics_plots(history)
    cm_fig = create_confusion_matrix_plot(preds, targets)
    f1_fig = create_per_class_f1_plot(preds, targets)

    combined_html = (
        acc_fig.to_html(include_plotlyjs=True, full_html=False)
        + loss_fig.to_html(include_plotlyjs=False, full_html=False)
        + cm_fig.to_html(include_plotlyjs=False, full_html=False)
        + f1_fig.to_html(include_plotlyjs=False, full_html=False)
    )
    flyte.report.log(combined_html, do_flush=True)


@pipeline_env.task
async def tumor_detection_pipeline() -> None:
    """Orchestrate dataset loading, GPU training, and report generation."""
    dataset_dir = await load_dataset()
    results_dir = await train_model(
        dataset_dir=dataset_dir,
        config_json=json.dumps(TRAINING_CONFIG.to_dict()),
    )
    await create_report(results_dir=results_dir)


if __name__ == "__main__":
    import pathlib
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.with_runcontext().run(tumor_detection_pipeline)
    print(f"\n✓ Pipeline submitted!")
    print(f"Run URL: {run.url}")
