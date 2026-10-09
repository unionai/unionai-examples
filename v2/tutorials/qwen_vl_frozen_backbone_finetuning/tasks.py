import asyncio
import json
import math
import os
from pathlib import Path
from typing import Optional

import flyte
import flyte.errors
import flyte.report
import lightning as L
import torch
from callbacks import (
    AdapterMetricsCallback,
    LiveTrainingReportCallback,
    RecoveryArtifactCallback,
)
from config import (
    DEFAULT_CHECKPOINT_BASE_URI,
    DEFAULT_MODEL_NAME,
    DEVICES_PER_NODE,
    NUM_NODES,
    Config,
    driver_env,
    evaluation_env,
    training_env,
)
from data import QwenOcclusionDataset, build_data_loaders, prepare_occlusion_dataset
from flyte.io import Dir
from flyteplugins.jsonl import JsonlFile
from flyteplugins.wandb import get_wandb_run, wandb_config, wandb_init
from model import (
    QwenVLAdapterModule,
    ResidualOcclusionAdapter,
    dense_images_to_packed_pixels,
    packed_pixels_to_dense_images,
)
from utils import (
    build_qwen_adapter_report_html,
    build_recovery_uri,
    download_dir_async,
    download_dir_sync,
    find_resume_checkpoint,
    load_jsonl_records,
    load_jsonl_records_async,
    normalize_text,
    set_seed,
    write_training_artifacts,
)


# {{docs-fragment training-task-signature}}
@wandb_init
@training_env.task(report=True)
def train_qwen_adapter_multinode(
    train_manifest: JsonlFile,
    val_manifest: JsonlFile,
    images_dir: Dir,
    config: Config,
    resume_from: Optional[Dir] = None,
    recovery_uri: Optional[str] = None,
) -> Optional[Dir]:
    # {{/docs-fragment training-task-signature}}
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.strategies import DeepSpeedStrategy

    set_seed(config.seed)

    train_records = load_jsonl_records(train_manifest)
    val_records = load_jsonl_records(val_manifest)

    images_root = download_dir_sync(images_dir)

    train_loader, val_loader = build_data_loaders(
        train_records=train_records,
        val_records=val_records,
        images_root=images_root,
        model_name=config.model_name,
        batch_size=config.per_device_batch_size,
        num_workers=config.num_workers,
        max_length=config.max_length,
        image_size=config.image_size,
    )

    # {{docs-fragment grad-accum}}
    world_size = NUM_NODES * DEVICES_PER_NODE
    per_step_batch = world_size * config.per_device_batch_size
    grad_accum_steps = max(
        1,
        math.ceil(config.target_global_batch_size / max(1, per_step_batch)),
    )
    # {{/docs-fragment grad-accum}}

    module = QwenVLAdapterModule(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        reconstruction_loss_weight=config.reconstruction_loss_weight,
    )

    resume_ckpt_path = None
    prior_metrics: list[dict] = []
    if resume_from is not None:
        resume_root = download_dir_sync(resume_from)
        resume_checkpoint = find_resume_checkpoint(resume_root)
        if resume_checkpoint is not None:
            resume_ckpt_path = str(resume_checkpoint)
            print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        else:
            print(
                "Resume artifacts provided, but no checkpoint was found. Starting fresh."
            )
        metrics_path = resume_root / "metrics.json"
        if metrics_path.exists():
            try:
                prior_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Could not load prior metrics for live report: {e}")

    output_dir = Path("/tmp/qwen_vl_multinode_output")
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    recovery_uri = recovery_uri or build_recovery_uri(
        os.environ.get("FLYTE_RECOVERY_BASE_URI"),
        flyte.ctx().action.run_name,
    )

    metrics_callback = AdapterMetricsCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="adapter-{epoch:02d}-{val_total_loss:.3f}",
        monitor="val_total_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    recovery_callback = RecoveryArtifactCallback(
        checkpoints_dir=checkpoints_dir,
        output_dir=output_dir,
        config=config,
        metrics_callback=metrics_callback,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
        resumed_from=resume_ckpt_path,
        recovery_uri=recovery_uri,
    )
    live_report_callback = LiveTrainingReportCallback(
        report_every_n_steps=config.report_every_n_steps,
        resumed_from=resume_ckpt_path,
        recovery_callback=recovery_callback,
        prior_metrics=prior_metrics,
    )

    # {{docs-fragment deepspeed-strategy}}
    strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=False,
        offload_parameters=False,
        process_group_backend="nccl",
        exclude_frozen_parameters=True,
    )
    # {{/docs-fragment deepspeed-strategy}}

    run = get_wandb_run()
    wandb_logger = WandbLogger(experiment=run, log_model=False) if run else False
    if run and getattr(run, "config", None) is not None:
        run.config.update(config.to_dict(), allow_val_change=True)

    # {{docs-fragment trainer-setup}}
    trainer = L.Trainer(
        accelerator="gpu",
        devices=DEVICES_PER_NODE,
        num_nodes=NUM_NODES,
        strategy=strategy,
        logger=wandb_logger,
        precision="bf16-mixed",
        max_epochs=config.epochs,
        accumulate_grad_batches=grad_accum_steps,
        callbacks=[
            checkpoint_callback,
            metrics_callback,
            recovery_callback,
            live_report_callback,
        ],
        gradient_clip_val=1.0,
        benchmark=True,
        log_every_n_steps=1,
    )
    # {{/docs-fragment trainer-setup}}

    final_status = "completed"
    error_message = None

    try:
        trainer.fit(module, train_loader, val_loader, ckpt_path=resume_ckpt_path)
        if trainer.global_rank == 0:
            live_report_callback._push_update(
                trainer=trainer,
                pl_module=module,
                status="completed",
                phase="final",
                train_total=trainer.callback_metrics.get("train/total_loss_epoch"),
                train_lm=trainer.callback_metrics.get("train/lm_loss_epoch"),
                train_recon=trainer.callback_metrics.get(
                    "train/reconstruction_loss_epoch"
                ),
                val_total=trainer.callback_metrics.get("val/total_loss"),
                note="Training completed successfully.",
            )
    except Exception as e:
        final_status = "failed"
        error_message = str(e)
        recovery_callback.sync_artifacts(
            trainer,
            module,
            status=final_status,
            error_message=error_message,
        )
        if trainer.global_rank == 0:
            live_report_callback._push_update(
                trainer=trainer,
                pl_module=module,
                status="failed",
                phase="final",
                train_total=trainer.callback_metrics.get("train/total_loss_epoch"),
                train_lm=trainer.callback_metrics.get("train/lm_loss_epoch"),
                train_recon=trainer.callback_metrics.get(
                    "train/reconstruction_loss_epoch"
                ),
                val_total=trainer.callback_metrics.get("val/total_loss"),
                note=f"Training failed: {error_message}",
            )
        raise

    if trainer.global_rank == 0:
        artifact_path = output_dir / "adapter_artifact.pt"
        torch.save(
            {
                "model_name": config.model_name,
                "adapter_state_dict": module.adapter.state_dict(),
                "config": config.to_dict(),
                "total_params": module.total_params,
                "trainable_params": module.trainable_params,
                "world_size": world_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "best_model_path": checkpoint_callback.best_model_path,
                "last_model_path": checkpoint_callback.last_model_path,
                "resumed_from": resume_ckpt_path,
                "recovery_uri": recovery_uri,
            },
            artifact_path,
        )
        write_training_artifacts(
            output_dir,
            config=config,
            module=module,
            metrics_history=metrics_callback.history,
            world_size=world_size,
            grad_accum_steps=grad_accum_steps,
            best_model_path=checkpoint_callback.best_model_path,
            last_model_path=checkpoint_callback.last_model_path,
            resumed_from=resume_ckpt_path,
            status=final_status,
            error_message=error_message,
        )
        return Dir.from_local_sync(str(output_dir))
    return None


# {{docs-fragment evaluation-task-header}}
@evaluation_env.task
async def evaluate_qwen_adapter(
    val_manifest: JsonlFile,
    images_dir: Dir,
    training_artifacts: Dir,
    config: Config,
) -> Dir:
    # {{/docs-fragment evaluation-task-header}}
    from torchvision.transforms.functional import pil_to_tensor
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    set_seed(config.seed)

    val_records, images_root, artifacts_root = await asyncio.gather(
        load_jsonl_records_async(val_manifest),
        download_dir_async(images_dir),
        download_dir_async(training_artifacts),
    )

    artifact = torch.load(artifacts_root / "adapter_artifact.pt", map_location="cpu")
    adapter = ResidualOcclusionAdapter()
    adapter.load_state_dict(artifact["adapter_state_dict"])
    adapter = adapter.cuda().to(dtype=torch.bfloat16).eval()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        artifact["model_name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.requires_grad_(False)
    model.eval()

    vision_patch_size = int(model.config.vision_config.patch_size)
    temporal_patch_size = int(
        getattr(model.config.vision_config, "temporal_patch_size", 1)
    )

    processor = AutoProcessor.from_pretrained(
        artifact["model_name"],
        min_pixels=config.image_size * config.image_size,
        max_pixels=config.image_size * config.image_size,
    )
    tokenizer = processor.tokenizer

    val_dataset = QwenOcclusionDataset(val_records, images_root)
    predictions = []
    matches = 0
    sample_images_dir = Path("/tmp/qwen_vl_eval_output") / "samples"
    sample_images_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(config.eval_examples, len(val_dataset))):
        item = val_dataset[idx]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["prompt"]},
                ],
            }
        ]
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = processor(
            text=[prompt_text],
            images=[item["image"]],
            padding=True,
            return_tensors="pt",
        )
        mask_tensor = (
            (pil_to_tensor(item["mask_image"]).float() / 255.0).unsqueeze(0).cuda()
        )
        pixel_values = model_inputs["pixel_values"].cuda().to(torch.bfloat16)

        with torch.no_grad():
            if pixel_values.ndim == 2:
                dense_pixels = packed_pixels_to_dense_images(
                    pixel_values,
                    model_inputs["image_grid_thw"],
                    patch_size=vision_patch_size,
                    temporal_patch_size=temporal_patch_size,
                )
                adapted_dense = adapter(dense_pixels, mask_tensor.to(torch.bfloat16))
                adapted_pixels = dense_images_to_packed_pixels(
                    adapted_dense,
                    model_inputs["image_grid_thw"],
                    patch_size=vision_patch_size,
                    temporal_patch_size=temporal_patch_size,
                )
            else:
                adapted_pixels = adapter(pixel_values, mask_tensor.to(torch.bfloat16))

            generate_kwargs = {
                "input_ids": model_inputs["input_ids"].cuda(),
                "attention_mask": model_inputs["attention_mask"].cuda(),
                "pixel_values": adapted_pixels,
                "max_new_tokens": 16,
                "do_sample": False,
            }
            if "image_grid_thw" in model_inputs:
                generate_kwargs["image_grid_thw"] = model_inputs[
                    "image_grid_thw"
                ].cuda()

            generated_ids = model.generate(**generate_kwargs)

        generated_text = tokenizer.batch_decode(
            generated_ids[:, model_inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        prediction_norm = normalize_text(generated_text)
        label_norm = normalize_text(item["label"])
        prediction_words = set(prediction_norm.split())

        is_match = label_norm in prediction_words or prediction_norm == label_norm
        matches += int(is_match)

        sample_image_path = sample_images_dir / f"sample-{idx:03d}.png"
        item["image"].save(sample_image_path)

        predictions.append(
            {
                "index": idx,
                "label": item["label"],
                "prediction": generated_text,
                "match": is_match,
                "occluded_image_path": f"samples/{sample_image_path.name}",
            }
        )

    output_dir = Path("/tmp/qwen_vl_eval_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "evaluated_examples": len(predictions),
                "exact_match": matches / max(1, len(predictions)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return await Dir.from_local(str(output_dir))


# {{docs-fragment driver-task-signature}}
@driver_env.task(report=True)
async def qwen_vl_multinode_deepspeed(
    model_name: str = DEFAULT_MODEL_NAME,
    max_train_samples: int = 1024,
    max_val_samples: int = 256,
    epochs: int = 8,
    per_device_batch_size: int = 1,
    target_global_batch_size: int = 16,
    learning_rate: float = 2e-4,
    reconstruction_loss_weight: float = 0.35,
    eval_examples: int = 16,
    resume_training_artifacts: Optional[Dir] = None,
    checkpoint_base_uri: Optional[str] = DEFAULT_CHECKPOINT_BASE_URI,
    wandb_project: str = "qwen-vl-multinode-deepspeed",
    wandb_entity: Optional[str] = None,
) -> Optional[Dir]:
    # {{/docs-fragment driver-task-signature}}
    config = Config(
        model_name=model_name,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        target_global_batch_size=target_global_batch_size,
        learning_rate=learning_rate,
        reconstruction_loss_weight=reconstruction_loss_weight,
        eval_examples=eval_examples,
    )

    recovery_uri = build_recovery_uri(
        checkpoint_base_uri or os.environ.get("FLYTE_RECOVERY_BASE_URI"),
        flyte.ctx().action.run_name,
    )

    train_manifest, val_manifest, images = await prepare_occlusion_dataset(
        config=config
    )

    # {{docs-fragment recovery-handler}}
    try:
        with wandb_config(
            project=wandb_project,
            entity=wandb_entity,
        ):
            training_artifacts = await train_qwen_adapter_multinode.aio(
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                images_dir=images,
                config=config,
                resume_from=resume_training_artifacts,
                recovery_uri=recovery_uri,
            )
    except flyte.errors.RuntimeUserError as e:
        if recovery_uri is None:
            raise e
        print(f"Training failed - recovering latest checkpoint bundle: {recovery_uri}")
        try:
            recovered_artifacts = Dir(path=recovery_uri)
            recovered_root = await download_dir_async(recovered_artifacts)
            flyte.report.log(
                build_qwen_adapter_report_html(recovered_root, None),
                do_flush=True,
            )
            return recovered_artifacts
        except Exception:
            raise e
    # {{/docs-fragment recovery-handler}}

    if training_artifacts is None:
        return None

    evaluation_artifacts = await evaluate_qwen_adapter(
        val_manifest=val_manifest,
        images_dir=images,
        training_artifacts=training_artifacts,
        config=config,
    )

    training_root, evaluation_root = await asyncio.gather(
        download_dir_async(training_artifacts),
        download_dir_async(evaluation_artifacts),
    )

    flyte.report.log(
        build_qwen_adapter_report_html(training_root, evaluation_root),
        do_flush=True,
    )

    return training_artifacts
