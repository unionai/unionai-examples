# {{docs-fragment imports}}
import logging
import math
import os
from pathlib import Path
from typing import Optional

import flyte
import flyte.report
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from flyte.io import Dir, File
from flyteplugins.pytorch.task import Elastic
# {{/docs-fragment imports}}

# {{docs-fragment constants}}
NUM_NODES = 1
DEVICES_PER_NODE = 8
VOCAB_SIZE = (
    50257  # GPT-2 BPE tokenizer vocabulary size (constant across all model sizes)
)
N_POSITIONS = 2048  # Maximum sequence length (constant across all model sizes)
# {{/docs-fragment constants}}

# {{docs-fragment image}}
image = flyte.Image.from_debian_base(
    name="distributed_training_h200"
).with_pip_packages(
    "transformers==4.57.3",
    "datasets==4.4.1",
    "tokenizers==0.22.1",
    "huggingface-hub==0.34.0",
    "mosaicml-streaming>=0.7.0",
    "pyarrow==22.0.0",
    "flyteplugins-pytorch>=2.0.0b33",
    "torch==2.9.1",
    "lightning==2.5.6",
    "tensorboard==2.20.0",
    "sentencepiece==0.2.1",
)
# {{/docs-fragment image}}


# {{docs-fragment task-envs}}
data_loading_env = flyte.TaskEnvironment(
    name="data_loading_h200",
    image=image,
    resources=flyte.Resources(cpu=5, memory="28Gi", disk="100Gi"),
    env_vars={
        "HF_DATASETS_CACHE": "/tmp/hf_cache",  # Cache directory for datasets
        "TOKENIZERS_PARALLELISM": "true",  # Enable parallel tokenization
    },
    cache="auto",
)

distributed_llm_training_env = flyte.TaskEnvironment(
    name="distributed_llm_training_h200",
    image=image,
    resources=flyte.Resources(
        cpu=64,
        memory="512Gi",
        gpu=f"H200:{DEVICES_PER_NODE}",
        disk="1Ti",
        shm="16Gi",  # Explicit shared memory for NCCL communication
    ),
    plugin_config=Elastic(nnodes=NUM_NODES, nproc_per_node=DEVICES_PER_NODE),
    env_vars={
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
        "NCCL_DEBUG": "WARN",
    },
    cache="auto",
)

driver_env = flyte.TaskEnvironment(
    name="llm_training_driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    cache="auto",
    depends_on=[data_loading_env, distributed_llm_training_env],
)
# {{/docs-fragment task-envs}}


# {{docs-fragment model-configs}}
MODEL_CONFIGS = {
    "1.5B": {
        "n_embd": 2048,
        "n_layer": 24,
        "n_head": 16,
        "batch_size": 8,
        "learning_rate": 6e-4,
        "checkpoint_every_n_steps": 10,
        "report_every_n_steps": 5,
        "val_check_interval": 100,
    },  # Good for testing and debugging
    "30B": {
        "n_embd": 6656,
        "n_layer": 48,
        "n_head": 52,
        "batch_size": 1,
        "learning_rate": 1.6e-4,
        "checkpoint_every_n_steps": 7500,
        "report_every_n_steps": 200,
        "val_check_interval": 1000,
    },
    "65B": {
        "n_embd": 8192,
        "n_layer": 80,
        "n_head": 64,
        "batch_size": 1,
        "learning_rate": 1.5e-4,
        "checkpoint_every_n_steps": 10000,
        "report_every_n_steps": 250,
        "val_check_interval": 2000,
    },
}


def get_model_config(model_size: str) -> dict:
    if model_size not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model size: {model_size}. Available: {available}")

    return MODEL_CONFIGS[model_size]
# {{/docs-fragment model-configs}}


# {{docs-fragment gpt-config}}
class GPTConfig:
    """Configuration for GPT model."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        n_positions: int = N_POSITIONS,
        n_embd: int = 2048,
        n_layer: int = 24,
        n_head: int = 16,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
# {{/docs-fragment gpt-config}}


# {{docs-fragment gpt-block}}
class GPTBlock(nn.Module):
    """Transformer block with causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = nn.MultiheadAttention(
            config.n_embd,
            config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Get activation function from config
        ACT_FNS = {
            "gelu": nn.GELU(),
            "gelu_new": nn.GELU(approximate="tanh"),  # GPT-2 uses approximate GELU
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "swish": nn.SiLU(),  # SiLU = Swish
        }
        act_fn = ACT_FNS.get(config.activation_function, nn.GELU())

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            act_fn,
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, causal_mask, key_padding_mask=None):
        x_normed = self.ln_1(x)

        # Self-attention with causal and padding masks
        attn_output, _ = self.attn(
            x_normed,  # query
            x_normed,  # key
            x_normed,  # value
            attn_mask=causal_mask,  # Causal mask: (seq_len, seq_len)
            key_padding_mask=key_padding_mask,  # Padding mask: (batch, seq_len)
            need_weights=False,
        )
        x = x + attn_output

        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x
# {{/docs-fragment gpt-block}}


# {{docs-fragment gpt-model}}
class GPTModel(nn.Module):
    """GPT-2 style language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights first before tying
        self.apply(self._init_weights)

        # Weight tying after initialization
        # Tie input embedding and output projection weights
        self.lm_head.weight = self.wte.weight

        # Caches for efficiency (created once, reused)
        self.register_buffer("causal_mask", None, persistent=False)
        self.register_buffer("position_ids_cache", None, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # Create causal mask for autoregressive LM (prevent attending to future)
        # Shape: (seq_len, seq_len), upper triangular = True (masked)
        # This mask is shared across all layers
        if self.causal_mask is None or self.causal_mask.size(0) != seq_length:
            self.causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=device, dtype=torch.bool),
                diagonal=1,
            )

        # Convert attention_mask to key_padding_mask if provided
        # attention_mask: 1 = real token, 0 = padding
        # key_padding_mask: True = ignore (padding), False = attend to
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        # Get position IDs (cache to avoid repeated allocation)
        if (
            self.position_ids_cache is None
            or self.position_ids_cache.size(0) < seq_length
        ):
            self.position_ids_cache = torch.arange(
                0, self.config.n_positions, dtype=torch.long, device=device
            )
        position_ids = (
            self.position_ids_cache[:seq_length].unsqueeze(0).expand(batch_size, -1)
        )

        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        hidden_states = self.drop(token_embeddings + position_embeddings)

        # Apply transformer blocks with masks
        for block in self.h:
            hidden_states = block(hidden_states, self.causal_mask, key_padding_mask)

        # Final layer norm and language modeling head
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
# {{/docs-fragment gpt-model}}


# {{docs-fragment lightning-module-init}}
class GPTPreTrainingModule(L.LightningModule):
    """PyTorch Lightning module for GPT pre-training."""

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 2048,
        n_embd: int = 2048,
        n_layer: int = 24,
        n_head: int = 16,
        learning_rate: float = 6e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
    ):
        super().__init__()
        self.save_hyperparameters()

        config = GPTConfig(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.model = GPTModel(config)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)
    # {{/docs-fragment lightning-module-init}}

    # {{docs-fragment lightning-module-training}}
    def training_step(self, batch, _batch_idx):
        # Convert int32 to int64 (long) - MDS stores as int32 but PyTorch expects long
        input_ids = batch["input_ids"].long()
        labels = batch["labels"].long()

        # Get attention mask if present (optional, for padded sequences)
        # attention_mask: 1 = real token, 0 = padding
        # Note: Current data pipeline creates fixed-length sequences without padding,
        # so attention_mask is not present. If using padded sequences, ensure:
        #   - Padded positions in labels are set to -100 (ignored by cross_entropy)
        #   - attention_mask marks real tokens (1) vs padding (0)
        attention_mask = batch.get("attention_mask", None)

        # Forward pass (causal mask is created internally in GPTModel)
        logits = self(input_ids, attention_mask=attention_mask)

        # Shift logits and labels for causal language modeling
        # Before shift: labels[i] = input_ids[i]
        # After shift: predict input_ids[i+1] from input_ids[:i+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Calculate loss
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Log loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Calculate and log perplexity only on epoch (exp is costly, less frequent is fine)
        perplexity = torch.exp(torch.clamp(loss, max=20.0))
        self.log(
            "train/perplexity",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, _batch_idx):
        # Convert int32 to int64 (long) - MDS stores as int32 but PyTorch expects long
        input_ids = batch["input_ids"].long()
        labels = batch["labels"].long()

        # Get attention mask if present (optional, for padded sequences)
        attention_mask = batch.get("attention_mask", None)

        # Forward pass (causal mask is created internally in GPTModel)
        logits = self(input_ids, attention_mask=attention_mask)

        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Calculate loss
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Log loss
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Calculate and log perplexity (exp is costly, but validation is infrequent so OK)
        perplexity = torch.exp(torch.clamp(loss, max=20.0))
        self.log("val/perplexity", perplexity, prog_bar=True, sync_dist=True)

        return loss
    # {{/docs-fragment lightning-module-training}}

    # {{docs-fragment lightning-module-optimizer}}
    def configure_optimizers(self):
        # Separate parameters into weight decay and no weight decay groups
        decay_params = []
        no_decay_params = []

        for param in self.model.parameters():
            if param.requires_grad:
                # 1D parameters (biases, LayerNorm) don't get weight decay
                # 2D+ parameters (weight matrices) get weight decay
                if param.ndim == 1:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Learning rate scheduler: warmup + cosine decay
        # Warmup: linear increase from 0 to 1.0 over warmup_steps
        # Decay: cosine decay from 1.0 to 0.0 over remaining steps
        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.hparams.warmup_steps))

            # Cosine decay after warmup
            progress = (current_step - self.hparams.warmup_steps) / max(
                1, self.hparams.max_steps - self.hparams.warmup_steps
            )
            # Cosine annealing from 1.0 to 0.0 (returns float, not tensor)
            return 0.5 * (1.0 + math.cos(progress * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    # {{/docs-fragment lightning-module-optimizer}}


# {{docs-fragment checkpoint-callback}}
class S3CheckpointCallback(L.Callback):
    """
    Periodically upload checkpoints to S3 for durability and resumption.

    This ensures checkpoints are safely stored in remote storage even if
    the training job is interrupted or the instance fails.
    """

    def __init__(self, checkpoint_dir: Path, upload_every_n_steps: int):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.upload_every_n_steps = upload_every_n_steps
        self.last_uploaded_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Upload checkpoint to S3 every N steps."""
        if trainer.global_rank != 0:
            return  # Only upload from rank 0

        current_step = trainer.global_step

        # Upload every N steps (aligns with ModelCheckpoint's every_n_train_steps)
        if (
            current_step % self.upload_every_n_steps == 0
            and current_step > self.last_uploaded_step
            and current_step > 0
        ):
            try:
                # Find the most recent checkpoint file
                checkpoint_files = list(self.checkpoint_dir.glob("*.ckpt"))
                if not checkpoint_files:
                    print("No checkpoint files found to upload")
                    return

                # Get the latest checkpoint (by modification time)
                latest_checkpoint = max(
                    checkpoint_files, key=lambda p: p.stat().st_mtime
                )

                # Upload the checkpoint file directly to S3 using File.from_local_sync
                checkpoint_file = File.from_local_sync(str(latest_checkpoint))
                print(f"Checkpoint uploaded to S3 at: {checkpoint_file.path}")

                self.last_uploaded_step = current_step
            except Exception as e:
                print(f"Warning: Failed to upload checkpoint to S3: {e}")
# {{/docs-fragment checkpoint-callback}}


# {{docs-fragment reporting-callback}}
class FlyteReportingCallback(L.Callback):
    """Custom Lightning callback to report training metrics to Flyte Report."""

    def __init__(self, report_every_n_steps: int = 100):
        super().__init__()
        self.report_every_n_steps = report_every_n_steps
        self.metrics_history = {
            "step": [],
            "train_loss": [],
            "learning_rate": [],
            "val_loss": [],
            "val_perplexity": [],
        }
        self.initialized_report = False
        self.last_logged_step = -1

    def on_train_start(self, trainer, pl_module):
        """Initialize the live dashboard on training start."""
        if trainer.global_rank == 0 and not self.initialized_report:
            self._initialize_report()
            self.initialized_report = True
    # {{/docs-fragment reporting-callback}}

    # {{docs-fragment reporting-callback-js}}
    def _initialize_report(self):
        flyte.report.log(
            """
        <h1>🚀 LLM Training Progress</h1>
        <div id="training-metrics" style="max-width:1000px;">
            <canvas id="lossChart" width="900" height="420" style="border:1px solid #eee;border-radius:8px;"></canvas>
            <canvas id="lrChart" width="900" height="240" style="margin-top:12px;border:1px solid #eee;border-radius:8px;"></canvas>
            <div id="stats" style="margin-top:12px;font-family: Arial, sans-serif;"></div>
        </div>

        <script>
        (function () {
            const lossCanvas = document.getElementById('lossChart');
            const lossCtx = lossCanvas.getContext('2d');
            const lrCanvas = document.getElementById('lrChart');
            const lrCtx = lrCanvas.getContext('2d');

            // High-DPI support
            function setupCanvas(canvas, ctx) {
                const dpr = window.devicePixelRatio || 1;
                const styleW = canvas.width;
                const styleH = canvas.height;
                canvas.style.width = styleW + 'px';
                canvas.style.height = styleH + 'px';
                canvas.width = Math.floor(styleW * dpr);
                canvas.height = Math.floor(styleH * dpr);
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            }
            setupCanvas(lossCanvas, lossCtx);
            setupCanvas(lrCanvas, lrCtx);

            // Data containers
            let steps = [];
            let trainLosses = [];
            let valLosses = [];
            let valSteps = [];  // Track which steps had validation
            let learningRates = [];
            let valPerplexities = [];

            // Layout / padding within canvas where chart lives
            const margin = { left: 60, right: 60, top: 40, bottom: 60 };

            // Utility: compute combined min/max with small padding
            function minMaxWithPadding(arr, padFraction=0.06) {
                if (!arr || arr.length === 0) return [0, 1];
                const min = Math.min(...arr);
                const max = Math.max(...arr);
                if (min === max) {
                    // add artificial range
                    const delta = Math.abs(min) * 0.05 || 1;
                    return [min - delta, max + delta];
                }
                const pad = (max - min) * padFraction;
                return [min - pad, max + pad];
            }

            // Draw axis gridlines, ticks and labels
            function drawAxes(ctx, width, height, xTicks, yTicks, xLabel, yLabel) {
                ctx.save();
                ctx.strokeStyle = '#dddddd';
                ctx.lineWidth = 1;

                // vertical gridlines (x ticks)
                xTicks.forEach(t => {
                    ctx.beginPath();
                    ctx.moveTo(t.x, margin.top);
                    ctx.lineTo(t.x, height - margin.bottom);
                    ctx.stroke();
                });

                // horizontal gridlines (y ticks)
                yTicks.forEach(t => {
                    ctx.beginPath();
                    ctx.moveTo(margin.left, t.y);
                    ctx.lineTo(width - margin.right, t.y);
                    ctx.stroke();
                });

                // axis lines
                ctx.strokeStyle = '#666';
                ctx.lineWidth = 1.2;
                // y axis
                ctx.beginPath();
                ctx.moveTo(margin.left, margin.top);
                ctx.lineTo(margin.left, height - margin.bottom);
                ctx.lineTo(width - margin.right, height - margin.bottom);
                ctx.stroke();

                // ticks labels
                ctx.fillStyle = '#222';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                xTicks.forEach(t => {
                    ctx.fillText(t.label, t.x, height - margin.bottom + 18);
                });

                ctx.textAlign = 'right';
                yTicks.forEach(t => {
                    ctx.fillText(t.label, margin.left - 8, t.y + 4);
                });

                // axis labels
                ctx.save();
                ctx.textAlign = 'center';
                ctx.font = '13px Arial';
                ctx.fillText(xLabel, (margin.left + (width - margin.right)) / 2, height - 12);
                ctx.restore();

                ctx.save();
                ctx.translate(14, (margin.top + (height - margin.bottom)) / 2);
                ctx.rotate(-Math.PI / 2);
                ctx.textAlign = 'center';
                ctx.fillText(yLabel, 0, 0);
                ctx.restore();

                ctx.restore();
            }

            // map data value to canvas y coordinate
            function yFor(val, yMin, yMax, canvasHeight) {
                const plotHeight = canvasHeight - margin.top - margin.bottom;
                const norm = (val - yMin) / (yMax - yMin);
                return margin.top + (1 - norm) * plotHeight;
            }

            // map index (0..N-1) or step value to x coordinate
            function xForIndex(index, total, canvasWidth) {
                const plotWidth = canvasWidth - margin.left - margin.right;
                if (total <= 1) return margin.left + plotWidth / 2;
                return margin.left + (index / (total - 1)) * plotWidth;
            }

            // create ticks (for readability we cap tick count)
            function makeTicksForY(yMin, yMax, canvasHeight, maxTicks=5) {
                const ticks = [];
                const span = yMax - yMin;
                const rawStep = span / maxTicks;
                // nice step rounding
                const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
                const niceStep = Math.ceil(rawStep / power) * power;
                const start = Math.floor(yMin / niceStep) * niceStep;
                for (let v = start; v <= yMax + 1e-12; v += niceStep) {
                    const y = yFor(v, yMin, yMax, canvasHeight);
                    ticks.push({ value: v, y: y, label: (Math.abs(v) >= 1000 ? v.toLocaleString() : Number(v.toPrecision(6)).toString()) });
                }
                return ticks;
            }

            function makeXTicksFromSteps(stepsArray, canvasWidth, maxTicks=8) {
                const ticks = [];
                const total = stepsArray.length;
                if (total === 0) return ticks;
                const step = Math.max(1, Math.ceil(total / maxTicks));
                for (let i = 0; i < total; i += step) {
                    ticks.push({ index: i, x: xForIndex(i, total, canvasWidth), label: String(stepsArray[i]) });
                }
                // ensure last step is included
                if (ticks.length === 0 || ticks[ticks.length-1].index !== total-1) {
                    const i = total-1;
                    ticks.push({ index: i, x: xForIndex(i, total, canvasWidth), label: String(stepsArray[i]) });
                }
                return ticks;
            }

            // Draw loss chart (train and val)
            function drawLossChart() {
                const ctx = lossCtx;
                const canvasWidth = lossCanvas.width / (window.devicePixelRatio || 1);
                const canvasHeight = lossCanvas.height / (window.devicePixelRatio || 1);

                // clear
                ctx.clearRect(0, 0, canvasWidth, canvasHeight);

                // combined loss range (train + val)
                const combined = [];
                if (trainLosses.length) combined.push(...trainLosses);
                if (valLosses.length) combined.push(...valLosses);
                const [yMin, yMax] = minMaxWithPadding(combined.length ? combined : [0,1], 0.06);

                // ticks
                const yTicks = makeTicksForY(yMin, yMax, canvasHeight, 5);
                const xTicks = makeXTicksFromSteps(steps, canvasWidth, 8);

                // axes + grid
                drawAxes(ctx, canvasWidth, canvasHeight, xTicks, yTicks, 'Step', 'Loss');

                // clip to plotting area so lines don't draw outside axes
                ctx.save();
                ctx.beginPath();
                ctx.rect(margin.left - 1, margin.top - 1, canvasWidth - margin.left - margin.right + 2, canvasHeight - margin.top - margin.bottom + 2);
                ctx.clip();

                // draw train loss
                if (trainLosses.length > 0) {
                    ctx.beginPath();
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = '#3498db';
                    for (let i = 0; i < trainLosses.length; i++) {
                        const x = xForIndex(i, steps.length, canvasWidth);
                        const y = yFor(trainLosses[i], yMin, yMax, canvasHeight);
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }
                    ctx.stroke();

                    // optional: small point markers
                    ctx.fillStyle = '#3498db';
                    for (let i = 0; i < trainLosses.length; i+= Math.max(1, Math.floor(trainLosses.length/50))) {
                        const x = xForIndex(i, steps.length, canvasWidth);
                        const y = yFor(trainLosses[i], yMin, yMax, canvasHeight);
                        ctx.beginPath();
                        ctx.arc(x, y, 2, 0, Math.PI*2);
                        ctx.fill();
                    }
                }

                // draw val loss at the actual steps where validation occurred
                if (valLosses.length > 0 && valSteps.length > 0) {
                    ctx.beginPath();
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = '#e74c3c';

                    for (let i = 0; i < valLosses.length; i++) {
                        const valStep = valSteps[i];
                        // Find x position based on where this step is in the training steps array
                        const stepIdx = steps.indexOf(valStep);
                        if (stepIdx >= 0) {
                            const x = xForIndex(stepIdx, steps.length, canvasWidth);
                            const y = yFor(valLosses[i], yMin, yMax, canvasHeight);
                            if (i === 0) ctx.moveTo(x, y);
                            else ctx.lineTo(x, y);
                        }
                    }
                    ctx.stroke();

                    // Draw markers at validation points
                    ctx.fillStyle = '#e74c3c';
                    for (let i = 0; i < valLosses.length; i++) {
                        const valStep = valSteps[i];
                        const stepIdx = steps.indexOf(valStep);
                        if (stepIdx >= 0) {
                            const x = xForIndex(stepIdx, steps.length, canvasWidth);
                            const y = yFor(valLosses[i], yMin, yMax, canvasHeight);
                            ctx.beginPath();
                            ctx.arc(x, y, 3.5, 0, Math.PI*2);
                            ctx.fill();
                        }
                    }
                }

                ctx.restore(); // remove clipping

                // legend
                ctx.fillStyle = '#3498db';
                ctx.fillRect(canvasWidth - margin.right - 140, margin.top - 28, 14, 10);
                ctx.fillStyle = '#222';
                ctx.font = '12px Arial';
                ctx.textAlign = 'left';
                ctx.fillText('Train Loss', canvasWidth - margin.right - 120, margin.top - 19);

                ctx.fillStyle = '#e74c3c';
                ctx.fillRect(canvasWidth - margin.right - 140, margin.top - 8, 14, 10);
                ctx.fillStyle = '#222';
                ctx.fillText('Val Loss', canvasWidth - margin.right - 120, margin.top + 1);
            }

            // Draw LR chart
            function drawLRChart() {
                const ctx = lrCtx;
                const canvasWidth = lrCanvas.width / (window.devicePixelRatio || 1);
                const canvasHeight = lrCanvas.height / (window.devicePixelRatio || 1);
                const lrMargin = { ...margin, left: 90 };

                ctx.clearRect(0, 0, canvasWidth, canvasHeight);

                if (learningRates.length === 0) {
                    // just draw axes and a message
                    const xTicks = makeXTicksFromSteps(steps, canvasWidth, 6);
                    const yTicks = makeTicksForY([0,1], 0.06, canvasHeight, 3);
                    drawAxes(ctx, canvasWidth, canvasHeight, xTicks, yTicks, 'Step', 'Learning Rate');
                    ctx.fillStyle = '#666';
                    ctx.font = '12px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText('Learning rate data not yet available', canvasWidth/2, canvasHeight/2);
                    return;
                }

                const [yMin, yMax] = minMaxWithPadding(learningRates, 0.08);
                const yTicks = makeTicksForY(yMin, yMax, canvasHeight, 4);
                const xTicks = makeXTicksFromSteps(steps, canvasWidth, 8);

                drawAxes(ctx, canvasWidth, canvasHeight, xTicks, yTicks, 'Step', 'Learning Rate');

                ctx.save();
                ctx.beginPath();
                ctx.rect(lrMargin.left - 1, lrMargin.top - 1, canvasWidth - lrMargin.left - lrMargin.right + 2, canvasHeight - lrMargin.top - lrMargin.bottom + 2);
                ctx.clip();

                ctx.beginPath();
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#2ecc71';
                for (let i = 0; i < learningRates.length; i++) {
                    const x = xForIndex(i, steps.length, canvasWidth);
                    const y = yFor(learningRates[i], yMin, yMax, canvasHeight);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();

                ctx.restore();

                ctx.fillStyle = '#2ecc71';
                ctx.fillRect(canvasWidth - lrMargin.right - 140, lrMargin.top - 18, 14, 10);
                ctx.fillStyle = '#222';
                ctx.font = '12px Arial';
                ctx.textAlign = 'left';
                ctx.fillText('Learning Rate', canvasWidth - lrMargin.right - 120, lrMargin.top - 8);
            }

            // API used by training loop to push metrics
            window.updateMetrics = function(step, trainLoss, lr, valLoss, valPpl) {
                // Handle training metrics
                const isNewStep = steps.length === 0 || step > steps[steps.length - 1];
                const isSameStep = steps.length > 0 && step === steps[steps.length - 1];

                if (isNewStep) {
                    // New step: add to training arrays
                    steps.push(step);
                    if (trainLoss !== null && trainLoss !== undefined) {
                        trainLosses.push(Number(trainLoss));
                    }
                    if (lr !== null && lr !== undefined) {
                        learningRates.push(Number(lr));
                    }
                } else if (isSameStep) {
                    // Same step: overwrite last training values
                    if (trainLoss !== null && trainLoss !== undefined && trainLosses.length > 0) {
                        trainLosses[trainLosses.length - 1] = Number(trainLoss);
                    }
                    if (lr !== null && lr !== undefined && learningRates.length > 0) {
                        learningRates[learningRates.length - 1] = Number(lr);
                    }
                } else {
                    // Out-of-order step: append anyway
                    steps.push(step);
                    if (trainLoss !== null && trainLoss !== undefined) {
                        trainLosses.push(Number(trainLoss));
                    }
                    if (lr !== null && lr !== undefined) {
                        learningRates.push(Number(lr));
                    }
                }

                // Handle validation metrics separately (only when provided)
                if (valLoss !== null && valLoss !== undefined) {
                    valSteps.push(step);
                    valLosses.push(Number(valLoss));
                    if (valPpl !== null && valPpl !== undefined) {
                        valPerplexities.push(Number(valPpl));
                    }
                }

                // draw
                drawLossChart();
                drawLRChart();

                // Debug logging
                console.log(`[updateMetrics] step=${step}, trainLoss=${trainLoss}, arrays: steps=${steps.length}, losses=${trainLosses.length}`);
                if (trainLosses.length > 0) {
                    console.log(`[updateMetrics] Loss range: ${Math.min(...trainLosses).toFixed(2)} - ${Math.max(...trainLosses).toFixed(2)}`);
                }

                // pretty stats panel
                document.getElementById('stats').innerHTML = `
                    <h3 style="lrMargin:0 0 6px 0;font-family:Arial">Current Metrics (Step ${step})</h3>
                    <div style="display:flex;gap:24px;flex-wrap:wrap;font-family:Arial">
                    <div><strong>Train Loss:</strong> ${trainLoss != null ? Number(trainLoss).toFixed(4) : 'N/A'}</div>
                    <div><strong>Learning Rate:</strong> ${lr != null ? Number(lr).toExponential(2) : 'N/A'}</div>
                    <div>${valLoss != null ? `<strong>Val Loss:</strong> ${Number(valLoss).toFixed(4)}` : ''}</div>
                    <div>${valPpl != null ? `<strong>Val Perplexity:</strong> ${Number(valPpl).toFixed(2)}` : ''}</div>
                    </div>
                `;
            };
        })();
        </script>
        """,
            do_flush=True,
        )
    # {{/docs-fragment reporting-callback-js}}

    # {{docs-fragment reporting-callback-update}}
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log training metrics every N steps."""
        if trainer.global_rank != 0:
            return

        current_step = trainer.global_step

        # Only log each step once and only at the reporting interval
        if (
            current_step % self.report_every_n_steps == 0
            and current_step != self.last_logged_step
        ):
            # Debug: Print available metrics (first time only)
            if not hasattr(self, "_printed_metrics"):
                print(
                    f"\n[FlyteReport] Available metrics: {list(trainer.callback_metrics.keys())}"
                )
                self._printed_metrics = True

            # Try multiple possible loss keys
            train_loss = (
                trainer.callback_metrics.get("train/loss_step", None)
                or trainer.callback_metrics.get("train/loss", None)
                or trainer.callback_metrics.get("loss", None)
            )

            # Try multiple possible LR keys
            learning_rate = (
                trainer.callback_metrics.get("lr-AdamW/pg1", None)
                or trainer.callback_metrics.get("lr-AdamW", None)
                or trainer.callback_metrics.get("train/lr", None)
                or trainer.callback_metrics.get("lr", None)
            )

            if train_loss is not None:
                self.metrics_history["step"].append(current_step)
                self.metrics_history["train_loss"].append(float(train_loss))
                if learning_rate is not None:
                    self.metrics_history["learning_rate"].append(float(learning_rate))

                self._update_report()
                self.last_logged_step = current_step  # Mark this step as logged
                print(
                    f"[FlyteReport] Updated report at step {current_step}: loss={train_loss:.4f}, lr={learning_rate}"
                )
            else:
                print(
                    f"[FlyteReport] Warning: No train loss found at step {current_step}, available: {list(trainer.callback_metrics.keys())}"
                )

    def on_validation_end(self, trainer, pl_module):
        """Log validation metrics."""
        if trainer.global_rank != 0:
            return

        val_loss = trainer.callback_metrics.get("val/loss", None)
        val_perplexity = trainer.callback_metrics.get("val/perplexity", None)

        if val_loss is not None:
            self.metrics_history["val_loss"].append(float(val_loss))
            if val_perplexity is not None:
                self.metrics_history["val_perplexity"].append(float(val_perplexity))

            self._update_report_validation(
                trainer.global_step, val_loss, val_perplexity
            )

    def _update_report(self):
        current_step = (
            self.metrics_history["step"][-1] if self.metrics_history["step"] else 0
        )
        train_loss = (
            self.metrics_history["train_loss"][-1]
            if self.metrics_history["train_loss"]
            else None
        )
        lr = (
            self.metrics_history["learning_rate"][-1]
            if self.metrics_history["learning_rate"]
            else None
        )

        train_loss_js = train_loss if train_loss is not None else "null"
        lr_js = lr if lr is not None else "null"

        flyte.report.log(
            f"""
<script>
(function() {{
    console.log('[FlyteReport] Training update for step {current_step}');
    if (typeof updateMetrics === 'function') {{
        updateMetrics({current_step}, {train_loss_js}, {lr_js}, null, null);
    }} else {{
        console.error('[FlyteReport] ERROR: updateMetrics function not found!');
    }}
}})();
</script>
""",
            do_flush=True,
        )

    def _update_report_validation(self, step, val_loss, val_perplexity):
        val_loss_js = val_loss if val_loss is not None else "null"
        val_ppl_js = val_perplexity if val_perplexity is not None else "null"

        flyte.report.log(
            f"""
<script>
(function() {{
    console.log('[FlyteReport] Validation update for step {step}');
    if (typeof updateMetrics === 'function') {{
        updateMetrics({step}, null, null, {val_loss_js}, {val_ppl_js});
    }} else {{
        console.error('[FlyteReport] ERROR: updateMetrics function not found!');
    }}
}})();
</script>
""",
            do_flush=True,
        )

    def state_dict(self):
        """Save metrics history to checkpoint for resumption."""
        return {"metrics_history": self.metrics_history}

    def load_state_dict(self, state_dict):
        """Restore metrics history from checkpoint."""
        self.metrics_history = state_dict.get("metrics_history", self.metrics_history)
        print(
            f"Restored metrics history with {len(self.metrics_history['step'])} steps"
        )
    # {{/docs-fragment reporting-callback-update}}


# {{docs-fragment data-loading-task}}
@data_loading_env.task
async def load_and_prepare_streaming_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    max_length: int,
    train_split: str,
    val_split: Optional[str],
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    shard_size_mb: int,
) -> Dir:
    """Tokenize dataset and convert to MDS format for streaming."""
    from datasets import load_dataset
    from streaming import MDSWriter
    from transformers import GPT2TokenizerFast

    output_dir = Path("/tmp/streaming_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # MDS schema: what each sample contains
    columns = {
        "input_ids": "ndarray:int32",
        "labels": "ndarray:int32",
    }
    # {{/docs-fragment data-loading-task}}

    # {{docs-fragment data-loading-task-process}}
    def process_and_write_split(split_name: str, max_samples: Optional[int]):
        """Process a dataset split and write to MDS format."""
        # Load dataset
        try:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split_name,
                streaming=True,  # Use streaming for memory efficiency
            )
        except Exception as e:
            print(f"Could not load {split_name} split: {e}")
            return None

        # Limit samples if requested
        if max_samples:
            dataset = dataset.take(max_samples)

        # Create output directory for this split
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MDS writer
        shard_size_bytes = shard_size_mb * 1024 * 1024
        writer = MDSWriter(
            out=str(split_dir),
            columns=columns,
            compression=None,  # No compression for faster loading
            size_limit=shard_size_bytes,
        )

        # Process and write samples
        num_samples = 0
        num_tokens = 0
        token_buffer = []

        for example in dataset:
            # Tokenize the text
            text = example.get("text", "")
            if not text:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)
            token_buffer.append(tokenizer.eos_token_id)

            # Write samples when we have enough tokens
            while len(token_buffer) >= max_length:
                # Create a fixed-length sample (no padding)
                # Documents are concatenated to create exactly max_length tokens
                sample_tokens = token_buffer[:max_length]
                token_buffer = token_buffer[max_length:]

                # Convert to numpy array
                sample_tokens_array = np.array(sample_tokens, dtype=np.int32)

                # Write to MDS
                # For causal LM: labels[i] = input_ids[i] (same token)
                # After shifting in training_step:
                #   predict input_ids[i+1] from input_ids[:i+1]
                # Note: No padding, so no need for -100 labels or attention_mask
                writer.write(
                    {
                        "input_ids": sample_tokens_array,
                        "labels": sample_tokens_array,  # labels[i] = input_ids[i]
                    }
                )

                num_samples += 1
                num_tokens += max_length

                if num_samples % 10000 == 0:
                    print(
                        f"  Processed {num_samples:,} samples ({num_tokens:,} tokens)"
                    )

        # Finish writing
        writer.finish()

        print(f"\n{split_name} split complete:")
        print(f"  Total samples: {num_samples:,}")
        print(f"  Total tokens: {num_tokens:,}")
        print(f"  Output: {split_dir}")

        return split_dir

    # Process training split
    process_and_write_split(train_split, max_train_samples)

    # Process validation split if specified
    if val_split:
        process_and_write_split(val_split, max_val_samples)

    return await Dir.from_local(str(output_dir))
    # {{/docs-fragment data-loading-task-process}}


def mds_collate_fn(batch):
    """Custom collate function to handle read-only numpy arrays from MDS."""
    collated = {}
    for key in batch[0].keys():
        arrays = [item[key] for item in batch]
        # Copy read-only arrays to make them writable
        arrays = [
            np.array(arr, copy=True) if not arr.flags.writeable else arr
            for arr in arrays
        ]
        collated[key] = torch.from_numpy(np.stack(arrays))
    return collated


# {{docs-fragment training-task-signature}}
@distributed_llm_training_env.task(report=True)
def train_distributed_llm(
    prepared_dataset: Dir,
    resume_checkpoint: Optional[Dir],
    vocab_size: int,
    n_positions: int,
    n_embd: int,
    n_layer: int,
    n_head: int,
    batch_size: int,
    num_workers: int,
    max_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    use_fsdp: bool,
    checkpoint_upload_steps: int,
    checkpoint_every_n_steps: int,
    report_every_n_steps: int,
    val_check_interval: int,
    grad_accumulation_steps: int = 1,
) -> Optional[Dir]:
    # ... setup code ...
    # {{/docs-fragment training-task-signature}}

    import streaming
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.strategies import FSDPStrategy
    from streaming import StreamingDataset
    from torch.utils.data import DataLoader

    logging.getLogger("torch.distributed.fsdp").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed.fsdp._common_utils").setLevel(logging.ERROR)

    remote_path = prepared_dataset.path
    print(f"Remote dataset path: {remote_path}")

    os.environ["S3_ENDPOINT_URL"] = os.environ.get("FLYTE_AWS_ENDPOINT")
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("FLYTE_AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("FLYTE_AWS_SECRET_ACCESS_KEY")

    ckpt_path = None
    if resume_checkpoint is not None:
        ckpt_local_dir = Path("/tmp/resume_checkpoint")
        ckpt_local_dir.mkdir(parents=True, exist_ok=True)
        resume_checkpoint.download(local_path=str(ckpt_local_dir))
        ckpt_files = list(ckpt_local_dir.glob("**/*.ckpt"))
        if ckpt_files:
            ckpt_path = str(ckpt_files[0])
            print(f"Found checkpoint: {ckpt_path}")

    output_dir = "/tmp/llm_training_output"
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    local_cache = Path("/tmp/streaming_cache")
    local_cache.mkdir(parents=True, exist_ok=True)

    streaming.base.util.clean_stale_shared_memory()

    # {{docs-fragment training-task-streaming}}
    # StreamingDataset streams shards from remote storage on-demand
    # It auto-detects torch.distributed and shards data across GPUs
    # This will stream shards from the remote Flyte storage on-demand
    # StreamingDataset automatically detects torch.distributed context
    # and shards data across GPUs - each rank gets different data automatically
    train_dataset = StreamingDataset(
        remote=f"{remote_path}/train",  # Remote MDS shard location
        local=str(local_cache / "train"),  # Local cache for downloaded shards
        shuffle=True,  # Shuffle samples
        shuffle_algo="naive",  # Shuffling algorithm
        batch_size=batch_size,  # Used for shuffle buffer sizing
    )

    # Create validation StreamingDataset if it exists
    val_dataset = None
    try:
        val_dataset = StreamingDataset(
            remote=f"{remote_path}/validation",
            local=str(local_cache / "validation"),
            shuffle=False,  # No shuffling for validation
            batch_size=batch_size,
        )
        print(
            f"Validation dataset initialized with streaming from: {remote_path}/validation"
        )
    except Exception as e:
        print(f"No validation dataset found: {e}")

    # Create data loaders
    # StreamingDataset handles distributed sampling internally by detecting
    # torch.distributed.get_rank() and torch.distributed.get_world_size()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,  # Drop incomplete batches for distributed training
        collate_fn=mds_collate_fn,  # Handle read-only arrays
    )

    # Create validation loader if validation dataset exists
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            collate_fn=mds_collate_fn,
        )
    # {{/docs-fragment training-task-streaming}}

    # Initialize model
    model = GPTPreTrainingModule(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
    )

    # Configure callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="gpt-step-{step:06d}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_train_steps=checkpoint_every_n_steps,
        ),
        LearningRateMonitor(logging_interval="step"),
        FlyteReportingCallback(report_every_n_steps=report_every_n_steps),
        S3CheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            upload_every_n_steps=checkpoint_upload_steps,
        ),
    ]

    # {{docs-fragment training-task-fsdp}}
    # Configure distributed strategy
    if use_fsdp:
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        strategy = FSDPStrategy(
            auto_wrap_policy=ModuleWrapPolicy([GPTBlock]),
            activation_checkpointing_policy=None,
            cpu_offload=False,  # H200 has 141GB - no CPU offload needed
            state_dict_type="full",
            sharding_strategy="FULL_SHARD",
            process_group_backend="nccl",
        )
    else:
        from lightning.pytorch.strategies import DDPStrategy

        strategy = DDPStrategy(process_group_backend="nccl")
    # {{/docs-fragment training-task-fsdp}}

    # {{docs-fragment training-task-trainer}}
    # Initialize trainer
    trainer = L.Trainer(
        strategy=strategy,
        accelerator="gpu",
        devices=DEVICES_PER_NODE,
        num_nodes=NUM_NODES,
        # Training configuration
        max_steps=max_steps,
        precision="bf16-mixed",  # BFloat16 for better numerical stability
        # Optimization
        gradient_clip_val=1.0,
        gradient_clip_algorithm=(
            "value" if use_fsdp else "norm"
        ),  # FSDP requires 'value', DDP can use 'norm'
        accumulate_grad_batches=grad_accumulation_steps,
        # Logging and checkpointing
        callbacks=callbacks,
        log_every_n_steps=report_every_n_steps,
        val_check_interval=val_check_interval,
        # Performance
        benchmark=True,
        deterministic=False,
        # Enable gradient checkpointing for memory efficiency
        enable_checkpointing=True,
        use_distributed_sampler=False,  # StreamingDataset handles distributed sampling
    )

    # Train the model (resume from checkpoint if provided)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Print final results
    if trainer.global_rank == 0:
        if val_loader is not None:
            print(
                f"Final validation loss: {trainer.callback_metrics.get('val/loss', 0.0):.4f}"
            )
            print(
                f"Final validation perplexity: {trainer.callback_metrics.get('val/perplexity', 0.0):.4f}"
            )
        print(f"Checkpoints saved to: {checkpoint_dir}")

        return Dir.from_local_sync(output_dir)

    return None
    # {{/docs-fragment training-task-trainer}}


# {{docs-fragment main-pipeline}}
@driver_env.task
async def distributed_llm_pipeline(
    model_size: str,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    max_length: int = 2048,
    max_train_samples: Optional[int] = 10000,
    max_val_samples: Optional[int] = 1000,
    max_steps: int = 100000,
    resume_checkpoint: Optional[Dir] = None,
    checkpoint_upload_steps: int = 1000,
    # Optional overrides (if None, uses model preset defaults)
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    use_fsdp: bool = True,
) -> Optional[Dir]:
    # Get model configuration
    model_config = get_model_config(model_size)

    # Use preset values if not overridden
    actual_batch_size = (
        batch_size if batch_size is not None else model_config["batch_size"]
    )
    actual_learning_rate = (
        learning_rate if learning_rate is not None else model_config["learning_rate"]
    )

    # Step 1: Load and prepare streaming dataset
    prepared_dataset = await load_and_prepare_streaming_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_length=max_length,
        train_split="train",
        val_split="validation",
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        shard_size_mb=64,  # 64MB shards
    )

    # Step 2: Run distributed training
    if resume_checkpoint is not None:
        print("\nStep 2: Resuming distributed training from checkpoint...")
    else:
        print("\nStep 2: Starting distributed training from scratch...")

    target_global_batch = 256
    world_size = NUM_NODES * DEVICES_PER_NODE
    effective_per_step = world_size * actual_batch_size
    grad_accumulation_steps = max(
        1, math.ceil(target_global_batch / max(1, effective_per_step))
    )

    checkpoint_dir = train_distributed_llm(
        prepared_dataset=prepared_dataset,
        resume_checkpoint=resume_checkpoint,
        vocab_size=VOCAB_SIZE,
        n_positions=N_POSITIONS,
        n_embd=model_config["n_embd"],
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        batch_size=actual_batch_size,
        num_workers=8,
        max_steps=max_steps,
        learning_rate=actual_learning_rate,
        weight_decay=0.1,
        warmup_steps=500,
        use_fsdp=use_fsdp,
        checkpoint_upload_steps=checkpoint_upload_steps,
        checkpoint_every_n_steps=model_config["checkpoint_every_n_steps"],
        report_every_n_steps=model_config["report_every_n_steps"],
        val_check_interval=model_config["val_check_interval"],
        grad_accumulation_steps=grad_accumulation_steps,
    )

    return checkpoint_dir
# {{/docs-fragment main-pipeline}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(
        distributed_llm_pipeline,
        model_size="30B",
        dataset_name="cerebras/SlimPajama-627B",
        dataset_config=None,
        max_length=2048,
        max_train_samples=5_000_000,
        max_val_samples=50_000,
        max_steps=15_000,
        use_fsdp=True,
        checkpoint_upload_steps=1000,
    )

    print(f"Run URL: {run.url}")
# {{/docs-fragment main}}
