"""
LLaMA Pre-training on AWS Trainium with Flyte and NeuronX Distributed (NxD)

Run:
* Build the trainium image with rootful Docker and push to ECR, then
  set TRAINIUM_IMAGE_URI as an environment variable. Example:
    docker build --platform linux/amd64 -f Dockerfile.trainium \
      -t <your-ecr-repo>:llama-trainium-training-v1 .
    docker push <your-ecr-repo>:llama-trainium-training-v1
    export TRAINIUM_IMAGE_URI=<your-ecr-repo>:llama-trainium-training-v1
* python llama.py
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import flyte
import torch
from flyte.io import Dir
from flyteplugins.jsonl import JsonlDir
from flyteplugins.pytorch.task import Elastic

BUCKET_NAME = "bert-trainium-aws"

DEFAULT_MODEL_CONFIG_DIR = "8B_config_llama3.1"
DEFAULT_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"

# --- trn1.32xlarge ---
NNODES = 1
NPROC_PER_NODE = 32  # 1 NeuronCore per worker
WORLD_SIZE = NNODES * NPROC_PER_NODE  # 32 total workers

# Tensor parallelism + data parallelism across 32 NeuronCores.
# DP_DEGREE = WORLD_SIZE / TP_DEGREE = 4 data-parallel replicas
TP_DEGREE = 8
KV_REPLICATOR = 1  # num_key_value_heads(8) / TP(8) = 1, no replication needed

TRAINIUM_RESOURCES = flyte.Resources(
    cpu="110",
    memory="400Gi",
    # trn1.32xlarge: 16 Trainium1 chips, 32 NeuronCores, 512 GiB device memory
    gpu="Trn1:16",
)

# --- trn2.48xlarge ---
# 16 Trainium2 chips, 128 NeuronCores (4 per worker), 1536 GiB device memory
#
# NNODES = 1
# NPROC_PER_NODE = 32  # 4 NeuronCores per worker
# WORLD_SIZE = NNODES * NPROC_PER_NODE  # 32 total workers
#
# TRAINIUM_RESOURCES = flyte.Resources(
#     cpu="100",
#     memory="800Gi",
#     gpu="Trn2:16",
# )

data_prep_env = flyte.TaskEnvironment(
    name="llama-trainium-data-prep",
    image=flyte.Image.from_debian_base(name="llama-trainium-data-prep")
    .with_apt_packages("git")
    .with_pip_packages(
        "git+https://github.com/flyteorg/flyte-sdk.git@53250d135c61f5447d43a294f0f8d373ad636764#subdirectory=plugins/pytorch",
    )
    .with_pip_packages(
        "datasets==4.4.1",
        "transformers==4.57.1",
        "tokenizers==0.22.1",
        "flyteplugins-jsonl==2.0.8",
    ),
    resources=flyte.Resources(cpu=3, memory="15Gi"),
    secrets=[flyte.Secret(key="samhita_hf_key", as_env_var="HF_TOKEN")],
    cache="auto",
)


trainium_env = flyte.TaskEnvironment(
    name="llama-trainium-training",
    # TODO: Fix the builder to support unpacking high-UID files so we can switch to a
    # from_base + with_pip_packages style.
    image=flyte.Image.from_base(image_uri=os.getenv("TRAINIUM_IMAGE_URI")),
    resources=TRAINIUM_RESOURCES,
    plugin_config=Elastic(
        nnodes=NNODES,
        nproc_per_node=NPROC_PER_NODE,
        neuron_parallel_compile=True,
    ),
    env_vars={
        "MALLOC_ARENA_MAX": "64",
        "NEURON_CC_FLAGS": "--model-type transformer --cache_dir /tmp/neuron_compile_cache",
        "NEURON_FUSE_SOFTMAX": "1",
        "NEURON_RT_STOCHASTIC_ROUNDING_EN": "0",
        "NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS": "3",
        "NEURON_COMPILE_CACHE_URL": "/tmp/neuron_compile_cache",
    },
    secrets=[flyte.Secret(key="samhita_hf_key", as_env_var="HF_TOKEN")],
    cache="auto",
)


pipeline_env = flyte.TaskEnvironment(
    name="llama-trainium-pipeline",
    image=flyte.Image.from_debian_base(name="llama-trainium-pipeline")
    .with_apt_packages("git")
    .with_pip_packages(
        "git+https://github.com/flyteorg/flyte-sdk.git@53250d135c61f5447d43a294f0f8d373ad636764#subdirectory=plugins/pytorch",
    )
    .with_pip_packages("flyteplugins-jsonl==2.0.8"),
    depends_on=[data_prep_env, trainium_env],
)


@dataclass
class DatasetConfig:
    """Configuration for FineWeb dataset loading and sampling"""

    shuffle: bool = True
    shuffle_seed: int = 42


@dataclass
class QuickTrainingConfig:
    """Training hyperparameters for LLaMA pre-training

    Default: QuickTest (for fast iteration)
    """

    # LLaMA-style hyperparameters
    learning_rate: float = 1.5e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    warmup_steps: int = 10
    max_steps: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048
    min_lr_ratio: float = 0.1

    # Logging and checkpointing
    logging_steps: int = 1
    save_steps: int = 10


@dataclass
class MediumTrainingConfig(QuickTrainingConfig):
    """Medium training profile

    trn1.32xlarge: batch_size=1 with gradient accumulation.
    Effective batch: 1 x 8 accum x 4 DP = 32 samples/step.
    """

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1

    warmup_steps: int = 100
    max_steps: int = 3000
    gradient_accumulation_steps: int = 8

    logging_steps: int = 5
    save_steps: int = 50


@dataclass
class ProductionTrainingConfig(QuickTrainingConfig):
    """Production training profile

    Global batch: 4 DP x 1 batch x 256 accum = 1024 samples/step
    At seq_length=4096: ~4.2M tokens/step (matches LLaMA paper recipe)
    """

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1

    warmup_steps: int = 2000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 256
    max_seq_length: int = 4096
    min_lr_ratio: float = 0.033

    logging_steps: int = 10
    save_steps: int = 500


####################
# Data Preparation #
####################
@data_prep_env.task
async def prepare_tokenized_dataset(
    fineweb_hf_path: str,
    num_samples: int,
    tokenizer_name: str,
    max_seq_length: int,
    records_per_shard: int = 5000,
) -> JsonlDir:
    """
    Prepare FineWeb dataset by tokenizing and saving as sharded JSONL.

    1. Downloads FineWeb-Edu from HuggingFace (streaming)
    2. Batch-tokenizes text for causal language modeling
    3. Writes to auto-sharded JsonlDir for distributed loading
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {num_samples:,} samples from FineWeb-Edu...")
    dataset = load_dataset(
        fineweb_hf_path,
        split="train",
        streaming=True,
    ).take(num_samples)

    out = JsonlDir.new_remote(f"fineweb-edu-seq{max_seq_length}/")

    TOKENIZE_BATCH_SIZE = 512
    total_written = 0

    print(
        f"Tokenizing with max_length={max_seq_length}, "
        f"shard_size={records_per_shard} records..."
    )

    async with out.writer(max_records_per_shard=records_per_shard) as writer:
        batch_texts = []
        for example in dataset:
            batch_texts.append(example["text"])

            if len(batch_texts) >= TOKENIZE_BATCH_SIZE:
                tokens = tokenizer(
                    batch_texts,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                )
                for i in range(len(batch_texts)):
                    await writer.write(
                        {
                            "input_ids": tokens["input_ids"][i],
                            "attention_mask": tokens["attention_mask"][i],
                        }
                    )
                total_written += len(batch_texts)
                if total_written % 50_000 == 0:
                    print(f"  Tokenized {total_written:,} / {num_samples:,} samples")
                batch_texts = []

        # Flush remaining
        if batch_texts:
            tokens = tokenizer(
                batch_texts,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
            )
            for i in range(len(batch_texts)):
                await writer.write(
                    {
                        "input_ids": tokens["input_ids"][i],
                        "attention_mask": tokens["attention_mask"][i],
                    }
                )
            total_written += len(batch_texts)

    print(f"Dataset prepared: {total_written:,} samples written to JsonlDir")
    return out


###########
# Dataset #
###########
class ShardedJsonlDataset(torch.utils.data.IterableDataset):
    """IterableDataset that reads rank-assigned JSONL shards for distributed training.

    Each rank reads only its assigned shards (round-robin assignment),
    so no data is downloaded or processed redundantly across workers.
    """

    def __init__(
        self,
        data_dir: str,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed

        all_shards = sorted(
            f for f in os.listdir(data_dir) if f.startswith("part-") and ".jsonl" in f
        )
        if not all_shards:
            raise ValueError(f"No JSONL shard files found in {data_dir}")

        self.shard_files = [
            os.path.join(data_dir, all_shards[i])
            for i in range(rank, len(all_shards), world_size)
        ]
        print(
            f"[Rank {rank}] Assigned {len(self.shard_files)}/{len(all_shards)} shards"
        )

    def __iter__(self):
        shard_order = list(self.shard_files)
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(shard_order)

        for shard_path in shard_order:
            with open(shard_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    yield {
                        "input_ids": torch.tensor(
                            record["input_ids"], dtype=torch.long
                        ),
                        "attention_mask": torch.tensor(
                            record["attention_mask"], dtype=torch.long
                        ),
                    }


####################################
# Flyte report dashboard (HTML/JS) #
####################################
REPORT_HTML = """
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    .llama-dashboard { font-family: 'Segoe UI', sans-serif; }
    .llama-dashboard h1 { color: #333; }
    .llama-dashboard .metric-card {
        background: white; border-radius: 8px; padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    .llama-dashboard .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }
    .llama-dashboard .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
    .llama-dashboard .plot { width: 100%; height: 400px; }
</style>
<div class="llama-dashboard">
    <h1>LLaMA Trainium Pre-training (TP=8, DP=4)</h1>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div class="metric-card">
            <div class="metric-value" id="current-step">0</div>
            <div class="metric-label">Current Step</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="current-loss">-</div>
            <div class="metric-label">Training Loss</div>
        </div>
    </div>
    <div class="metric-card">
        <h3>Training Loss</h3>
        <div id="loss-plot" class="plot"></div>
    </div>
</div>
<script>
    var lossTrace = {
        x: [], y: [], type: 'scatter', mode: 'lines+markers',
        name: 'Loss', line: {color: '#2196F3', width: 2}, marker: {size: 6}
    };
    Plotly.newPlot('loss-plot', [lossTrace], {
        xaxis: {title: 'Step'}, yaxis: {title: 'Loss'}, showlegend: false
    });
    function updateMetrics(step, loss) {
        document.getElementById('current-step').textContent = step;
        document.getElementById('current-loss').textContent = loss.toFixed(4);
        Plotly.extendTraces('loss-plot', {x: [[step]], y: [[loss]]}, [0]);
    }
</script>
"""


def _download_dir(
    remote_dir,
    local_path,
    *,
    attempts=5,
    base_delay=2.0,
    max_workers=64,
):
    """Robustly + concurrently download a remote Dir into ``local_path``.

    Per-file downloads are dispatched to a thread pool because large NxD
    checkpoints have hundreds of small xser shard files; serialised S3 GETs
    easily exceed the 120 s NeuronCore barrier timeout that xm.rendezvous
    enforces between collective ops.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    root = remote_dir.path.rstrip("/")
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            os.makedirs(local_path, exist_ok=True)
            tasks = []
            for f in remote_dir.walk_sync(recursive=True):
                rel = f.path[len(root) :].lstrip("/")
                dest = os.path.join(local_path, rel)
                os.makedirs(os.path.dirname(dest) or local_path, exist_ok=True)
                tasks.append((f, dest))
            if not tasks:
                raise FileNotFoundError(f"No files listed under {remote_dir.path}")

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(f.download_sync, dest) for f, dest in tasks]
                for fut in as_completed(futures):
                    fut.result()  # propagate per-file errors
            return local_path
        except FileNotFoundError as e:
            last_err = e
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"Dir download attempt {attempt}/{attempts} failed for "
                f"{remote_dir.path} ({e}); retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
    raise last_err


#################
# Training task #
#################
@trainium_env.task(report=True)
def train_llama_on_trainium(
    dataset_path: JsonlDir,
    training_config: QuickTrainingConfig,
    dataset_config: DatasetConfig,
    model_config_path: str,
    compilation_cache: Optional[Dir] = None,
    resume_from_checkpoint: Optional[Dir] = None,
) -> tuple[Optional[Dir], Optional[Dir]]:
    """
    Distributed LLaMA pre-training on AWS Trainium.

    Uses tensor parallelism (TP=8) + data parallelism (DP=4) via neuronx_distributed
    on trn1.32xlarge (32 NeuronCores).

    Flow:
        1. Setup: distributed init, file-limit bump, cache/data download
        2. Model: NxD config, model init, optional checkpoint resume
        3. Optimizer: AdamW_FP32OptimParams + ZeRO-1, cosine LR schedule
        4. Training loop: forward -> loss/grad_accum -> backward -> mark_step ->
           all_reduce -> optimizer.step -> scheduler.step
        5. Checkpointing: NxD sharded checkpoints uploaded to S3
    """
    import resource

    import neuronx_distributed as nxd
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_backend
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.utils.adamw_fp32_optim_params import AdamW_FP32OptimParams
    from torch.utils.data import DataLoader
    from training_utils import get_mixed_precision_config
    from transformers import LlamaConfig, set_seed

    set_seed(12349)
    extract_graphs_only = os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY") == "1"

    # 1. Distributed setup
    dist.init_process_group("xla")
    rank = dist.get_rank()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch_xla.device()

    # Raise open-file limit (needed for NxD checkpoint I/O)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65535 if hard == resource.RLIM_INFINITY else min(65535, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    # 2. Flyte report dashboard (rank 0 only)
    metrics_history = []
    if rank == 0:
        flyte.report.log(REPORT_HTML, do_flush=True)
        step_start_time = time.time()

    # 3. Download compilation cache & dataset
    cache_dir = os.getenv("NEURON_COMPILE_CACHE_URL", "/tmp/neuron_compile_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if local_rank == 0 and compilation_cache:
        print(f"[Rank {rank}] Downloading compilation cache to {cache_dir}...")
        _download_dir(compilation_cache, cache_dir)
        print(f"[Rank {rank}] Compilation cache restored")
    xm.rendezvous("cache_setup_complete")

    if local_rank == 0:
        _download_dir(dataset_path, "/tmp/flyte_dataset")
    xm.rendezvous("dataset_ready")

    local_data_dir = "/tmp/flyte_dataset"

    # 4. Pre-download the resume checkpoint (if any).
    start_step = 0
    metrics_history: list = []
    if resume_from_checkpoint and not extract_graphs_only:
        if local_rank == 0:
            print(f"[Rank {rank}] Downloading checkpoint to /tmp/flyte_checkpoint...")
            _download_dir(resume_from_checkpoint, "/tmp/flyte_checkpoint")
            print(f"[Rank {rank}] Checkpoint download complete")
        xm.rendezvous("checkpoint_download_complete")

        ckpt_dir = "/tmp/flyte_checkpoint"
        with open(f"{ckpt_dir}/training_state.json", "r") as f:
            training_state = json.load(f)
        start_step = training_state.get("step", 0)
        metrics_history = training_state.get("metrics_history", [])

        # Diagnostic: log what's actually on disk so a hang at load_checkpoint
        # (NeuronCore barrier timeout) is easy to attribute.
        if rank == 0:
            tag_dir = os.path.join(ckpt_dir, f"step_{start_step}")
            print(f"[Rank {rank}] Resume layout under {ckpt_dir}:")
            for root, _, files in os.walk(ckpt_dir):
                rel = os.path.relpath(root, ckpt_dir)
                print(f"  {rel}/  ({len(files)} files)")
            if not os.path.isdir(tag_dir):
                raise FileNotFoundError(
                    f"Expected checkpoint tag dir not found: {tag_dir}. "
                    f"The remote checkpoint likely predates the synchronous-save "
                    f"fix and contains only training_state.json. Resume from a "
                    f"newer checkpoint or rerun training to produce one."
                )

    # 5. NxD model initialization
    nxd_config = nxd.neuronx_distributed_config(
        tensor_parallel_size=TP_DEGREE,
        context_parallel_size=1,
        optimizer_config={
            "zero_one_enabled": True,
            "grad_clipping": True,
            "max_grad_norm": training_config.max_grad_norm,
        },
        activation_checkpoint_config="full",
        mixed_precision_config=get_mixed_precision_config(True),
    )

    def get_model():
        from modeling_llama_nxd import LlamaForCausalLM

        config = LlamaConfig.from_pretrained(model_config_path)
        config.use_cache = False
        config.dtype = torch.bfloat16
        config.use_flash_attention = True
        config.selective_checkpoint_enabled = True
        config.qkv_linear = True
        config.fuse_qkv = True
        config.transpose_nki_inputs = True
        config.kv_shared_group_size = KV_REPLICATOR
        config.max_position_embeddings = max(
            config.max_position_embeddings, training_config.max_seq_length
        )
        config.head_dim = config.hidden_size // config.num_attention_heads
        return LlamaForCausalLM(config)

    model = nxd.initialize_parallel_model(
        nxd_config, get_model, True
    )  # include_buffers

    # 6. Checkpoint resume (optional). The download already happened above; this
    # block only does the device-bound model parameter load.
    if resume_from_checkpoint and not extract_graphs_only:
        ckpt_dir = "/tmp/flyte_checkpoint"
        if rank == 0:
            print(f"[Rank {rank}] Calling nxd.load_checkpoint(model=...)")

        nxd.load_checkpoint(ckpt_dir, tag=f"step_{start_step}", model=model)

        if rank == 0:
            print(
                f"[Rank {rank}] Model load completed. "
                f"Resuming from step {start_step}"
            )

    # 7. Optimizer & scheduler
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n.lower() for nd in ("bias", "norm"))
            ],
            "weight_decay": training_config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n.lower() for nd in ("bias", "norm"))
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = nxd.initialize_parallel_optimizer(
        nxd_config,
        AdamW_FP32OptimParams,
        param_groups,
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        weight_decay=training_config.weight_decay,
    )
    optimizer.zero_grad()

    # Cosine decay with linear warmup
    def lr_lambda(step: int):
        if step < training_config.warmup_steps:
            return float(step) / float(max(1, training_config.warmup_steps))
        progress = float(step - training_config.warmup_steps) / float(
            max(1, training_config.max_steps - training_config.warmup_steps)
        )
        return max(
            training_config.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Restore optimizer/scheduler state if resuming
    if resume_from_checkpoint and not extract_graphs_only:
        if rank == 0:
            print(f"[Rank {rank}] Calling nxd.load_checkpoint(optimizer+scheduler=...)")

        nxd.load_checkpoint(
            "/tmp/flyte_checkpoint",
            tag=f"step_{start_step}",
            optimizer=optimizer,
            scheduler=scheduler,
        )

        if rank == 0:
            print(f"[Rank {rank}] Optimizer+scheduler load completed.")

    # 8. DataLoader
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_size()

    dataset = ShardedJsonlDataset(
        data_dir=local_data_dir,
        rank=dp_rank,
        world_size=dp_size,
        shuffle=dataset_config.shuffle,
        seed=dataset_config.shuffle_seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.per_device_train_batch_size,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    parallel_dataloader = pl.MpDeviceLoader(dataloader, device)

    # Restore historical metrics to the dashboard if resuming
    if rank == 0 and metrics_history:
        for m in metrics_history:
            flyte.report.log(
                f"""<script>updateMetrics({m['step']}, {m['loss']});</script>""",
                do_flush=False,
            )
        flyte.report.log("", do_flush=True)

    xm.rendezvous("ready_to_train")
    if rank == 0:
        print(
            f"Starting training (TP={TP_DEGREE}, DP={dp_size}, "
            f"max_steps={training_config.max_steps})"
        )

    # 9. Training loop
    model.train()
    global_step = start_step
    training_ustep = start_step * training_config.gradient_accumulation_steps
    total_loss = 0.0
    output_dir = "/tmp/llama_checkpoints"
    latest_remote_checkpoint = None
    latest_remote_cache_dir = None

    # During graph extraction, run enough steps to cover accumulation/save boundaries
    max_steps = 15 if extract_graphs_only else training_config.max_steps

    for batch in parallel_dataloader:
        if global_step >= max_steps:
            xm.mark_step()
            break

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Causal LM: labels = input_ids (model shifts internally), mask padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / training_config.gradient_accumulation_steps
        loss.backward()
        total_loss += loss.detach()
        training_ustep += 1

        if training_ustep % training_config.gradient_accumulation_steps != 0:
            continue

        # Accumulation boundary
        xm.mark_step()

        # All-reduce loss across DP replicas
        total_loss_reduced = xm.all_reduce(
            xm.REDUCE_SUM,
            total_loss,
            groups=parallel_state.get_data_parallel_replica_groups(),
            scale=1.0 / dp_size,
        )

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        global_step += 1

        # Logging (via step closure to avoid blocking tracing)
        if rank == 0 and global_step % training_config.logging_steps == 0:

            def _log(loss_val=total_loss_reduced, step=global_step):
                nonlocal step_start_time
                step_time = time.time() - step_start_time
                avg_loss = (loss_val / training_config.logging_steps).cpu().item()
                perplexity = math.exp(min(avg_loss, 20.0))
                lr = scheduler.get_last_lr()[0]
                samples_per_sec = (
                    training_config.logging_steps
                    * training_config.per_device_train_batch_size
                    * dp_size
                ) / step_time

                metrics_history.append(
                    {
                        "step": step,
                        "loss": avg_loss,
                        "perplexity": perplexity,
                        "throughput": samples_per_sec,
                    }
                )
                flyte.report.log(
                    f"""<script>updateMetrics({step}, {avg_loss});</script>""",
                    do_flush=True,
                )
                print(
                    f"Step {step}/{training_config.max_steps} | "
                    f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                    f"LR: {lr:.2e} | {samples_per_sec:.1f} samples/s"
                )
                step_start_time = time.time()

            xm.add_step_closure(_log)

        total_loss = 0.0

        # Checkpoint saving via NxD (TP-aware sharded checkpoints)
        if not extract_graphs_only and global_step % training_config.save_steps == 0:
            checkpoint_path = f"{output_dir}/checkpoint-{global_step}"
            os.makedirs(checkpoint_path, exist_ok=True)

            # All ranks must call this (TP-sharded collective).
            nxd.save_checkpoint(
                checkpoint_path,
                f"step_{global_step}",
                model,
                optimizer,
                scheduler,
                {"step": global_step},
                8,  # num_workers
                True,  # use_xser
                -1,  # num_kept_ckpts (-1 = keep all)
            )

            # Barrier so every rank's shards are on disk before rank 0 uploads.
            xm.rendezvous("checkpoint_saved")

            # Defer the JSON write + upload via a step closure so it runs after
            # this step's _log closure. _log fires at the next mark_step and
            # appends step N to metrics_history; this closure (added right
            # after) fires immediately after _log and snapshots the now-complete
            # metrics_history into training_state.json before uploading. Result:
            # the persisted JSON contains steps 1..N.
            if rank == 0:

                def _finalize_save(
                    ckpt_path=checkpoint_path,
                    step=global_step,
                ):
                    nonlocal latest_remote_checkpoint
                    with open(f"{ckpt_path}/training_state.json", "w") as f:
                        json.dump(
                            {
                                "step": step,
                                "metrics_history": metrics_history.copy(),
                            },
                            f,
                        )
                    s3_base = f"s3://{BUCKET_NAME}/{flyte.ctx().action.run_name}"
                    latest_remote_checkpoint = Dir.from_local_sync(
                        local_path=ckpt_path,
                        remote_destination=f"{s3_base}/checkpoints/step-{step}",
                    )
                    Dir.from_local_sync(
                        local_path=ckpt_path,
                        remote_destination=f"{s3_base}/checkpoints/latest",
                    )
                    print(f"Checkpoint uploaded for step {step}")

                xm.add_step_closure(_finalize_save)

    # 10. Post-training
    # Compilation cache upload: drain all device ops so any in-flight
    # compiles finalize and release their locks before we snapshot the cache.
    # This avoids capturing compile_lock / *.tmp entries that would deadlock
    # downstream runs on a phantom-lock NeuronCore barrier timeout.
    xm.wait_device_ops()
    xm.rendezvous("cache_quiescent")

    if rank == 0:
        s3_base = f"s3://{BUCKET_NAME}/{flyte.ctx().action.run_name}"
        cache_subdir = "post-compile" if extract_graphs_only else f"step-{global_step}"
        latest_remote_cache_dir = Dir.from_local_sync(
            local_path=cache_dir,
            remote_destination=f"{s3_base}/neuron-compile-cache/{cache_subdir}",
        )
        print(f"[Rank 0] Compilation cache uploaded to {latest_remote_cache_dir.path}")

    xm.rendezvous("training_complete")

    if rank == 0:
        print(f"Training completed at step {global_step}")

    del parallel_dataloader, dataloader, model, optimizer
    xm.wait_device_ops()
    dist.destroy_process_group()

    return latest_remote_checkpoint, latest_remote_cache_dir


############
# Pipeline #
############
@pipeline_env.task
async def llama_pretraining_pipeline(
    fineweb_path: str = "HuggingFaceFW/fineweb-edu",
    model_config_path: str = DEFAULT_MODEL_CONFIG_DIR,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    dataset_config: DatasetConfig = DatasetConfig(),
    training_config: QuickTrainingConfig = QuickTrainingConfig(),
    num_samples: Optional[int] = None,
    compilation_cache: Optional[Dir] = None,
    resume_from_checkpoint: Optional[Dir] = None,
) -> tuple[Optional[Dir], Optional[Dir]]:
    """
    LLaMA pre-training pipeline on AWS Trainium.

    Steps:
        1. Tokenize FineWeb-Edu into sharded JSONL
        2. Distributed training with NxD (TP=8, DP=4)
    """
    max_seq_length = training_config.max_seq_length
    dp_degree = WORLD_SIZE // TP_DEGREE

    print("=" * 80)
    print(f"Starting LLaMA pre-training on Trainium")
    print(f"  Model config: {model_config_path}")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  Parallelism: TP={TP_DEGREE}, DP={dp_degree}")
    print(f"  Sequence length: {max_seq_length}")
    print(f"  Max steps: {training_config.max_steps}")
    print("=" * 80)

    if num_samples is None:
        num_samples = (
            training_config.max_steps
            * training_config.gradient_accumulation_steps
            * training_config.per_device_train_batch_size
            * dp_degree
        )
        print(f"Auto-calculated num_samples: {num_samples:,}")

    # Step 1: Prepare tokenized dataset
    records_per_shard = max(100, num_samples // (dp_degree * 2))
    records_per_shard = min(records_per_shard, num_samples // max(1, dp_degree))
    dataset_path = await prepare_tokenized_dataset(
        fineweb_hf_path=fineweb_path,
        num_samples=num_samples,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
        records_per_shard=records_per_shard,
    )

    # Step 2: Distributed training on Trainium
    try:
        model_dir, cache_dir = train_llama_on_trainium(
            dataset_path=dataset_path,
            training_config=training_config,
            dataset_config=dataset_config,
            model_config_path=model_config_path,
            compilation_cache=compilation_cache,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        print(f"Training completed successfully!")
        return model_dir, cache_dir

    except flyte.errors.RuntimeUserError as e:
        s3_base = f"s3://{BUCKET_NAME}/{flyte.ctx().action.run_name}"
        latest_ckpt_path = f"{s3_base}/checkpoints/latest"
        print(f"TRAINING FAILED - recovering latest checkpoint: {latest_ckpt_path}")
        try:
            return Dir(path=latest_ckpt_path), None
        except Exception:
            raise e


if __name__ == "__main__":
    flyte.init_from_config()

    # Run name of the previous run that produced the checkpoint.
    RESUME_RUN_NAME = "<previous run name>"  # User TODO: set to the run that saved the checkpoint and cache
    STEP_TO_RESUME_FROM = 10  # User TODO: set to the step number of the checkpoint you want to resume from

    s3_base = f"s3://{BUCKET_NAME}/{RESUME_RUN_NAME}"

    run = flyte.with_runcontext(copy_style="all").run(
        llama_pretraining_pipeline,
        # resume_from_checkpoint=Dir.from_existing_remote(
        #     f"{s3_base}/checkpoints/{STEP_TO_RESUME_FROM}"
        # ),
        # compilation_cache=Dir.from_existing_remote(
        #     f"{s3_base}/neuron-compile-cache/{STEP_TO_RESUME_FROM}"
        # ),
    )
    print(f"Run URL: {run.url}")
