# Async Multi-Node RL Post-Training with GRPO on Union

*Part 2 of the GRPO tutorial. Part 1 —
[`rl_grpo_lora`](../rl_grpo_lora/) — builds the single-GPU loop; this tutorial turns its trainer
into a gang of pods on a `ClusteredTaskEnvironment`, upgrades the objective to real GRPO, pipelines
generation against training, and proves it on an 8B policy with a held-out benchmark.*

RL post-training with verifiable rewards (RLVR — the DeepSeek-R1 recipe) is where the
competitive frontier moved in 2025–26, and GRPO is the algorithm everyone runs. Outside the big
labs, multi-node GRPO is usually held together with shell scripts across several terminals, and a
dead GPU ruins an overnight run. The promise of this tutorial:

> **Frontier-lab-style RL post-training — a warm rollout fleet, a multi-node trainer, async
> off-policy pipelining, a held-out eval — declared in Python, run with one command, on your own
> cluster.**

The receipts, from the reference run (Qwen3-8B, GSM8K, 2 trainer pods × 4 L40S, 20 iterations):

| | |
|---|---|
| Held-out GSM8K accuracy (64 questions, greedy) | **60.9 % → 92.2 % peak (step 14) → 85.9 % final** |
| Steady-state iteration | **≈ 5 min, all of it generation** — the ≈ 2.3 min clustered train step (JobSet spin-up + 8B load + two epochs) is entirely hidden under the next batch's generation |
| vs. the same loop run sequentially | ≈ 7.3 min per iteration → **pipelining is worth ~⅓ of the wall-clock** |
| Held-out eval | never on the critical path (runs in the background, overlapping the next train step) |
| Actions | 1,323 (640 rollout calls, 640 reward calls, 20 JobSets, 21 evals), **0 failures, 0 restarts** |

---

## Why Union fits multi-node RL

- **Heterogeneous fleet in one run.** Rollouts run on a warm, autoscaling pool of single-GPU
  vLLM actors; the trainer bursts onto multi-GPU nodes only while it trains; reward and the
  driver are CPU tasks. Each stage is a right-sized `TaskEnvironment`; the driver is an `async`
  `for` loop.
- **The trainer is an environment declaration, not a rewrite.** Moving `train_step` from one GPU
  to *N pods × G GPUs* is a `ClusteredTaskEnvironment(replicas=…, nproc_per_node=…)` — one
  Kubernetes JobSet per call, torchrun bootstrapped, whole-gang restarts on any pod death.
- **The orchestrator's async driver *is* the async-RL controller** script-land setups have to
  hand-build: overlapping generation N+1 with training N is a `create_task` in the driver.
- **Weight sync is not a network protocol.** Trainer → generator handoff is a few-MB LoRA
  adapter through the object store, so the rollout fleet and the trainer are decoupled — no
  gang-scheduled two-cluster topology, no NCCL broadcast, no tight coupling.

## 1. What changes from Part 1

Part 1 already *is* post-training — a validated GRPO + LoRA loop — but small: a 0.6B model, eight
arithmetic questions, one GPU, a simplified objective, no eval. This tutorial adds four things:

1. **A multi-node trainer** on a `ClusteredTaskEnvironment` (`train_step_clustered`).
2. **The real GRPO objective.** Part 1's loss is `−Â · mean_t log π(o_t)` — REINFORCE with
   group-normalized advantages. That is fine *on-policy*, but the pipelined driver below trains
   on rollouts sampled by the *previous* adapter. This trainer uses the token-level
   clipped-importance-ratio surrogate with a KL penalty against the base model, which is what
   makes off-policy training sound. AdamW state persists across iterations (Part 1 rebuilt it
   every step).
3. **Async pipelining.** Generation of batch N+1 overlaps the clustered step on batch N;
   `pipelined=False` runs the identical loop sequentially for a clean comparison.
4. **A held-out eval** (greedy, on the warm pool) before training and after every step — the
   report charts the delta against the base model — plus a real dataset (GSM8K).

Everything else — the warm vLLM pool, the verifiable reward, the adapter handoff, the live
report — is imported from Part 1 unchanged.

## 2. How the work is laid out

```
  driver task  (rl-grpo-driver-v2, CPU)            the async loop; owns checkpoints + the live report
     │
     ├─ init_adapter          (rl-grpo-train, 1 GPU)        mint LoRA v0
     ├─ evaluate              (rl-grpo-rollout, warm pool)  baseline: base model, no adapter
     │
     │   per iteration N (pipelined):
     ├─ generate ×P           (rl-grpo-rollout, warm pool)  batch N+1 with adapter v_N  ─┐ overlaps
     ├─ score_group ×P        (rl-grpo-reward, CPU)         verifiable reward            │
     ├─ train_step_clustered  (rl-grpo-train-cluster)       JobSet: R pods × G GPUs     ─┘ ← batch N → adapter v_N+1
     └─ evaluate              (rl-grpo-rollout, warm pool)  in the background; overlaps train N+1
```

| Environment | Kind | Runs |
|---|---|---|
| `rl-grpo-rollout` | reusable warm pool, 1 GPU/replica, `ReusePolicy(replicas=(1–2, 4), concurrency=1)` | `generate`, `evaluate` — in-process vLLM with `enable_lora=True`; the adapter version is attached per request via `LoRARequest` |
| `rl-grpo-reward` | plain CPU task env | `score_group` — exact-match + format reward, one call per prompt group |
| `rl-grpo-train` | single-GPU task env (Part 1) | `init_adapter` only |
| `rl-grpo-train-cluster` | **`ClusteredTaskEnvironment`** — `replicas` pods × `nproc_per_node` ranks, `TorchRun`, `ClusterFailurePolicy(max_restarts=2)` | `train_step_clustered` |
| `rl-grpo-driver-v2` | plain CPU task env, `depends_on` all of the above | `train_rl_clustered` |

Two hard rules about clustered tasks shape this layout: **rank 0's return value is the task's
output** (other ranks' are discarded, but every rank must return matching types), and **a
clustered task has no controller** — it cannot call other tasks — so the loop lives in the plain
driver and *awaits* the clustered step like any other task.

## 3. Getting started

**Prerequisites**

- A Union deployment with GPU capacity: single-GPU nodes for the pool, and — for the stress
  profile — multi-GPU nodes such as `g6e.12xlarge` (4× L40S). The clustered environment
  currently supports `interconnect="tcp"` only, which is exactly why this tutorial trains a
  LoRA adapter: cross-node traffic is the few-MB adapter gradient, not the model.
- A Hugging Face token stored as a cluster secret named `hf-token`
  (`flyte create secret hf-token --value …`).
- Locally: a `config.yaml` for your tenant, or an API key for the Colab path.

**Profiles.** Everything is sized by one environment variable, read at import time:

| `GRPO_PROFILE` | policy / data | iterations | rollouts / iter | trainer | wall-clock |
|---|---|---|---|---|---|
| `smoke` (default) | Qwen3-0.6B, 8 arithmetic questions | 3 | 4 × 6 | 2 pods × 1 GPU | ~25 min |
| `stress` | **Qwen3-8B, GSM8K** (64 held-out) | 20 | 32 × 6, 512 tokens | **2 pods × 4 L40S**, 2 epochs/step | ~2 h + node provisioning |

`GRPO_GPU_TYPE` (default `L4`) picks the accelerator; set it to `L40s` on tenants whose L4 pool is
unavailable. Both variables are re-injected into every environment's `env_vars`, so the module
constants resolve identically inside the containers (see `_PROFILE_ENV_VARS` in
`rl_grpo_lora.py`).

**Run from a terminal**

```bash
# smoke: proves every path in ~25 minutes
python rl_grpo_lora_clustered.py

# the real thing
GRPO_PROFILE=stress GRPO_GPU_TYPE=L40s python rl_grpo_lora_clustered.py
# the sequential half of the timing comparison
GRPO_PROFILE=stress GRPO_GPU_TYPE=L40s python rl_grpo_lora_clustered.py --sequential
```

The entrypoint prefetches the profile's base model once (`flyte.prefetch.hf_model`) and submits
the driver with the profile's defaults; `--dataset`, `--iterations`, and `--sequential` override
them.

**Run from Colab** — `grpo_rl_colab_client.ipynb` drives the same code with Colab as a pure
client (no torch, no CUDA locally). It walks: upload the three modules → pick the profile →
authenticate with an API key → secret → prefetch → launch → read the receipts → switch to
`stress` in place (reloads the modules, re-prefetches) → optional sequential comparison. Two
notebook-specific gotchas it handles for you:

- In a notebook the SDK defaults to *interactive mode* and ships the task as a cloudpickle
  **without source files**, so the container fails with `ModuleNotFoundError` on the tutorial's
  own modules. The run cells use `flyte.with_runcontext(interactive_mode=False).run(...)`, which
  builds a normal source bundle from `root_dir`.
- The profile is read at import, so switching it means reloading the modules (the switch cell
  does this in dependency order) — no runtime restart needed.

## 4. Walkthrough

### 4.1 The clustered environment

```python
train_env = ClusteredTaskEnvironment(
    name="rl-grpo-train-cluster",
    image=image,                                  # same image as Part 1
    resources=flyte.Resources(gpu=flyte.GPU("L40s", 4), cpu=16, memory="120Gi", shm="auto"),  # PER POD
    replicas=2,                                   # pods == nodes
    nproc_per_node=4,                             # ranks per pod; world_size = 8
    runtime=TorchRun(rdzv_backend="static"),      # in-pod torchrun; restarts are JobSet-level
    failure_policy=ClusterFailurePolicy(max_restarts=2),  # any pod dies → the whole gang restarts
    secrets=[HF_SECRET],
    env_vars=_PROFILE_ENV_VARS,
)
```

Each call to a task in this environment emits one JobSet. `resources` are per pod and must
cover `nproc_per_node` GPUs. Inside the task, `torchrun` has already set `RANK`, `WORLD_SIZE`,
`MASTER_ADDR`; `flyte.ctx()` exposes `local_rank`, `node_rank`, `nnodes`. `reusable` is
rejected on purpose — a JobSet runs to completion, it is not a warm actor; the warm pool stays
where it belongs, on the rollout environment.

### 4.2 Rollouts carry what the trainer needs (`Rollout`, `generate`)

vLLM tokenizes prompt and completion separately, while a trainer that re-tokenizes
`prompt + completion` can merge boundary tokens and misalign — fatal for an importance ratio.
So `generate` requests `SamplingParams(logprobs=0)` and each `Rollout` carries
`prompt_token_ids`, `token_ids`, and `logprobs` (log π<sub>old</sub> of every sampled token,
valid as the behaviour policy because sampling runs at temperature 1.0). The trainer feeds the
model exactly those ids. All three fields default to empty, so Part-1 payloads still deserialize.

### 4.3 The objective (`grpo_sample_loss`)

```python
ratio     = exp(new_logp - old_logp)                                # π_θ / π_old, per token
surrogate = min(ratio * Â, clamp(ratio, 1-ε, 1+ε) * Â).mean()       # PPO-style clip, ε = 0.2
kl        = (exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1).mean()   # k3 estimator ≥ 0
loss      = -(surrogate - β * kl)                                   # β = 0.03
```

`old_logp` comes from the rollout; `ref_logp` is the same weights with the adapter disabled
(`with peft_model.disable_adapter():`) — one extra forward, no second model. Logits are sliced
to the completion positions *before* the fp32 upcast and log-softmax so the transient is
(T × vocab), not (sequence × vocab). At ratio = 1 the surrogate's gradient equals REINFORCE's
(its value does not — it is the constant Â), which is why `mean_loss` sits near zero while the
policy learns; watch the gradient norm and the eval instead.

### 4.4 The DDP discipline (`ddp_shard_step`)

Every rank must execute the same number of collective operations. Part 1's pattern —
per-sample `backward()` with `if advantage == 0: continue` — breaks that under DDP the moment
ranks' shards disagree on how many samples carry signal (they always do: rollouts are
group-contiguous and whole groups get advantage 0), and NCCL desyncs.

```python
def ddp_shard_step(ddp_model, sample_loss_fns, sync_forward):
    with ddp_model.no_sync():
        for fn in sample_loss_fns:
            fn().backward()                          # local grads; graph freed per sample
    (sync_forward().float().sum() * 0.0).backward()  # the ONE synchronizing backward, every rank
```

Backwards inside `no_sync()` accumulate locally with no collectives; the first forward-backward
after the context syncs everything. Every rank unconditionally runs one dummy 1-token forward
and a zero-scaled backward, so the collective count is identical by construction — empty shards
included. (A bare `(0.0 * Σ p.sum()).backward()` would *not* work: it never passes through
`DDP.forward`, so the reducer is not prepared.) Each per-sample loss is scaled by
`world_size / n_valid` so the DDP-averaged gradient equals the single-process mean; `n_valid`
is a property of the rollout payload every rank holds, so no pre-backward communication is
needed. Zero-advantage samples still train — the KL term anchors them; skipping them would bias
the KL and reintroduce data-dependent divergence.

### 4.5 State that persists (`opt_state`, the return contract)

The step takes `opt_state: flyte.io.Dir | None` and returns
`(new_adapter, new_opt_state, TrainMetrics)`. All ranks build AdamW over the same parameter
sequence and all ranks load the state; rank 0 saves the adapter and `optimizer.state_dict()`,
uploads both, and `dist.broadcast_object_list`s their URIs so every rank returns real,
identically-typed `Dir`s. `TrainMetrics` (`mean_loss`, `mean_ratio`, `clip_fraction`, `mean_kl`,
`grad_norm`, `contributing`, `num_valid`) is all-reduced across ranks, so the driver sees global
numbers, not rank 0's shard.

### 4.6 The async driver (`train_rl_clustered`)

```
sequential:  [gen 0][train 0][gen 1][train 1]...
pipelined:   [gen 0][gen 1   ][gen 2   ]...
                    [train 0 ][train 1 ]...      ← JobSet cold start hidden under generation
                             [eval 0  ][eval 1  ] ← off the critical path in both modes
```

```python
rollouts, rewards, gen_s = await pending
if pipelined:                       # kick batch N+1 NOW, with the current (soon one-step-stale) adapter
    pending = asyncio.create_task(_timed_generate_and_score(base, train_pairs, it + 1, adapter, version, seed))
adapter, opt_state, tm = await train_step_clustered(base, rollouts, rewards, adapter, version, opt_state)
if not pipelined:
    pending = asyncio.create_task(...)   # sequential: only now start the next batch
await _settle_eval()                 # the previous step's eval overlapped this train step; back-fill its row
pending_eval = asyncio.create_task(evaluate(base, eval_qs, eval_as, adapter=adapter, version=version))
```

One-step-off-policy training is standard in async GRPO at the labs; the clipped ratio in 4.3 is
what makes it legitimate here. Each step's eval is launched in the background and awaited only
after the *next* step's training, so its row in the report fills in one iteration late; only the
final eval is waited on. Loop state — adapter, optimizer state, baseline accuracy, history — is
checkpointed every iteration; a resumed driver re-launches a lost in-flight eval.

### 4.7 The report

`report_helpers.render_report` publishes a single HTML page after every iteration
(`flyte.report.replace.aio(..., do_flush=True)`) with the Part-1 charts plus three new ones,
rendered only when their data exists:

- **Held-out eval accuracy vs. iteration**, against the base-model baseline.
- **Off-policy drift**: |mean ratio − 1|, clip fraction, KL — near zero on-policy, lifting when
  the policy moves faster than one step's worth.
- **Iteration timing**: generate+score, clustered train step, iteration end-to-end. The overlap
  hidden by pipelining is Σ(gen + train − iteration).

Axis tick precision follows the tick spacing, so small-range series (KL ≈ 0.001, clip fraction
≈ 0.007) read correctly instead of collapsing onto `0.00`.

## 5. Results from the reference runs

**Smoke (0.6B, arithmetic, 2 × 1 GPU, 3 iterations — 25 min).** Every invariant we wanted from
a smoke test, and one lesson: `contributing = 24/24` on all steps (the gradient path really
ran on both ranks); `mean_ratio` 1.0003 / 1.0001 / 0.9996 with clip fraction ≈ 0.6 % (token
alignment between vLLM and the trainer is correct); `mean_kl` **exactly 0.0** on step 1 — a
fresh LoRA has B = 0, so policy ≡ base — then 9e-4 (the adapter-disabled reference works);
adapter and optimizer state chained across steps. The lesson: the three clustered steps took
641 s / 204 s / 329 s against ~60 s of generation, so at smoke scale training is the long pole
and pipelining can only hide that minute. The async claim needs generation to dominate.

**Stress (8B, GSM8K, 2 × 4 L40S, 20 iterations).** Generation of 192 rollouts × up to 512
tokens on four pool replicas takes ≈ 275 s; the clustered step (JobSet spin-up + 8B load + two
epochs on world size 8) ≈ 140 s. Batch N+1 launches 1–2 s before train N and finishes ≈ 130 s
*after* it: the train step, cold start included, is entirely hidden, and the iteration costs
≈ 298 s — the generation time. Sequentially the same iteration is ≈ 440 s. Evals (median
129 s) started before the next train step began and ended well before it did, every time.

Held-out accuracy climbed 60.9 % → 92.2 % (step 14) and settled at 85.9 % — 25 points on 64
questions, against a standard error of ~4. `contributing` fell from 180 to ~100 of 192 as more
prompt groups became all-correct: textbook GRPO saturation, and the cue to move to harder
prompts. Importance ratios stayed at 1.000 ± 0.0003 with a 0.6–0.8 % clip fraction and KL rising
smoothly from 7e-4 to 4.6e-3; gradient norms ≈ 0.02 (the clip at 1.0 never engaged).

Three honest notes:

- **The 61 % baseline is low for Qwen3-8B on GSM8K** (published numbers are 90 %+ with a chat
  template and free chain-of-thought). Our prompt is deliberately plain — no chat template,
  "reason briefly", a 512-token cap, strict `####` exact match — so a good part of the gain is
  the model learning the task's format and length budget. That is legitimate RLVR behaviour
  and a real held-out gain, but it is not new arithmetic ability.
- **The staleness signal is invisible at this learning rate.** Clip fraction was the same
  on-policy and pipelined; the ≈ 0.6 % floor is vLLM-vs-HuggingFace numerics mismatch — the
  well-known training/inference gap, measured here for free. The importance correction is cheap
  insurance that starts to matter at higher LR or more epochs per step.
- **The first JobSet waited 3 h 29 min for nodes.** The JobSet was created at 06:54, torchrun
  started inside the pods at 10:23, and the step then took 90 s. That is 4-GPU instance
  capacity (Karpenter retrying for `g6e.12xlarge`), i.e. gang admission on a shared cluster.
  The driver simply waited — iteration-1 generation had already finished — and the run
  completed without a failure. Budget for it; a nodepool that allows several 4-GPU instance
  types helps.

For scale: Part 1's single-GPU loop on the same 20 × 192 rollouts took ≈ 12.4 min per iteration
(247 min total) versus ≈ 5 min here — but that is not the right pipelining baseline (different
objective, one optimizer step per iteration, no eval). `pipelined=False` under the same profile
is.

## 6. Going further

- **Bigger policies.** L40S-48GB fits an 8B in bf16, so DDP suffices here. Past ~30B, swap the
  DDP wrap for FSDP at the marked line in `train_step_clustered` — on a TCP interconnect prefer
  hybrid sharding (shard within a node over NVLink, replicate across nodes) so cross-node
  traffic stays at LoRA-gradient scale.
- **Harder prompts / curriculum.** `contributing` decaying to ~50 % of the batch is the signal:
  sample prompts the current policy gets *sometimes* right.
- **Kill-a-node demo.** `ClusterFailurePolicy` already restarts the whole gang; with in-step
  checkpointing (`flyte.ctx().checkpoint`) the loss curve continues across a pod death — the
  fault-tolerance receipt no other GRPO tutorial shows.
- **Full-weight RL.** When LoRA capacity runs out: full-parameter training, multi-GB weight
  handoff, vLLM `sleep`/`wake_up` + `load_weights` instead of `LoRARequest`. That is where
  multi-role JobSets (trainer + rollout replicas in one gang, NCCL weight sync) pay off.
- **Model-based or executable rewards.** Replace `score_group`'s rule with an LLM judge on a
  second warm pool, or with code/SQL execution — the reward is just a task, so it scales,
  retries, and versions like one.

## 7. Files

| File | What it is |
|---|---|
| `rl_grpo_lora_clustered.py` | Part 2: profiles' trainer knobs, `ClusteredTaskEnvironment`, `grpo_sample_loss`, `ddp_shard_step`, `train_step_clustered`, `_load_pairs` (arithmetic / GSM8K), the async driver `train_rl_clustered`, CLI entrypoint |
| `rl_grpo_lora.py` | Part 1, extended: `PROFILES`, `Rollout` with token ids + logprobs, `generate` (logprobs on), `_extract_answer` (comma-safe), `evaluate`, the four Part-1 environments with profile env-vars injected |
| `report_helpers.py` | Live report: `IterationMetrics` (+ eval/drift/timing fields), `render_report` with the three new charts, adaptive axis ticks |
| `grpo_rl_colab_client.ipynb` | Colab-as-client notebook: setup → smoke → in-place switch to stress → sequential comparison; Part-1 loop as an optional appendix |
