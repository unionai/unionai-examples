# SkyRL × Flyte — should we, and where? A decision framework

*Status: design/decision document. No code. Verified against SkyRL `main` (2026-08-24) and flyte-sdk
2.6.7 source; experiment numbers from org-demo (CPU only) are in [experiments.md](./experiments.md).*

## 1. The question, and the honest short answer

**Question.** SkyRL is a full-stack RL-for-LLM framework (Ray trainer + vLLM engines + gym/agent
environments). Can Flyte improve its performance, parallelization, and durability — and should it?

**Short answer.** *Mostly no for the inner loop; yes in three specific places.*

1. **Kubernetes substrate + the pipeline around the run — and, on top of it, training steps as
   durable tasks on one long-lived Ray cluster (L1+).** SkyRL has no Kubernetes story at all.
   Wrapping the trainer in a Ray-plugin task and putting data prep, image builds, evals, artifacts,
   sweeps and deploy around it is Flyte's home turf and touches nothing inside SkyRL. Keeping the
   cluster reusable and making each step its own task separates the driver from the cluster as
   failure domains — verified on org-demo (E6).
2. **Durable, observable, *branchable* rollouts for long-horizon agentic RL** (SWE / Harbor-style
   trials that run minutes each in a sandbox). Flyte's one property nobody else has — completed
   child actions survive a driver crash and are reused on retry, and a forked run re-executes only
   what changed — converts directly into recovered GPU-hours and cheap what-if experiments.
3. **Flyte-owned inference replicas with pull-based weight sync for LoRA / small-model RL.** Works
   at any trajectory length (gsm8k included); does *not* work for full-weight RL, where per-step
   weight-sync volume forces engines to stay NCCL-attached inside Ray.

Everything else — engines as Flyte apps, rollouts-as-tasks for short trajectories, wrapping the
Tinker loop in `flyte.trace`, "durability via retries" — is a second-orchestrator tax with a cheaper
alternative that SkyRL users already have. Section 4 argues each.

## 2. SkyRL as built (not as the README says)

The repo was reorganized into one `skyrl/` package (`skyrl-train/`, `skyrl-tx/` are README stubs).

| Concern | Where | What matters for us |
|---|---|---|
| Driver | `skyrl/train/trainer.py` `RayPPOTrainer.train` (L271); runs as `@ray.remote(num_cpus=1) skyrl_entrypoint` | Sync loop is strictly sequential: generate → sleep engines → fwd → train → ckpt → weight sync |
| Fully async | `skyrl/train/fully_async_trainer.py` | K generation coroutines (default 768) + `_AsyncStalenessManager`; bounded `asyncio.Queue` of groups; **generator exception → `os._exit(1)`** |
| Workers | `skyrl/backends/skyrl_train/workers/worker.py` `PPORayActorGroup` | policy / critic / ref roles; FSDP2 or Megatron; **no `max_restarts` anywhere** |
| Engines | `inference_servers/{vllm_server_actor,server_group,vllm_router,remote_inference_client}.py` | vLLM OpenAI servers *inside Ray actors* behind `vllm-router`; control plane (`pause/resume/update_weights/sleep/wake_up`) fans out to **individual server URLs**; `ServerActorPool` docstring: "for now it's just a simple wrapper around a list of actor handles" |
| Weight sync | `weight_sync/{broadcast,cuda_ipc,delta}_strategy.py`, `sharded_rdt/` | NCCL broadcast (<2 s), CUDA-IPC when colocated, **delta via S3/GCS** for non-NCCL-reachable setups, RDT; LoRA path is HTTP (`load_lora_adapter`), no pause |
| Disaggregation seam | `generator.inference_engine.{run_engines_locally,external_server_urls,external_proxy_url}`; `skyrl.train.entrypoints.serve` | Engines can live anywhere HTTP-reachable **if** the control plane can still address each one |
| Generator | `skyrl/train/generators/{base,skyrl_gym_generator}.py` | `GeneratorInterface.generate(GeneratorInput) -> GeneratorOutput` is the pluggable seam; one asyncio task per trajectory; no per-trajectory timeout; env steps in a 32-thread pool with blocking retries |
| Checkpoint | `save_checkpoints()` (trainer.py L1660), `resume_mode=latest` | Synchronous cloud upload blocks training (#252, #266: a 601 s save crashed a run); every worker re-downloads (#800); fully-async persists consumed UIDs only at ckpt time |
| Launchers | Slurm (ThunderAgent), SkyPilot, Anyscale, Runpod, Modal | **No Kubernetes manifests, operator, or docs** |

Fault tolerance in one line: **none.** Any worker/engine crash, NCCL hang, or generator exception
kills the Ray job; recovery is a human relaunch with `resume_mode=latest`, losing up to
`ckpt_interval` steps *plus* everything in flight. The open-issue themes are silent hangs (#1173
buffer deadlock at 125/128 on SWE-Gym + Megatron + fully-async; #24; #757; #76 never exits after
traceback), engine restart fragility (#284 `EADDRINUSE`, #224), and one failed rollout killing the
step (#1613).

Where wall-clock goes on agentic work: SkyRL's own ThunderAgent recipe reports **3.01× (8.84 h vs
26.58 h)** from rollout *scheduling* alone on Qwen3-32B / R2EGym — generation, not training, is the
lever.

## 3. The framework

### 3.1 Classify state by tier

| Tier | Examples | Durability today | Rule |
|---|---|---|---|
| **T1 GPU-resident** | model shards, optimizer, KV cache, NCCL groups | none, and shouldn't need any | **Never cross this boundary with Flyte.** ms-latency coupling; Ray/torch.distributed own it |
| **T2 driver-resident control state** | staleness manager, generation buffer, in-flight rollouts, consumed-prompt set | **lost on crash** | This is where Flyte's durable-children property applies |
| **T3 object store** | checkpoints, datasets, HF exports, trajectories on disk | already durable | Flyte adds lineage, caching, versioning, triggers |

### 3.2 Four tests for any "put Flyte here" proposal

1. **Uniqueness.** Does KubeRay `RayJob.backoffLimit`, Anyscale `max_retries`, or SkyPilot managed
   jobs already give this? (All three restart the whole job on a fresh cluster and resume from the
   last checkpoint. Ray actors fate-share with the driver; nothing below job level survives.)
2. **Preservation.** Does it leave T1 and SkyRL's in-process perf levers untouched — fully-async
   staleness control, KEEP-pause in-flight weight updates, `sticky_least_loaded` session routing,
   CUDA-IPC transfer? Flyte cannot make any of these faster; it can only add hops.
3. **Granularity fit.** Is the work unit long enough (≳ 1 min) that action overhead disappears, and
   is its payload object-store-shaped? Controller defaults: 100 launches/s (`_core.py:91`), 1000
   in-flight children per parent (`_controller.py:146`), inputs always offloaded to blob
   (`_controller.py:242`), 10 MiB inline I/O cap (`models.py:28`).
4. **Hang semantics.** Does it *localize* a hang ("group 37 on world X timed out") or *mask* it
   (whole-job retry that "succeeds" on attempt 3 and hides #1173)?

### 3.3 The Flyte facts the verdicts rest on (verified in source)

- **Child actions and `@flyte.trace` steps are reused across a parent retry.** Sub-action names
  hash `parent-name + inputs + task-body AST + call-seq` with **no attempt number**
  (`src/flyte/models.py:71-88`; `convert.py:793-812`; regression test
  `tests/internal/controllers/test_remote_controller.py:217`). On retry, the informer replays
  known children as already-started and the controller declines to relaunch (`_core.py:524-535`).
- **Fork = recover + new code + changed inputs, in one call.** `flyte.rerun(run, recover=True,
  task_template=<new code>, **changed_inputs)`: succeeded actions from the source run are reused
  (RunSpec `recover` is present in flyteidl2 ≥ 2.0.39), the new code is substituted
  (`task_template=` / `flyte run --rerun-from`, in flyteplugins-union), and inputs you name are
  replaced while the rest keep the prior run's values. Because action identity is content-hashed
  (inputs + task-body AST), only actions whose code or inputs actually changed re-execute; name
  extra ones in `force_rerun_actions` if you want them redone anyway.
  `Volume.fork()` (flyteplugins-union ≥ 0.7, `_base_volume.py:1055`) is a metadata-only
  copy-on-write branch of a filesystem, live or cold, no chunk copy.
- **Apps are one load-balanced URL.** No per-replica addressing, no headless service, request
  timeout capped at 1 h (`src/flyte/app/_types.py:125`, `_app_environment.py:347-365`).
- **Ray plugin** (`plugins/ray`): heterogeneous GPU worker groups yes; `backoffLimit` /
  `submissionMode` absent from the IDL (5 fields); Ray version unpinned by the plugin; SkyRL pins
  `ray==2.57.0` + cu130 and resolves deps at launch via `uv run --isolated --extra fsdp`.
- **Reusable replicas** can spawn children (`examples/stress/long_running_reuse.py`); Ray-plugin
  tasks can spawn children; `ClusteredTaskEnvironment` tasks cannot (`controller_enabled=False`).
- **`flyte.sandbox`** is either Monty (no IO) or a stateless one-shot container
  (`_code_sandbox.py:390`) — not a stateful trial sandbox. `allow_nested_sandboxing()`
  (`_pod.py:162`) is the bwrap prerequisite for an alpha `SandboxEnvironment`.

## 4. Verdicts

| # | Proposal | Verdict | Uniqueness | Preservation | Granularity | Hangs |
|---|---|---|---|---|---|---|
| L1 | Trainer as a Ray-plugin task (`RayJobConfig` + retries + `max_runtime` + object-store `ckpt_path` + `resume_mode=latest`) | **Substrate only** | ✗ (RayJob/Anyscale/SkyPilot) | ✓ | ✓ | masks |
| L1+ | **Steps as tasks on one reusable Ray cluster** (`ReusePolicy(replicas=1, scope="run")`; trainer actors + engines as detached namespaced Ray actors; `train_step(k)` re-attaches) | **Yes** — verified (E6) | ✓ (driver and cluster become separate failure domains; steps replay in ~0.07 s; no launcher does this) | ✓ (T1 stays in Ray; NCCL/KEEP-pause untouched) | ✓ if a step is minutes (job overhead ≈ 18 s) | localizes (per-step `timeout` kills the step, not the cluster) |
| L2 | Engines as Flyte apps via `external_server_urls` | **No** | — | ✗ NCCL needs pod IPs; control plane needs per-replica URLs; 1 h cap | — | — |
| L2b | Engines as **reusable-task replicas**, pull-based weight sync (each action carries a weights version; replicas lazily load, `alru_cache`-keyed) | **Yes for LoRA / small models; no for full-weight** | ✓ (Flyte owns engine lifecycle, autoscale, restart) | ✓ for LoRA (sync is HTTP already); ✗ full-weight (14 GB/replica/step for 7B) | ✓ at group/chunk granularity | localizes |
| L3 | Rollouts as Flyte tasks, engines stay in Ray (task = rollout *client*) | **No gain for short trajectories; yes at group granularity for long-horizon agentic** | ✓ (durable children) | ✓ (KEEP-pause is in-engine) | ✓ only when unit ≥ ~1 min | localizes |
| L4 | Tinker/skyrl-tx loop wrapped in `flyte.trace` | **No** | ✗ (skyrl-tx persists requests itself) | — | — | — |
| P | Pipeline around the run: cached prep, `prefetch.hf_model`, sweeps as runs, evals, artifacts, deploy | **Yes** | ✓ (SkyRL has nothing here) | ✓ | ✓ | n/a |
| S | Flyte as Harbor sandbox provider | **Conditional** (k8s shops; measure cold start) | ✓ | ✓ | ✓ | localizes |

Notes per row:

- **L1.** Do it because it's the k8s on-ramp and the prerequisite for L3/P, not because it's
  "durability" — RayJob `backoffLimit` gives the same restart-to-checkpoint. Costs: a Flyte-owned
  image built from SkyRL's Dockerfile with the Ray pin matched; the driver must run in the Flyte
  task process (not `skyrl_entrypoint.remote`) if it is to spawn children. Retry cost is the
  same as anywhere: cluster respin + image pull + engine start + ckpt reload ≈ 13–24 min.
- **L1+.** Ketan's refinement of L1 and the strongest "durability" claim in this document, because
  it is the one thing whole-job launchers structurally cannot do: the training driver and the GPU
  cluster stop being one failure domain. A `setup` task creates the actor groups and engines as
  detached, namespaced actors; each `train_step(k)` (or `train_steps(k..k+n)`) is a Flyte task that
  re-attaches, runs, and returns metrics + weight version; the driver loop is a plain task. A
  driver crash replays completed steps (E6: 0.07 s each) and resumes on the same warm cluster;
  a raising step retries on the same cluster; a hung step is killed by `timeout`. Costs and
  conditions: ~18 s Ray-job overhead per task (E6), so the boundary must be minutes of work;
  `concurrency=1` means one job at a time on the cluster — overlap must come from rollouts that
  live *outside* the cluster tasks (L3) or from a fully-async epoch inside one step task;
  SkyRL-side, `PPORayActorGroup` / `VLLMServerActor` must be created detached + named and
  `WorkerDispatch` state must be re-derivable from the actors (a contained change, plausibly
  upstreamable); a hung NCCL collective inside an actor still needs an explicit actor reset on
  retry; and if the *cluster* dies you are back to `ckpt_path` — T1 durability stays with the
  checkpoint. Composes with fork: fork at step N = replay steps < N, rebuild actors from ckpt N.
- **L2.** Dead as designed. Delta-via-S3 sync would survive but costs 5–30 s/step against <2 s NCCL.
  Keep engines inside the Ray cluster (SkyRL's default).
- **L2b.** The `rl_grpo_lora` tutorial pattern. Trainer untouched (plug in via `GeneratorInterface`).
  You give up KEEP-pause in-flight updates — staleness becomes explicit per action, which
  fully-async's off-policy correction already handles — and the task must return token ids +
  logprobs (~50 KB/trajectory) to keep token-in/token-out. **The dividing line is weight-sync
  volume per step, not trajectory length.** Granularity rule: one action per prompt group or per
  chunk of ~32 prompts, never per trajectory (gsm8k = 5,120 trajectories/step; at 100 launches/s
  that is 50 s of scheduling for ~30 s of engine work).
- **L3.** For gsm8k-shaped work there is nothing to recover (seconds of GPU time), the engine is
  already saturated by continuous batching, and per-rollout timeouts are an in-process
  `asyncio.wait_for`. For Harbor/SWE trials (minutes each) the math flips — see §5. The unit must
  be the whole group/trial (idempotent regeneration); do **not** `flyte.trace` individual turns —
  a replayed "tool returned X" is only correct if the sandbox is in the same state, and after a
  retry it isn't. The task should use SkyRL's `RemoteInferenceClient` against
  `external_server_urls` so `prompt_token_ids`, `rollout_logprobs`, `cache_salt` and session ids
  survive. Honest alternative: SkyRL could persist completed groups to S3 itself in ~200 lines.
- **L4.** `flyte.trace` replays *return values*; Tinker returns future ids and sampling-client
  handles that mean nothing after a server restart. And skyrl-tx is the JAX path.
- **S.** Harbor already has an undocumented plugin path (`TrialConfig.environment.import_path` →
  a `BaseEnvironment` subclass with ~8 async methods: `start/stop/exec/upload/download`). See §6.

## 5. The business case for L3: loss per crash

Fully-async resume (`fully_async_trainer.py`) persists consumed/filtered prompt UIDs only inside
`save_checkpoints()`. On a crash you lose:

```
lost_groups ≈ (S + 1) · B          # in-flight + buffered-but-untrained (buffer maxsize = B·(S+1))
            + ckpt_interval · B    # trained since last checkpoint → prompts regenerated on resume
```

with `S = max_staleness_steps` (default 4), `B = policy_mini_batch_size` (groups). Every group is
`n_samples_per_prompt` trials. For a ThunderAgent-shaped run (1,200 s trial timeout, ~8 min mean,
`n=4`, `B=32`, `ckpt_interval=10`): `(5·32 + 10·32) · 4 · 8 min ≈ 256 sandbox-hours` of rollouts
plus the matching inference time, per crash, plus 13–24 min of respin. Under L3, the in-flight and
buffered terms go to **zero** (completed trial actions are reused; only genuinely in-flight ones
re-run), and the `ckpt_interval` term stays unless the trainer state is also checkpointed via
`flyte.ctx().checkpoint`. No launcher recovers the first term; that is the whole case.

Gates before building it (numbers in [experiments.md](./experiments.md)): p95 action dispatch on a
warm pool ≤ ~5 s (E1 → **0.89 s, 105 actions/s**); durable-children reuse confirmed on a parent
retry (E2 → **1001 and 106 actions across 4 and 6 attempts**, no re-execution); Ray-plugin retry
semantics understood (E3 → **fresh cluster per attempt, body on the head pod, ~20 s respin on
CPU**). All three pass on org-demo; if E1 had failed the fallback was "upstream group persistence
to SkyRL". Sandbox shape #2 also measured: `Volume.fork()` + mount gives a trial its 1 GB world
in **< 0.7 s** (E5) vs 6.1 s for a blob download of the same bytes.

## 6. Concrete target: a Harbor-style agentic RL stack

A representative production layout (from a team's "what runs where" diagram): HF parquet → 1,928 task dirs across 112 worlds; ECR, one image
per world; Ray GPU cluster (Megatron fully-async, vLLM, in-flight NCCL sync); `TITOHarborGenerator`
= one Ray task per trial ×N; Modal sandbox per trial (Harbor `ModalEnvironment`: MCP gateway → MCP
servers → world filesystem → verifier `tests/test.sh` + grading runner + LLM judge + snapshot diff);
Gemini judge API. Runs on Kubernetes.

| Box | Tier | What breaks today | Flyte |
|---|---|---|---|
| GPU cluster (Ray) | T1 | nothing Flyte can improve | leave; L1 substrate optional |
| Rollout tier — 1 Ray task per trial ×N | **T2** | driver death loses N × minutes × a Modal sandbox each, plus buffered groups; `os._exit(1)`; #1173 is this exact shape | **own** — trial-as-action (§6.1) |
| Modal sandbox per trial | external | per-world image pull cold start; quotas; egress GPU↔sandbox | keep by default; Flyte-hosted sandbox conditional (§6.2) |
| Gemini judge | external | 429/timeout → reward 0 silently; re-billed on every retry | keep Gemini; `@flyte.trace`-memoized call + `RetryStrategy(backoff)` + queue cap (don't pitch replacing it) |
| Task data, 112 ECR images | T3 | manual; no task-set ↔ run versioning; broken images discovered mid-run | cached extraction + artifact; image build/prepull fan-out that fails early |

### 6.1 Trial-as-action, in two actions

```
generate(trial_spec, weights_version)  -> trajectory (token ids, loss mask, logprobs), world snapshot
verify_and_judge(trajectory, snapshot, rubric) -> reward, judge transcript
```

Splitting generation from judging is what makes **fork** pay: a fork with a new rubric has
identical `generate` actions (reused) and changed `verify_and_judge` bodies (re-run) — "would
rubric v2 have changed the last 20 steps' reward distribution?" costs judge calls only. Fork at
step N with a different `max_staleness_steps` or algorithm config reuses generation up to the
fork point. Per-trial `timeout=Timeout(max_runtime=1200)` and `retries` turn a buffer deadlock
into "trial 37 / world X timed out" with logs; a `queue` with `maxActionConcurrency` is real
backpressure against Modal and Gemini quotas. The driver runs inside the Flyte task process —
their custom `main_tito_harbor_fully_async.py` is the natural place — and trial pods need HTTP
reachability to the vLLM router. KEEP-pause weight sync is unaffected (it's in-engine).

### 6.2 Sandbox shape on Kubernetes

Union's `flyte.sandbox` is the wrong primitive (one-shot, stateless). The right shape today is
**trial = Flyte task, pod = sandbox**: one `TaskEnvironment` per world image (generated from the
dataset), the Harbor trial runs in-pod (MCP servers are already in the image), agent → router over
HTTP, verifier in-pod, teardown = task end. v2 is a `FlyteEnvironment(BaseEnvironment)` loaded via
Harbor's `environment.import_path` so Harbor keeps owning lifecycle. What this buys: co-location
with the GPU cluster (no egress), no vendor spend/quota, node-level image cache, secrets, and every
trial is a durable Flyte action. What it costs: cold start — pod scheduling + world-image pull vs
Modal's snapshot restore.

**Shape #2 flips the cold-start comparison:** if the 112 images differ in *data* (PDFs, XLSX,
emails, chat history) rather than code, use one shared MCP-server image and put world data on a
`Volume`; each trial `fork()`s the world volume (metadata-only, instant) instead of pulling a
multi-GB image. "Reset world" becomes free, and a trial's filesystem can be forked
*mid-trajectory* — "what if the agent had taken action B at turn 7?" — which no sandbox vendor
offers. FUSE is allowed in task pods via `PodTemplate().allow_fuse()` (not in app pods).

## 7. Unknowns and experiments

See [unknowns.md](./unknowns.md) (what the SDK cannot tell us — backend-owned behaviors) and
[experiments.md](./experiments.md) (E1 dispatch overhead + cold start, E2 durable children, E3
Ray-plugin retry, E5 Volume fork, E6 steps-as-tasks on a reusable Ray cluster; all CPU-only on org-demo).

## 8. Non-goals

- Making SkyRL's training step, weight sync, or engine routing faster. Those are in-process.
- Replacing Ray inside the trainer, or `ClusteredTaskEnvironment` as the trainer (cannot spawn children).
- Engines as Flyte apps.
- A Flyte sandbox for short, stateless envs (gsm8k, math) — nothing to recover, nothing to isolate.

## 9. Next steps

0. Done: [`agentic_rl_durable.py`](./agentic_rl_durable.py) composes L1+, L3, S2, L2b, the memoized
   judge and fork in one CPU-only example, run and forked on org-demo (E7, [README](./README.md)).
1. `FlyteHarborGenerator` prototype implementing SkyRL's `GeneratorInterface` with the two-action
   trial, against SkyRL's `FullyAsyncTrainerSim` (`fully_async.simulate_training=true`) so it can
   be exercised without training GPUs.
2. Upstream comment on SkyRL #1613 / #1173 describing group-granular durability and per-trial
   deadlines, independent of Flyte.
3. The customer conversation, using [discovery.md](./discovery.md).
