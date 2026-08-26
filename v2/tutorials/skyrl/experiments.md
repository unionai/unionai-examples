# Experiments — org-demo, CPU only (2026-08-25)

All runs: `demo.hosted.unionai.cloud`, org `demo`, project `ketan`, domain `development`.
Scripts in [`experiments/`](./experiments/). Numbers are single runs on a shared demo cluster —
directional, not benchmarks.

## E2 — durable children / traces across a parent retry  ✅ confirmed

**Question.** When a parent task crashes and is retried, are completed child actions and
`@flyte.trace` steps reused, or re-executed?

**Trace variant** — `flyte-sdk/examples/stress/crash_recovery_trace.py`
(run `ushp9k7qwd78284644fd`): parent iterates 1000 `@flyte.trace` calls, deliberately crashing at
i=100 / 200 / 300 on attempts 0 / 1 / 2, then succeeds on attempt 3.

| attempt | phase | duration | what happened |
|---|---|---|---|
| 1 | FAILED | 13.2 s | 100 traces executed, crash |
| 2 | FAILED | 14.5 s | 100 replayed + 100 executed, crash |
| 3 | FAILED | 26.5 s | 200 replayed + 100 executed, crash |
| 4 | SUCCEEDED | 62.4 s | 300 replayed + 700 executed |

**Total sub-actions in the run: exactly 1001** (a0 + 1000 traces) across 4 attempts — replayed
steps produced no new actions. Replay/execute round-trip ≈ 60–90 ms per trace step as seen from the
parent. Gaps between attempts: 5–13 s (pod reschedule on a cached image).

**Child-task variant** — `examples/stress/long_recovery.py` (run `u24lz45h64bl7vhnqbk7`):
5 parallel children + 100 sequential children, deliberate crashes on attempts 0–4 (the last two
at i=30 and i=90 of the sequential loop). **6 attempts, exactly 106 actions** (a0 + 5 + 100);
re-execution would have produced ≥ 220. Attempt durations 12.5 / 26.6 / 26.5 / 129.5 / 223.4 /
59.8 s — the final attempt replayed 95 children and ran the remaining 10.

**Implication for L3.** A driver crash does not lose completed rollout actions; only genuinely
in-flight ones re-run. This is the property behind the loss-per-crash argument in
[design.md §5](./design.md#5-the-business-case-for-l3-loss-per-crash).

## E3 — Ray-plugin task retry semantics  ✅ answered

**Question.** Does a Flyte retry of a `RayJobConfig` task get a fresh Ray cluster, where does the
task body run, and what does the respin cost?

`experiments/e3_ray_retry.py` (run `u7pzszbz7djczssb248k`): `retries=2`, crash on attempt 0,
head + 1 CPU worker, `ray[default]==2.46.0`.

| attempt | task pod | Ray node id seen from body | pod start → body start | result |
|---|---|---|---|---|
| 0 | `…-a0-0` | `6bfed1cb0733` | 17.2 s | simulated crash |
| 1 | `…-a0-1` | `dde902525a90` | 16.8 s | SUCCEEDED |

- **Fresh cluster per attempt** (unknowns #4): different Ray node ids, different pods.
- **The task body runs on the Ray head pod** (unknowns #5): the only node visible at body start
  is the task's own hostname; `RAY_ADDRESS` is unset and `ray.init()` attaches locally. So the
  SkyRL driver would live on the head — which SkyRL's own `skyrl_entrypoint.remote` avoids on
  purpose ("make sure the training loop is not run on the head node"); the head must be sized for
  the driver, or the loop must be pushed to a worker as SkyRL does today.
- **Respin cost on CPU with a cached image: ~4 s between attempts + ~17 s to a running body.**
  On GPU nodes with a 20 GB image this becomes the 13–24 min estimated in design.md; the Flyte
  overhead itself is negligible.

## E1 — per-action overhead on a warm pool; cold start of fresh pods  ✅ measured

`experiments/e1_warm_only.py` (run `ugtv4zscl77k7vsx678h`): 500 no-op children on a
`ReusePolicy(replicas=(2,8), concurrency=16)` pool in batches of 100, wall time per child as
seen by the parent (`create_task` → result), then 10 concurrent fresh-pod children on a small image.

| | p50 | p95 | max | min |
|---|---|---|---|---|
| warm pool, per action | **0.68 s** | **0.89 s** | 1.41 s | 0.31 s |
| fresh pod, small image (cached on node) | 5.5 s | 13.0 s | 13.0 s | 3.3 s |

Sustained: **105 actions/s** over 500 actions (4.76 s wall) — i.e. the parent's own launch
limiter (`_F_MAX_QPS`, default 100) was the ceiling, not the backend or the pool.

`experiments/e1_dispatch_latency.py` (run `u4jkm4kvwl8nxcnwg4dz`) repeats the above (warm p50
0.53 s / p95 0.80 s, 90 actions/s; fresh small-image pods p50 4.0 s / p95 8.8 s) and adds 3
sequential fresh pods on a ~10 GB image (`vllm==0.11.0`):

| big-image pod | wall time | what dominated |
|---|---|---|
| 1st (image not on node) | **93.4 s** | image pull |
| 2nd | 4.0 s | pod scheduling only (image cached) |
| 3rd | 3.0 s | same |

So a per-world container image costs ~90 s per *cold node* and ~4 s afterwards; with 112 world
images and autoscaled nodes, most trials would land on a node that has not seen their image. Against
that, E5's `Volume.fork()` gives a trial its world in < 0.7 s from any node. This is the number
behind sandbox shape #2 in [design.md §6.2](./design.md#62-sandbox-shape-on-kubernetes).

### E4 — what E1 means for rollout granularity (analytic)

Per-action cost ≈ 0.7 s latency, ≥ 100/s throughput (raise `_F_MAX_QPS` to test higher).

| workload | unit | actions / step | launch time at 100/s | overhead vs work | verdict |
|---|---|---|---|---|---|
| gsm8k (1024 prompts × 5) | per trajectory | 5,120 | ~51 s | ≫ the ~30 s of engine time | ✗ |
| gsm8k | per prompt group (`n=5`) | 1,024 | ~10 s | comparable; only worth it with L2b's other gains | marginal |
| gsm8k | per chunk of 32 prompts | 32 | < 1 s | negligible | ✓ (L2b) |
| Harbor / SWE trial (5–20 min) | per trial | N in flight (≤ 1000 default) | ~1 s per 100 | ≈ 0.1–0.2 % | ✓ (L3) |
| Harbor / SWE | per GRPO group of 4–8 trials | N/4–N/8 | negligible | negligible | ✓ (L3) |

Pull-based weight sync per replica (L2b): LoRA adapter (MBs) → sub-second first-use load per
replica per version; full-weight 7B bf16 (14 GB) → tens of seconds per replica per step, on top of
the blob download — the reason L2b is LoRA/small-model only.

**Gates for L3 (design.md §5): E1 p95 ≤ 5 s → 0.89 s ✓; E2 durable children ✓ → proceed.**

## E5 — `Volume.fork()` as a per-trial world filesystem  ✅ measured

`experiments/e5_volume_fork.py` (run `uzjcn6zhlftgrt7z8hfw`): populate a 1 GB (100 × 10 MB)
world volume, `finalize()`; three "trials" in parallel each cold-`fork()` it, mount, read all
1 GB, write a file; compare with a `flyte.io.Dir` download of the same bytes.

| step | trial 1 | trial 2 | trial 3 |
|---|---|---|---|
| `parent.fork(name)` (cold, no parent mount) | 0.12 s | 0.12 s | 0.10 s |
| `forked.mount()` (JuiceFS FUSE, sqlite meta) | 0.54 s | 0.57 s | 0.57 s |
| **trial has its world** | **< 0.7 s** | | |
| sequential read of all 1 GB over FUSE | 11.1 s | 9.8 s | 9.8 s |
| fresh pod wall time incl. the above | 15.5 s | 13.5 s | 13.5 s |

Baseline: `flyte.io.Dir.download()` of the same 1 GB = **6.1 s** (9.8 s action). Populating the
parent (write 1 GB + `finalize`) = 12.2 s. All three forks wrote independently (disjoint key
spaces), as documented.

Reading: the world is available to a trial in **~0.7 s independent of world size**, and bytes
stream lazily at ~100 MB/s only for files the agent actually touches — an agentic trial touches a
handful. A per-world container image pays the full pull up front on every cold node. This is the
"sandbox shape #2" number in [design.md §6.2](./design.md#62-sandbox-shape-on-kubernetes).

Notes: first attempt (`ucmwmzqlzmcf564rvpnz`) was an authoring bug — the 0.8.2 default mount path is
`/root/flyte-volume/<name>` (use `vol.mount_path`), not `/workspace`. The demo cluster has the
FUSE device plugin (`PodTemplate().allow_fuse()` → JuiceFS 1.3.1, data in the project S3 bucket).

## E6 — training steps as Flyte tasks on ONE reusable Ray cluster  ✅ confirmed (Ketan's proposal)

`experiments/e6_reusable_ray_steps.py` (runs `uw9hcdljtx698wnprdrx`, then `u299zpzp7d4srlnctdpc`
with a Ray namespace): a plain driver task loops `train_step(k)` for k=0..5 on a
`RayJobConfig` environment with `ReusePolicy(replicas=1, idle_ttl=600, scope="run")`. Each step
bumps a version counter on a named, detached Ray actor ("the policy"). Step 2 raises on its first
attempt (`retries=1`); the driver crashes after step 3 on its first attempt (`retries=1`).

| k | driver attempt | task pid | Ray job id | actor pid | policy version | wall seen by driver |
|---|---|---|---|---|---|---|
| 0 | 0 | 1182 | 02 | 259 (created) | 1 | 0.29 s (replayed on attempt 1) |
| 1 | 0 | 1421 | 03 | 259 (re-attached) | 2 | 0.07 s (replayed) |
| 2 | 0, retried | 1817 | 05 | 259 | 4 (3 was consumed by the failed attempt) | 0.07 s (replayed) |
| 3 | 0 | 2025 | 06 | 259 | 5 | 0.06 s (replayed) |
| 4 | 1 | 2253 | 07 | 259 | 6 | 18.5 s |
| 5 | 1 | 2451 | 08 | 259 | 7 | 18.6 s |

All steps ran on the same head pod (`ray-a2ee255c989fcee4-head`). Findings:

1. **The cluster and its actors survive everything above it**: a step that raises, a step retry,
   and a driver crash + retry. The driver's retry replayed steps 0–3 from Flyte's record
   (~0.07 s each) and continued with 4–5 on the *same* cluster with the *same* actor state.
2. **Each Flyte task is a new Ray job in a new process** (distinct pids and job ids). Module
   globals do *not* carry across steps. State that must persist across steps has to live in
   **detached, named, namespaced** actors (`.options(name=..., namespace=..., lifetime="detached",
   get_if_exists=True)` / `ray.get_actor(name, namespace=...)`). Without an explicit namespace
   (first run) every task got its own anonymous namespace and re-created the actor.
3. **Per-step overhead ≈ 18 s** — Ray job submission plus the plugin's `runtime_env`
   `working_dir` upload in `pre()` — versus 0.7 s for a plain reusable task. Cluster creation
   was ~40–60 s once per run. So the step boundary should be minutes of work, not seconds.
4. Constraints: the Ray env is `replicas=1, concurrency=1` → one job at a time on the cluster;
   the reusable head needs `wget` in the image (readiness probe) and more memory than a
   throwaway head (`(2000Mi, 4000Mi)` worked; `(1Gi, 2Gi)` stalled in "cluster is creating").

**What this means for SkyRL (design.md, "L1+").** The trainer's actor groups and vLLM engines
become detached, namespaced actors created by a `setup` task and re-attached by every
`train_step(k)` task; each step is a durable Flyte boundary; a driver crash costs a replay, not a
cluster respin; a hung step is killed by `timeout` without taking the cluster down (the actors
may still need an explicit reset on retry). T1 durability is still the checkpoint's job — if the
cluster itself dies you rebuild from `ckpt_path` — but the driver and the cluster are no longer one
failure domain.

## E7 — the integrated example (`agentic_rl_durable.py`)  ✅ run + forked

See [README.md](./README.md#what-happened-when-we-ran-it-org-demo-cpu) for the full tables.
Runs: `uxvp56fkxpp2hrdz52sm` (crash-injected training: driver crash after step 1, 25% flaky
sandboxes, 30% judge flakes — 18/18 trials completed, 6+6 retries, 3 steps on one actor, learning
visible), `uz8f7wxl7fl2jtjjrzjt` (fork with rubric v2: worlds + setup + step-0 generation RECOVERED,
judging + training + steps 1–2 generation re-run). `unj795hxr268jvdshxv6` (a `durable.now()` await
bug) and `upq5qwpvvfmj5gb8pqxp` (the as_completed-order / double-apply lesson) preceded them.

## Not run

- Unknown #1 (orphaned children on SIGKILL), #7 (reuse-worker behavior on hang), #8 (checkpoint
  across `rerun --recover`), #10 (trial pod → Ray router reachability): cheap variants are listed in
  [unknowns.md](./unknowns.md); not needed for the current verdicts.
