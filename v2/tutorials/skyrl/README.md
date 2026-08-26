# Agentic RL on Flyte: durable rollouts, warm trainer, forkable runs

This folder has two halves:

- **`agentic_rl_durable.py`** — one runnable, CPU-only example that composes every idea below.
  The training is a toy bandit so it runs anywhere; the *structure* is the one a SkyRL/Harbor-style
  agentic RL stack needs.
- **`design.md`** and friends — the decision document behind it: where Flyte helps a SkyRL-shaped
  system, where it doesn't, and the measurements (`experiments.md`) that back each claim.

## The example in one picture

```
driver task (CPU, retries)                      one reusable Ray cluster (scope="run")
 for step in range(n):                           ┌─────────────────────────────────────┐
   ├─ generate(trial, weights, world) ×N ──┐     │ PolicyTrainer  (detached, namespaced │
   │    sandbox pod + Volume.fork(world)   │     │  actor: the trainable state)         │
   │    retries=1, timeout=300s            │     │                                      │
   ├─ verify_and_judge(traj, rubric) ×N ◄──┘     │ train_step(k) ── task per step ──────┤
   │    @flyte.trace judge (never re-billed)     │   re-attaches, applies exactly once  │
   └─ train_step(step, rewards) ─────────────────►│   returns small weights              │
                                                 └─────────────────────────────────────┘
```

| Idea (from `design.md`) | Where in the code | What it buys |
|---|---|---|
| **L1+** steps as tasks on one reusable Ray cluster | `trainer_env` (`ReusePolicy(replicas=1, scope="run")`), `train_step`, `_attach_policy` | driver and cluster are separate failure domains; a driver crash replays steps in ~0.07 s and resumes on the warm actor |
| **L3** trial = two actions | `generate` + `verify_and_judge` | completed trials survive a driver crash; a failed trial is retried/skipped, never fatal; rubric is an input |
| **S2** pod-as-sandbox + `Volume.fork()` | `build_world` (cached), `generate` | each trial gets a private copy of the world in < 1 s; writes are isolated; reset is free |
| **L2b** pull-based weights | `Weights` is an input to `generate` | no NCCL, no engine control plane in the rollout tier (LoRA/small-model shape) |
| **J** memoized judge | `@flyte.trace call_judge` | a retry of the judging task replays the judge instead of paying again |
| **F** fork | `fork_with_new_rubric` → `flyte.rerun(run, recover=True, rubric=...)` | re-score with a new rubric: generation reused, judging + training re-run |
| exactly-once training | `PolicyTrainer.applied[step]`, `rewards` sorted by `trial_id` | a re-executed step can't double-apply; step inputs hash identically across attempts |

## Run it

```bash
# needs a Flyte/Union cluster with the FUSE device plugin (for Volumes) and KubeRay (for the Ray env)
flyte --config <your-config> run agentic_rl_durable.py train

# same, with every failure path exercised: driver crash after step 1, 25% flaky sandboxes,
# 30% judge tasks that fail after the judge call
flyte --config <your-config> run agentic_rl_durable.py train \
    --crash_driver_at_step 1 --flaky_trial_rate 0.25 --judge_flake_rate 0.3

# fork the finished run with a new rubric (re-scores without regenerating what it can)
FLYTE_CONFIG=<your-config> python agentic_rl_durable.py fork <run-name>
```

## What happened when we ran it (org-demo, CPU)

**Run with all failures injected** — `uxvp56fkxpp2hrdz52sm`, 3 steps × 3 worlds × 2 samples:

| | count | note |
|---|---|---|
| `generate` succeeded / retried | 18 / 6 | 6 injected sandbox failures, each retried once, none fatal |
| `verify_and_judge` succeeded / retried | 18 / 6 | retried actions show `JUDGE CALLED` **once** (first attempt) and **zero** times on the retry — the trace replayed |
| `train_step` | 3 | all on actor pid 235 across the driver crash |
| driver | 2 attempts | crashed after step 1; attempt 2 replayed steps 0–1 and ran step 2 |
| learning | mean turns 3.83 → 1.33 → 1.50; mean reward 1.62 → 2.35 → 2.36 | the bandit learned where the secret lives |

**Fork with rubric v2** — `uz8f7wxl7fl2jtjjrzjt` from the run above:

| | phase | count |
|---|---|---|
| `build_world`, `setup_trainer`, durable-time trace | RECOVERED | 3 + 1 + 1 |
| `generate` (step 0) | RECOVERED | 6 |
| `verify_and_judge` (all steps, new rubric) | re-run | 18 |
| `train_step` | re-run | 3 |
| `generate` (steps 1–2) | re-run | 12 |

Step-0 rollouts were re-scored without regeneration (mean reward 1.62 → 2.41 on the *same*
trajectories). Steps 1–2 regenerated — correctly: the new rewards changed the policy at step 0, so
later rollouts depend on different weights. Content-hashed action identity found that invalidation
frontier by itself; nothing in the driver knows about forks.

**A bug we kept as a lesson.** The first crash-injected run (`upq5qwpvvfmj5gb8pqxp`) regenerated
step 1 after the driver retry: rewards were collected in `as_completed` order, which differs between
attempts, so `train_step(0)`'s inputs hashed differently, it re-ran, and the *stateful* actor applied
the update twice. Two rules fell out: canonicalize a step's inputs (sort by `trial_id`), and make the
trainer's update idempotent per step index. Both are in the code.

## Gotchas

- Volumes mount at `/root/flyte-volume/<name>`; use `vol.mount_path`, not `/workspace`.
- A reusable Ray head needs `wget` in the image (readiness probe) and ~2 Gi memory; with 1 Gi it
  sits in "cluster is creating" forever.
- Each Flyte task on the reusable cluster is a **new Ray job in a new process** (~18 s overhead).
  State that must survive across steps lives in detached, **namespaced** actors; module globals don't.
- `flyte.durable.now()` is sync — don't `await` it.

## Files

- `agentic_rl_durable.py` — the example.
- `design.md` — framework, verdicts, the Harbor-style target architecture.
- `discovery.md` / `discovery.html` — the customer-conversation script (HTML is self-contained).
- `unknowns.md` — what the SDK source can't settle; owner + cheapest experiment each.
- `experiments.md` + `experiments/` — E1–E7 measurements and scripts.
