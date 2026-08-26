# Discovery script — SkyRL / Harbor agentic-RL teams

Purpose: surface pain we can actually solve, in the order the [design framework](./design.md#3-the-framework)
predicts it. Each question lists the painful answer to listen for and the Union proposal it maps to.
Five questions are non-negotiable, one per pain: **recovery** ("when the run dies, what do you lose and what does getting back look like — who, how long?"), **scale of loss** (N × trial duration), **observability** ("when it stalls, how do you find which attempt and its logs?"), **cost** ("sandbox + judge spend per step, and how much is discarded attempts?"), **cold start** (decides whether to pitch hosting sandboxes).

## 1. Failure & recovery — T2 state, the unique win

| Ask | Painful answer sounds like | Proposal |
|---|---|---|
| When the training driver dies (OOM, NCCL timeout, a bad trial), what do you lose and how do you get back? | "Everything in flight; we relaunch with `resume_mode=latest`." | Trial-as-action: completed trials survive driver death and are reused on retry |
| How many trials are in flight at once, and how long is a typical one? | e.g. "128 × 5–20 min" | **Loss per crash = N × mean trial × (sandbox $ + inference time)** — get the number |
| How often does a run die per day, and who notices? | "Someone checks wandb in the morning." | `flyte.notify`, retries, `rerun(recover=True)` |
| What's your `ckpt_interval`, and how long does a save block training? | "Every 10 steps; a 32B HF export takes 10 min and once crashed a run (#266)." | Async upload off the step; checkpoint as artifact |
| Did you ever lose a run to a *buffered-but-untrained* batch? | (fully-async buffer = `B·(S+1)` groups) | Same — durable children |

## 2. Hangs & observability — detection, not retry

| Ask | Painful answer | Proposal |
|---|---|---|
| Have you had a run that just… stopped — buffer stuck, no error, Ray refs never resolve? | SkyRL #1173 is their exact config (SWE + Megatron + fully-async + 128 workers) | Per-trial `timeout` → hang localized to "trial 37 / world X"; `max_runtime` as a backstop, not the primary |
| When a trial misbehaves, can you find its logs, its sandbox, its judge transcript? | "One Ray driver log; grep." | Per-action logs + lineage; `flyte.report` per run |
| Which worlds are slowest, most expensive, most often masked? | "Not really." | Per-world metrics from action metadata |
| What happens to the run when *one* trial raises? | `os._exit(1)` in fully-async; `asyncio.gather` in the base generator (#1613) | Per-action retries + `return_exceptions`; failed trial is a record, not a crash |

## 3. Rollout tier & sandbox economics

| Ask | Painful answer | Proposal |
|---|---|---|
| What's the Modal bill per training step, and what fraction of trials are *billed but masked* (timeout, error, judge failure)? | "We don't split it out." | Backpressure + durability so paid rollouts aren't thrown away |
| Do you hit Modal concurrency/quota limits, and what does the trainer do when you do? | "Generation stalls; we lower N." | `queue` with `maxActionConcurrency` = explicit backpressure |
| Time from "trial starts" to the agent's first tool call — how much is image pull? | "Minutes for cold worlds." | Measure vs Flyte pod cold start (E1) and Volume fork (E5) before proposing |
| What differs between your 112 world images — code or data? | "Mostly the files." | Shared MCP image + world data on a `Volume`; `fork()` per trial, reset for free, mid-trajectory branches |
| Where does GPU-cluster ↔ sandbox traffic flow, and who pays the egress? | cross-cloud | Co-locate trials next to the Ray cluster on k8s |
| Sandbox leaks on run failure / Ctrl-C? (Harbor #1194) | "We sweep Modal by hand." | Task lifecycle owns teardown |

## 4. Judge

| Ask | Painful answer | Proposal |
|---|---|---|
| What does a Gemini 429 or timeout do to the reward? | SkyRL pattern: `except: return 0.0` | `@flyte.trace`-memoized judge step + `RetryStrategy(backoff=...)`; a retry never re-bills |
| What's the judge spend per step, and do you ever pay twice for the same attempt? | "Every retry re-calls the judge." | Keep Gemini; rate-limit via queue and memoize per attempt so retries/forks never re-bill |
| Can you re-score last week's rollouts with a new rubric without regenerating them? | "No." | **Fork**: `generate` reused, `verify_and_judge` re-run — judge cost only |

## 5. Ops around the run — T3

| Ask | Painful answer | Proposal |
|---|---|---|
| How do you pin the 1,928-task set to a run? Could you reproduce last week's run exactly? | "Parquet path + git sha, roughly." | Cached extraction → artifact; run inputs are the artifact version |
| Who builds and pre-pulls the 112 world images, and how do you learn one is broken? | mid-run, as masked trials (ThunderAgent's prepull lesson) | Build/prepull fan-out that fails early, cached per image digest |
| How do you eval a checkpoint on held-out worlds? | "Separate script, by hand." | Same trial task against a served checkpoint; eval as a fan-out; results as artifacts |
| How many configs do you run concurrently, and do they interfere? | shared cluster, shared Modal quota | Runs as first-class with isolated pools; sweeps as runs |
| Fork a run at step N with a different staleness / algorithm config — possible today? | "Restart from checkpoint by hand, regenerate everything." | Run fork reuses generation up to N |

## 6. Team & infra

| Ask | Painful answer | Proposal |
|---|---|---|
| You're on Kubernetes — who operates KubeRay and the sandbox fleet, and how much of their week is that? | | L1 substrate; SkyRL has no k8s story |
| Which Ray / vLLM / CUDA pins are you on, and how often does the launch-time `uv` resolve break? | `ray==2.57.0`, cu130, resolved at job start | Flyte-owned image from their Dockerfile; pinned, cached, versioned |
| What would you do with the GPU-hours you lose to crashes and stragglers today? | closes the loop on §1 | |

## Sequencing the conversation

1. §1 first — the N × duration number anchors everything.
2. §2 — #1173 will land; ask if they've seen it.
3. §3 — cold start and image data-vs-code decide the sandbox pitch.
4. §4/§5 — fork/what-if is the "didn't know that was possible" moment; lead with re-scoring old
   rollouts with a new rubric.
5. §6 last — infra questions after the pain is on the table.
