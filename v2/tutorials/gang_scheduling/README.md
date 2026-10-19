# Gang scheduling: all workers or none

A distributed training job is only useful when every worker is running. Without gang scheduling,
a cluster can start some workers and not the rest. Those workers hold GPUs and do nothing, and two
jobs that compete for the same GPUs can both end up half-started, each waiting on the other.

This demo submits two GPU training jobs, `team-a` and `team-b`, at the same moment. Together they
need more GPUs than the pool has. On Union one job is admitted as a whole and all of its workers
start together. The other job waits without starting a single worker or holding a GPU, then starts
as a whole once the first job frees its GPUs.

Each job is a [`ClusteredTaskEnvironment`](https://www.union.ai/docs/v2/union/api-reference/flyte-sdk/flyte.clustered/clusteredtaskenvironment/)
with one extra line, the queue label:

```python
pod_template=flyte.PodTemplate(labels={"kueue.x-k8s.io/queue-name": "shared"})
```

## Requirements

- A Union cluster with gang scheduling turned on and a queue the jobs can use (`shared` by default).
- `flyte>=2.11.2`, and a Flyte config pointing at that cluster.
- A GPU pool that can fit one job but not two: `GANG_WORKERS ≤ GPUs < 2 × GANG_WORKERS`, within both
  the node pool's capacity and the queue's quota. If both jobs fit, they run side by side and the report
  says so. If one job doesn't fit, it is rejected as infeasible.

| Knob | Default | Meaning |
| --- | --- | --- |
| `GANG_WORKERS` | `3` | Workers (pods, one GPU each) per job |
| `GANG_GPU` | `L4:1` | Accelerator per worker. Set to `""` for a CPU-only rehearsal |
| `GANG_QUEUE` | `shared` | The queue the jobs are submitted to |

## Run it

```bash
GANG_WORKERS=3 GANG_GPU=L4:1 uv run --script main.py
```

The script prints the run URL. One run takes about 5 to 8 minutes: GPU nodes start, `team-a`
trains for about 90 s, then `team-b` trains.

To rehearse without GPUs, use `GANG_GPU="" uv run --script main.py`. On CPU the two jobs usually
fit side by side, so you see the gang start but not the waiting.

## What to show

1. **The run graph, while it runs.** `team-a` and `team-b` were submitted together. One shows
   *waiting for gang admission*, then *k of N workers ready*, and turns *Running* only once every
   worker is ready. The other stays queued with no pods.
2. **The report tab on the driver, once it finishes.** One timeline with a row per worker:
   - **Every gang started together.** All of a job's bars begin at the same point. The faint part
     at the start of each bar is how long a worker waited for its peers, and it stays seconds long.
   - **The waiting gang held nothing.** No `team-b` bar starts until every `team-a` bar has ended,
     so `team-b` used no GPU time while it waited.
3. **The code.** Turning on gang scheduling is the queue label above. The training function is
   ordinary PyTorch DDP.
