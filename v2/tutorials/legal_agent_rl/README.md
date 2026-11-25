# Rubric rewards, judges and checkpoints on Harvey's LAB

Runnable code for *Reading a frontier RL recipe: legal agents*, which reads Harvey's
[Tenet recipe on US open-weight models](https://www.harvey.ai/blog/applying-the-harvey-tenet-training-recipe-to-us-open-weight-models)
against the benchmark it trains on: Harvey's [Legal Agent Benchmark](https://github.com/harveyai/harvey-labs) (LAB,
MIT), 2,010 legal tasks with ~114,000 pass/fail rubric criteria, an agent harness and a grader.

Everything runs LAB's own harness and grader (`lab_core`, pinned to commit `7f9b1ee`) inside Flyte tasks, with US
open-weight models served by vLLM on the same GPU node:

| Model | Lab | Role |
|---|---|---|
| `openai/gpt-oss-120b` | OpenAI | policy, the training judge, and `sweep.py`'s default judge on an 8-GPU node |
| `openai/gpt-oss-20b` | OpenAI | policy, the one we train, and judge A in the posts |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | NVIDIA | policy |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | NVIDIA | judge B |

We also tried `allenai/Olmo-3-7B-Instruct` (it answered in prose instead of calling LAB's tools) and
`thinkingmachines/Inkling-2-Mini` (vLLM 0.34 doesn't support its architecture yet).

This is not a replication of Harvey's results. Our tasks are a seeded sample of LAB's public set (LAB publishes no
hold-out split), our judges are open-weight models, our policies are 20B–120B, and our episodes are capped at a few
dozen turns. Numbers here describe this setup, not Harvey's.

## Files

| File | What it does |
|---|---|
| `images.py` | GPU images built FROM LAB's sandbox image (pandoc, LibreOffice, parsers), plus vLLM 0.34 for ROCm or CUDA |
| `lab_harness.py` | LAB in a pod: `PodSandbox` (LAB's sandbox interface without Podman), a rollout with token capture, an open-weight judge for LAB's grader, Harvey's reward terms |
| `servers.py` | vLLM servers pinned to GPU sets inside one pod; LoRA hot-loading |
| `catalog.py` | Profiles all 2,010 tasks without downloading documents → `lab_tasks.json` |
| `sweep.py` | Base rollouts × judges on one node; each rollout and grade is a `@flyte.trace` |
| `reward_audit.py`, `audit.py` | Which reward term carries the GRPO advantage; judge noise as a share of it; judge agreement |
| `padding.py` | The padding probe: does the rubric pay for non-responsive extra issues, and does a pairwise judge? |
| `train.py` | CISPO + Dr. GRPO, rank-64 LoRA, G=8, on gpt-oss-20b; four data-parallel trainer ranks; resumes after a reclaimed node |
| `diag.py` | Does the trainer's log-probability match the sampler's on real LAB turns? |
| `gptoss_attention.py` | gpt-oss attention sinks and sliding windows on flex attention (exact against eager) |
| `grade_shards.py` | Grade an existing rollout file anywhere, one GPU per (judge, shard) |
| `probes.py` | Padding variants and both-orders pairwise judging (policy pairs, A/A, padding) |
| `checkpoints.py` | (checkpoint × suite) matrix and a pairwise tournament, each cell a cached task on a warm reusable node |
| `preference.py` | Both-orders pairwise judging, win rates with CIs, Bradley–Terry/Elo, power |
| `calibrate.py` | Grades a calibration subset with LAB's standard judges (`claude-sonnet-4-6`, `gpt-5.5`) |
| `smoke.py` | One rollout per model, both judges, one LoRA step |

## Run it

On an 8× AMD MI355X node behind a queue (the configuration we used), or `CLUSTER=demo` for one NVIDIA GPU:

```bash
export CLUSTER=amd GPU_QUEUE=<your GPU queue>
python smoke.py
python sweep.py --tasks 48 --samples 8          # base rollouts and grades (judges: gpt-oss-120b, Nemotron 30B)
python audit.py --sweep-run <run>               # reward-term and judge audit (CPU)
python padding.py --sweep-run <run>             # the padding probe, on the same node
python train.py --steps 30                      # checkpoints
python checkpoints.py --train-run <run>         # matrix + tournament
```

The posts' numbers were graded off the node, one GPU per (judge, shard), with `gpt-oss-20b` and Nemotron 3 Nano 4B
as judges. That path works from any rollout file (the `rollouts` task's output), on any cluster:

```bash
CLUSTER=demo python grade_shards.py --rollouts sweep_rollouts.jsonl.gz --shards 4 --max-model-len 32768
python grade_shards.py merge --rollouts sweep_rollouts.jsonl.gz --runs <run>[,<run>]    # merged file + reward audit
CLUSTER=demo python probes.py variants --rollouts sweep_rollouts.jsonl.gz             # padded copies of each deliverable
CLUSTER=demo python probes.py pairs --rollouts sweep_rollouts.jsonl.gz --variants variants.jsonl.gz --judge gpt-oss-20b
```

On a whole-node GPU pool, build images off-queue first and pin them (`GPU_IMAGE_URI=...`), as the pool's builder may
not be able to push; see `config.py`.

## Results

Rollouts and training ran on one 8× AMD MI355X node behind a GPU market. Grading and the probes ran on NVIDIA
L4/L40S GPUs on a second cluster, from the rollout file.

**Sweep** (48 LAB tasks × 8 rollouts × 3 policies: 1,152 episodes launched and 1,151 finished, in 96 minutes), judged by gpt-oss-20b:

| Policy | Criteria passed | All-pass | Cut off | No deliverable |
|---|---|---|---|---|
| gpt-oss-120b | 31.0% | 0 / 384 | 0.3% | 5.5% |
| Nemotron 3 Nano 30B | 13.0% | 0 / 384 | 7.3% | 34.6% |
| gpt-oss-20b | 11.6% | 0 / 383 | 2.1% | 12.5% |

- The all-pass term was constant in all 144 groups; criteria carried 60–69% of within-group reward variance and
  issue groups 28–40%.
- Re-grading flipped 2.4–4.7% of verdicts: a median 1–4% of the group signal, 7–14% at the 90th percentile.
- A second judge from another family (Nemotron 3 Nano 4B) agreed on 90% of 41,196 criterion verdicts (kappa 0.70),
  passed more criteria (25% against 19%), and showed no sign of the gpt-oss judge favoring its own family.

**Padding** (94 deliverables from gpt-oss-120b): +0.52 / +0.51 points of criteria passed (generic / speculative) on a
33.0% baseline, against 3.9 points of re-grading noise. Pairwise, neither judge penalized the padded drafts.

**Tournament** (both orders): first-shown draft won 69% (gpt-oss-20b judge) / 60% (Nemotron 4B judge) of judgments;
A/A 0.54 (0.46–0.61) / 0.56 (0.48–0.64); gpt-oss-120b beat both other policies at 0.81–0.87 under both judges.

**Training:** the first run's importance ratio was 0.33 because Transformers re-initialized gpt-oss's experts when
reading a dequantized copy back. Training from the original checkpoint (dequantized in memory) fixed it: 0.997–1.001
over seven steps, after which the run was stopped for budget. No learning claims from seven steps.
