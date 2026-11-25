# Rubric rewards, judges and checkpoints on Harvey's LAB

Runnable code for *Reading a frontier RL recipe, season 2*, which reads Harvey's
[Tenet recipe on US open-weight models](https://www.harvey.ai/blog/applying-the-harvey-tenet-training-recipe-to-us-open-weight-models)
against the benchmark it trains on: Harvey's [Legal Agent Benchmark](https://github.com/harveyai/harvey-labs) (LAB,
MIT), 2,010 legal tasks with ~114,000 pass/fail rubric criteria, an agent harness and a grader.

Everything runs LAB's own harness and grader (`lab_core`, pinned to commit `7f9b1ee`) inside Flyte tasks, with US
open-weight models served by vLLM on the same GPU node:

| Model | Lab | Role |
|---|---|---|
| `allenai/Olmo-3-7B-Instruct` | Ai2 | policy; the one we train |
| `openai/gpt-oss-20b` | OpenAI | policy |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | NVIDIA | policy, and judge B |
| `openai/gpt-oss-120b` | OpenAI | policy, and judge A |

This is not a replication of Harvey's results. Our tasks are a seeded sample of LAB's public set (LAB publishes no
hold-out split), our judges are open-weight models, our policies are 7B–120B, and our episodes are capped at a few
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
| `train.py` | CISPO + Dr. GRPO, rank-64 LoRA, G=8, on Olmo 3 7B, to make checkpoints |
| `checkpoints.py` | (checkpoint × suite) matrix and a pairwise tournament, each cell a cached task on a warm reusable node |
| `preference.py` | Both-orders pairwise judging, win rates with CIs, Bradley–Terry/Elo, power |
| `calibrate.py` | Grades a calibration subset with LAB's standard judges (`claude-sonnet-4-6`, `gpt-5.5`) |
| `smoke.py` | One rollout per model, both judges, one LoRA step |

## Run it

On an 8× AMD MI355X node behind a queue (the configuration we used), or `CLUSTER=demo` for one NVIDIA GPU:

```bash
export CLUSTER=amd GPU_QUEUE=<your GPU queue>
python smoke.py
python sweep.py --tasks 48 --samples 8          # base rollouts and grades
python audit.py --sweep-run <run>               # reward-term and judge audit (CPU)
python padding.py --sweep-run <run>             # the padding probe
python train.py --steps 30                      # checkpoints
python checkpoints.py --train-run <run>         # matrix + tournament
```

On a whole-node GPU pool, build images off-queue first and pin them (`GPU_IMAGE_URI=...`), as the pool's builder may
not be able to push; see `config.py`.

## Results

See the docs tutorial and the blog series; this section is filled in from the runs.
