# Speculative decoding models: train, serve, and keep improving draft models

Train EAGLE-3, DFlash, and DSpark draft models for Qwen3-8B, benchmark them against no
speculation and against draft-free methods, serve them with vLLM, and retrain them nightly on
the traces your agent produces.

The tutorial that walks through this code is at
https://www.union.ai/docs/v2/union/tutorials/speculative-decoding/.

## Files

| File | What it does |
| --- | --- |
| `config.py` | Models, GPU type, images, and task environments |
| `workloads.py` | Prompt sources (UltraChat, MT-Bench, GSM8K, ToolACE) and the tool-calling agent |
| `vllm_server.py` | Starts vLLM inside a task and reads its speculative decoding metrics |
| `data.py` | Regenerates chat responses and collects agent traces with the target model |
| `recipes.py` | Per-family settings: starting head, target layers, draft length, training flags |
| `train.py` | Trains any draft model family with `speculators` in online mode |
| `benchmark.py` | Benchmarks one serving configuration per GPU |
| `report.py` | Speedup, acceptance length, losslessness, and the benchmark and gate reports |
| `data_reports.py` | Reports for sampled prompts, generated shards, the dataset, and agent traffic |
| `train_report.py` | Parses training metrics and renders the live training report |
| `report_ui.py` | Shared HTML building blocks for every report and artifact card |
| `pipeline.py` | EAGLE-3: data, training, and benchmarks end to end |
| `dflash.py` | DFlash: fine-tune the public block drafter and benchmark it |
| `dspark.py` | DSpark: train a drafter from scratch and benchmark it |
| `serve.py` | The vLLM app with the draft head, and the agent job that calls it |
| `flywheel.py` | Nightly retraining on agent traces, gated on acceptance length |
| `promote.py` | Union only: draft heads and traces as artifacts with model and data cards, plus rollback |

## Requirements

- A Flyte or Union cluster with GPUs that have at least 40 GB of memory. The defaults use
  one NVIDIA L40S per task. Change `GPU` in `config.py` for other cards, and set
  `TRAIN_NUM_GPUS = 4` to train on a 4-GPU node, which is several times faster.
- A secret named `specdec-api-key` for the serving app:

  ```bash
  flyte create secret specdec-api-key --project <project> --domain <domain>
  ```

  The CLI prompts for the value; use any random string. Organizations with several cluster
  pools reject secrets without a project and domain scope.

## Run it

```bash
# A smoke run: small data, 20 training steps, 12 prompts per workload
python pipeline.py --smoke

# The full pipeline
python pipeline.py

# DFlash and DSpark reuse the cached dataset
python dflash.py
python dspark.py

# Serve the trained head and send agent traffic through it
python serve.py
python serve.py --agent 500

# Retrain on that traffic now, or register the nightly trigger
python flywheel.py --now
python flywheel.py
```
