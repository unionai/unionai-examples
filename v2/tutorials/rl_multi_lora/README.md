# Several RL runs at once on one shared base model

This example teaches `Qwen/Qwen2.5-1.5B-Instruct` two skills with reinforcement learning (GRPO), one LoRA
adapter per skill. Both runs train at the same time on the same warm GPU replicas and share one copy of
the base weights. Each run checkpoints every iteration, pauses for a person to review its result, and
publishes an approved model as an artifact. `serve.py` then serves every approved model next to the base
model, with a side-by-side chat.

The walkthrough is in the docs: **Tutorials > Reinforcement learning > Concurrent RL runs on a shared
base model**.

## Run it

You need a Union deployment with L4 GPUs and a Flyte config pointing at it (`flyte create config`).

```bash
uv sync

# 1. Train both skills: one run per skill, sharing the same warm replicas.
#    The first time, this also copies the base model into your object store (no HF token needed).
python train.py --task all

# 2. When a run finishes its final eval, it waits for a review. Approve it on the run page, or:
python review.py <run-name> --approve     # or --reject

# 3. Serve the base model, the approved models, and the side-by-side chat.
python serve.py
```

## Files

| File | Role |
|---|---|
| `tasks.py` | Problem generators, system prompts, rewards, held-out eval set, chat presets |
| `train.py` | Environments, rollouts, reward, trainer, driver loop, review gate, artifact publishing |
| `review.py` | Approve or reject a run's review from the terminal |
| `serve.py` | One vLLM app per model bound to its artifact, plus the chat app |
| `chat_ui.py` | Side-by-side Gradio chat with pass or fail grading |
| `report_helpers.py` | The live training report |
| `DEMO.md` | Talk track for presenting the chat live |
