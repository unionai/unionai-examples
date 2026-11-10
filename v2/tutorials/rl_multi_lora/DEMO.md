# Before vs. after RL: a narrative demo

This is the presenter's runbook. For how the code works, see the tutorial linked from `README.md`.

A 1.5B model fails at a skill. A short, cheap GRPO run (one L4 for training, one or two for rollouts)
fixes it. A side-by-side chat shows both models answering the same prompt, each with a ✅/❌ badge.

| Skill | What the base model gets wrong | What RL teaches it |
|---|---|---|
| **Countdown**: hit a target using each number once with + − × ÷ | guesses, reuses numbers, bad arithmetic | search, backtrack, check the arithmetic |
| **JSON extraction**: messy support email → strict 8-field record | wraps JSON in markdown, grabs the cc'd person's email or the old amount, skips normalization (cents, ISO dates, `ORD-` ids, refund negation) | reliable, schema-exact output |

## Files

| File | Role |
|---|---|
| `tasks.py` | Problem generators, system prompts, verifiable rewards, held-out eval set, stage presets |
| `train.py` | GRPO + LoRA loop (forked from the `rl_grpo_lora` tutorial). Runs the base eval, trains, runs the final eval, waits for a review, then merges the adapter and publishes the **model artifact** |
| `serve.py` | One `VLLMAppEnvironment` per model, bound to its artifact with streamed weights and prefix caching, plus the chat app |
| `chat_ui.py` | Gradio side-by-side chat. Can also run locally against any endpoints |
| `report_helpers.py` | Live training report (before/after accuracy, reward curves, samples) |

## Run it

```bash
uv sync

# 1. Train (prefetches Qwen/Qwen2.5-1.5B-Instruct the first time; no HF token needed)
python train.py --task all          # two runs: json (~30 iterations) + countdown (~60)
#    or one at a time: --task json / --task countdown
#    Each run waits for a review, then publishes an artifact: qwen2-5-1-5b-rl-json / qwen2-5-1-5b-rl-countdown
python review.py <run-name> --approve   # or approve on the run page

# 2. Serve: base + trained models + chat
python serve.py                     # or: python serve.py --tasks json
```

Before the talk, set `MODEL_SCALING` in `serve.py` to `Scaling(replicas=(1, 1))` and redeploy, so no model
scales to zero mid-demo. Then open the chat and press **Settings → Warm up all models**.

Once the demo is over, put `MODEL_SCALING` back and redeploy, so idle GPUs cost nothing.

## Talk track (~5 min)

1. **The problem.** Open the chat with *Countdown* selected and set **both** model dropdowns to `base`.
   Press **Preset 1**, then **Send to both**. The base model answers confidently and gets ❌ (for example
   `7+3+9+2` = 21, not 24). Try **Preset 2** to show it isn't a fluke.
2. **The fix is a loop, not a dataset.** Open the training run's report. No labeled
   solutions exist anywhere: the reward is a 20-line checker (`tasks.score`). Show the reward curve
   climbing and the before/after card ("base 6% → RL 26%" for Countdown, "5% → 92%" for JSON).
3. **The artifact.** In the UI, show the model artifact `qwen2-5-1-5b-rl-countdown`: lineage to the
   training run, plus a model card holding the accuracy numbers. `serve.py` binds to it by name.
4. **After.** Set the right column to `rl-countdown` and resend. It thinks, backtracks, and gets ✅.
   Click **🎲 Random problem** a few times, or ask the audience for numbers.
5. **Second act: JSON.** Switch to *JSON extraction* and press **Preset 1**. The base wraps output in a
   code fence or picks the cc'd email (❌ with the wrong fields listed). `rl-json` returns a bare,
   exact record (✅). Paste in someone's real messy email; it gets a schema check.

## Tuning notes

- **Check the base eval first.** The first report card is the base model's held-out accuracy. If it is
  above about 30%, the task is too easy for a dramatic jump: raise `COUNTDOWN_SIZES` to `(4,)` or widen
  `COUNTDOWN_TARGET_RANGE`; for JSON, add distractors to `_make_ticket`.
- **If the reward curve is flat:** raise `LEARNING_RATE` (3e-5 → 5e-5) or `PROMPTS_PER_ITER`. If it
  collapses, lower the learning rate.
- **The two runs share GPUs.** The rollout and trainer environments are reusable with global scope, so
  both runs use the same replicas: one vLLM engine serves both skills' rollouts and evals, and one trainer
  holds the base weights once with a LoRA adapter + optimizer per skill. While one run trains, the other
  generates. If rollouts queue up, raise the rollout replica max in `rollout_env`.
- Each run is checkpointed. A preempted driver resumes mid-run with its report history.
