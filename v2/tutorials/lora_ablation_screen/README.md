# Can a LoRA screen pick the right RL ablation?

Trains the same GRPO loss-aggregation arms (`token_mean`, `sequence_mean`, `prompt_mean`) twice from one base
model, once as LoRA adapters and once at full weight, and reports whether the two rankings agree (Kendall τ and
"same winner").

The question comes from Mercor's [397B SkyRL guide](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/),
whose 13-row ablation table was all full-weight runs on a 35B model (`prompt_mean` won by 3.9 points). This
example tests the screening *method* on a cheap proxy: Qwen3-0.6B on GSM8K. It says nothing about long-horizon
agentic RL, where completion lengths span 2k–128k tokens instead of tens to hundreds.

## Files

- `lora_screen.py` — the screen. One `train_arm` action per arm on one GPU (cached, `retries=2`); `screen`
  fans out the arms, evaluates the base model once, and builds the comparison report.
- `vllm_generate.py` — sampling in a subprocess (so vLLM's GPU memory is released before each training step),
  returning token ids for token-in/token-out training.

## Run it

```bash
python lora_screen.py                 # 3 aggregations x {lora, full} x 2 seeds = 12 arms, one L4 each
SCREEN_GPU_TYPE=L40s python lora_screen.py
```

Needs a Hugging Face token secret named `HF_TOKEN`.

## What happened when we ran it

Two runs on demo.hosted, Qwen3-0.6B, L4s, 20 GRPO steps of 16 prompts × 8 samples, greedy eval on 300 held-out
GSM8K questions (standard error ≈ 2.9 points per arm). Base model: 37.7%.

**The real screen** (run `uqxmnrv72jl52fkx29kj`, 12 arms: 3 aggregations × {lora, full} × 2 seeds):

| aggregation | full weight, seed 0 / 1 → mean | LoRA, seed 0 / 1 → mean |
|---|---|---|
| `prompt_mean` | 61.7 / 59.0 → **60.3** | 45.3 / 56.0 → 50.7 |
| `token_mean` | 54.3 / 58.3 → 56.3 | 49.3 / 51.0 → 50.2 |
| `sequence_mean` | 52.3 / 56.7 → 54.5 | 49.3 / 53.7 → 51.5 |

- Full weight ranks `prompt_mean` first, the same winner as Mercor's 35B ablation; both of its seeds beat every
  seed of the other two arms.
- LoRA does not: its three arms sit within 1.3 points of each other while its own seeds disagree by up to 10.7
  points. Kendall τ between the rankings is −0.33 and the winners differ.
- LoRA arms learned less in the same 20 steps (mean 50.8% vs 57.0%). The LoRA learning rate (`2e-5`, 10× full)
  wasn't tuned; a longer or better-tuned LoRA screen might rank differently.

**The accidental A/A test** (run `u6w5k62pd5gfccs4pm7t`). The first screen had no stop strings, so every
completion ran to `max_tokens` (length CV 0.0). With equal lengths the three aggregations are the same loss, so
the six arms differed only in noise. Full-weight "arms" still spread from 49.0% to 58.0%, LoRA from 50.0% to 52.7%.
Measure a screen's noise floor before trusting its ranking. `ScreenConfig.stop` now ends completions after the
answer, and because it's a field on an input, changing it invalidated the cached arms.

**Cost per arm** (means per step): full weight 18 s engine init / 17 s generate / 26 s train, 22–23 minutes per arm,
2.4 GB policy written per step. LoRA 19 / 30 / 34 s, 29–34 minutes per arm, 40 MB per step. At 0.6B, LoRA saves
bytes and memory, not wall-clock.
