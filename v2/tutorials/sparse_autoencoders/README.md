# Sparse autoencoders: find features in a language model

Decompose the residual stream of Pythia-70m into interpretable features with TopK sparse
autoencoders, and have Claude label them. One GPU task harvests activations from FineWeb text
and caches them. One GPU task per (layer, width) trains TopK SAEs at several sparsities and
measures variance explained, L0, dead features, and next-token loss with the SAE spliced into
the model. Claude explains a sample of features from their top-activating contexts and is
scored on held-out contexts. The report is a feature browser sorted by that score, and the most
interpretable SAE is published as an artifact with a model card.

The tutorial that walks through this code is at
https://www.union.ai/docs/v2/union/tutorials/frontier-ai/sparse-autoencoders/.

## Files

| File | What it does |
| --- | --- |
| `find_features.py` | Image, task environments, the harvest, training and evaluation, autointerp, publishing, and the driver |
| `sae.py` | The TopK SAE, its training loop with AuxK, its metrics, and feature-context sampling (plain PyTorch) |
| `autointerp.py` | The prompts, schemas, and scoring for explaining features and testing explanations |
| `reports.py` | The training report, the overview, the feature browser, and the model card |
| `viz.py` | Shared HTML and SVG building blocks for reports and cards |

## Requirements

- A Union cluster with NVIDIA L4 GPUs (any CUDA GPU with 24 GB works; change `gpu_env` in
  `find_features.py`).
- An Anthropic API key stored as a secret named `anthropic-api-key`. To use another name, set
  `ANTHROPIC_SECRET` when you launch.

## Run it

```bash
flyte create secret anthropic-api-key
python find_features.py                              # 3 layers x 2 widths, 8M tokens
python find_features.py --smoke                      # 1 layer x 1 width, 0.5M tokens
python find_features.py --widths 4096 16384 32768    # reuses the harvest and the first two widths
```
