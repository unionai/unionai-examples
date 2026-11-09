# Pretraining data refinery

Turn raw web text into pretraining data the way LLM data teams do: heuristic filters,
exact and near-duplicate removal across shards, decontamination against eval sets, and a
quality classifier distilled from LLM labels. The input is a slice of FineWeb
(`HuggingFaceFW/fineweb`, `sample-10BT`). An open 14B instruct model under vLLM labels a few
thousand documents for educational value on the FineWeb-Edu rubric, a small classifier learns
those labels, and the classifier scores every surviving document. The curated corpus is
published as an artifact with a dataset card.

The tutorial that walks through this code is at
https://www.union.ai/docs/v2/union/tutorials/frontier-ai/pretraining-data-refinery/.

## Files

| File | What it does |
| --- | --- |
| `refinery.py` | Images, task environments, every stage, and the driver |
| `filters.py` | Gopher, C4 and FineWeb heuristics; token, n-gram and MinHash hashing |
| `labeler.py` | The FineWeb-Edu prompt and the vLLM server on the labeler's GPU |
| `classifier.py` | The frozen encoder, the regression head, and agreement metrics |
| `reports.py` | The run report, its tabs, and the dataset card |
| `viz.py` | Shared HTML and SVG building blocks for reports and cards |

## Requirements

A Union cluster with L40S and L4 GPUs. The reusable environments are Union only. The data
and models are public and need no Hugging Face token.

## Run it

```bash
python refinery.py                     # 15 shards of 20,000 documents, 4,000 LLM labels
python refinery.py --smoke             # 2 shards of 2,000 documents, 300 labels
python refinery.py --threshold 2.5     # same corpus, new threshold: only curate reruns
```
