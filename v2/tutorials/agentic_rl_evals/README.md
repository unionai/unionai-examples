# Harness evals and trace audits for agentic RL

Two pipelines for the work that happens before an agentic RL run spends GPU time:

- **`harness_eval.py`**: a Claude agent answers analyst questions from a synthetic company drive (PDF, XLSX, PPTX,
  inbox) under two harness versions. The harness is a typed input, every (task × harness × model × pass) cell
  is a cached action, grading is separate from generation, and an LLM audits the failures.
- **`trace_audit.py`**: reads published agent trajectories from Hugging Face (Mercor's Terminal-Bench 2.1 eval
  traces by default), computes features for every trajectory, and labels every failure with an LLM. Downloads
  and features are cached; the classifier is an input, so changing it re-labels and nothing else re-runs.

`harness_world.py` holds the worlds, tasks, tools and agent loop. `HARNESS_V1` reproduces the bug classes
from Mercor's [397B SkyRL guide](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/)
(a PDF reader that scrambles a table, a slides tool that returns `None`, missing sandbox packages, a tool with no
timeout in front of a flaky server). `HARNESS_V2` fixes them.

## Run it

```bash
# needs an Anthropic API key in a secret named ANTHROPIC_API_KEY
# (or another name: ANTHROPIC_SECRET=<name> flyte run ...)
flyte run harness_eval.py compare_harnesses
flyte run harness_eval.py leaderboard
flyte run trace_audit.py audit
flyte run trace_audit.py audit --classify false   # no LLM calls
```

## What happened when we ran it (demo.hosted, 2026-11-11)

**Harness v1 vs v2** (`compare_harnesses`, run `u4ss77std7jtxdrwvwhj`, `claude-haiku-5-5`, 6 worlds × 4 tasks × 3
passes = 72 trials per harness, about $0.05 of model spend in total):

| | v1 | v2 |
|---|---|---|
| pass@1 | 76.4% (±2.0 over passes) | 100% |
| pdf_table / email_lookup | 50% / 56% | 100% / 100% |
| slides / sheet_formula | 100% / 100% | 100% / 100% |
| agent timeouts | 8 | 0 |
| masked by a Mercor-style rule | 12.5% | 0% |
| tool calls / mean turns | 231 / 3.83 | 180 / 3.24 |
| tool error rate | 0.0% | 0.6% |

- The broken harness reports a 0% tool error rate: the costly bugs return wrong answers without raising.
- Slides pass on v1 because the agent routes around the bug: `read_slides` returns `None`, `import pptx` fails,
  and the agent unzips the `.pptx` and reads the XML.
- The audit named the PDF layout bug and the email hang unprompted. In an earlier version it also blamed
  `read_spreadsheet` for truncation that was our own digest clipping; the digest now marks its clips.

**Leaderboard** (run `uvwqcfm2bz8nt87lmgd5`): `claude-haiku-5-5` reused 60 of its 72 v2 cells from the run above;
`claude-sonnet-5-5` ran 72 new cells (about $0.28). The 12 Haiku cells that re-ran are all world `w5`: in the
first comparison, the v1 and v2 evaluations built `w5` at the same moment, both missed the cache, and v2's cells
were keyed on the copy that didn't win. The drivers now build every world once before fanning out.

**Trace audit** (run `uz7dpptghk6vxnk625wv`, `mercor/ApexAgentsRecipe-TBench2_1-EvalTraces`, 4 × 267 trajectories,
failures labeled by `claude-sonnet-5-5`). Pass@1 matches the dataset card.

| set | pass@1 | minutes p50 / p90 / max | time in slowest 10% | failed but ≥ 75% tests pass | top label |
|---|---|---|---|---|---|
| Qwen3.6-35B untrained | 44.6% | 27 / 184 / 243 | 27% | 24 / 148 | false_completion 89 |
| Qwen3.6-35B trained | 50.9% | 31 / 184 / 249 | 30% | 14 / 131 | false_completion 55, tool_or_harness 38 |
| Qwen3.5-397B untrained | 50.6% | 10 / 182 / 201 | 49% | 31 / 132 | false_completion 74 |
| Qwen3.5-397B trained | 55.4% | 11 / 183 / 212 | 42% | 21 / 119 | false_completion 60 |

## Gotchas we hit

- **Concurrent calls to one cached task can both run.** Build shared inputs once, then fan out (`warm_worlds`).
- **Child task specs are built in the parent's container.** A secret name read from an environment variable at
  import time must also be in the parent environment's `env_vars`, or children get the default.
- **Warm replicas keep the secrets they started with.** Rotating a key? Rename the reusable environment or let it
  idle out.
- **Cache keys cover a task's inputs and its own body, not imported helpers.** Bump `Harness.version` when tool
  code changes.
