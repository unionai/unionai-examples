# One million decisions in five minutes

A fan-out batch-inference demo: an open-weights **decision model** judges a
million (code, hypothesis) pairs on sixteen T4s, fed by a shared
`DynamicBatcher` per GPU on reusable containers. The data is synthetic code
with planted weaknesses, written by Claude Haiku behind the Union LLM Gateway,
so every decision can be scored against ground truth.

```
driver ──┬─ chunk 1 ─▶ synthesize ─▶ assess ─┐
         ├─ chunk 2 ─▶ synthesize ─▶ assess ─┤
         ├─ chunk 3 ─▶ synthesize ─▶ assess ─┼─▶ merge stats ─▶ report
         └─ chunk N ─▶ synthesize ─▶ assess ─┘

template bank: 21 cached gateway calls ──▶ Union LLM Gateway (claude-haiku-4-5)
synthesize:    CPU,  replicas=1, concurrency=16 ── template instantiation, pure Python
assess:        T4:8, replicas=2, concurrency=16 ── one DynamicBatcher per GPU ─▶ NLI encoder (ModernBERT-base-zeroshot-v2.0)
```

`num_decisions` sets the scale. Each snippet gets seven decisions (a seven-way
choice between "no weakness" and six weakness classes, one hypothesis each),
so a million decisions is about 143k snippets. `chunk_size` is snippets per synthesize task; each chunk is assessed
in batches of `assess_batch_size` snippets so every task's input stays small
enough to inspect in the UI.

## What a "decision" is

A decision model answers narrow, typed questions with a probability attached
and never writes text. TypeSafe AI's Jev is the reference for the category: it
answers **Noul** (a yes/no probability), **Choice** (one option from a list)
and **Score** (one of several ordered levels) questions with calibrated
confidence. This demo uses an open-weights model of the same kind: a zero-shot
natural-language-inference (NLI) encoder,
[`MoritzLaurer/ModernBERT-base-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0)
(Apache-2.0, 8k-token context, ~150M parameters).

Each option of a question is written as a hypothesis about the code. **One
forward pass over a (code, hypothesis) pair is one decision**: it returns how
strongly the code entails the hypothesis. A Noul question is the entailment
probability of its single hypothesis; a Choice question normalises its
options' scores into a distribution.

| Question | Kind | Hypotheses | Decisions |
|---|---|---|---|
| Which best describes this code? | choice | no weakness / SQL injection / command injection / path traversal / hard-coded secret / insecure deserialization / SSRF | 7 |

"No weakness" competes with the six classes in one choice, so every decision
is relative to its alternatives and P(vulnerable) is simply one minus the
probability of "no weakness". A lone yes/no hypothesis (Jev's Noul) is
supported by the code too, but an NLI encoder's raw entailment probability is
not calibrated against anything, so it answers "no" far too often; keep
questions as choices between stated alternatives.

Because every decision carries a probability, the pipeline can **abstain**: a
snippet whose top option is below `REVIEW_THRESHOLD` is routed to a
human-review bucket, and the report shows a calibration table so you can see
whether P(vulnerable) tracks the real rate of planted weaknesses.

### Other open-weights models of this kind

Any model below drops into `assess_snippets` with a different `_decision_model`
loader; the batchers, the questions and the report stay the same.

| Model | Shape | Trade-off |
|---|---|---|
| [`MoritzLaurer/ModernBERT-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) | Same API, ~400M parameters | More accurate, about three times the cost per decision |
| [`knowledgator/gliclass-modern-large-v3.0`](https://huggingface.co/knowledgator/gliclass-modern-large-v3.0) | Zero-shot classifier; all labels scored in **one** pass per text | A six-way choice becomes one forward pass; needs the `gliclass` package |
| [`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) | Zero-shot NLI, same API | Stronger on prose, but a 512-token window |
| Reward models such as [`Skywork/Skywork-Reward-V2-Qwen3-1.7B`](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-1.7B) | A scalar score per (prompt, response) | A natural fit for Score questions, not for Choice |
| Fine-tuned vulnerability classifiers (CodeBERT or UniXcoder heads trained on Devign or BigVul) | A single binary head | Purpose-built for the noul question only; no zero-shot options |

A generative LLM can be forced into this shape too, by reading its next-token
distribution over option letters. It works, but the probabilities are not
trained to be calibrated and every option still costs a full prompt; the
encoders above are built for the job.

## Layout

| Path | What it is |
|---|---|
| `main.py` | The whole demo: environments, template bank, synthesis, decisions, and the report |

The file is organised top to bottom in the order the pipeline runs:

1. **Models and environments.** A small CPU image for the synthesizer, a
   torch + transformers image for the decision model, a plain image for the
   driver. Both worker environments use `ReusePolicy`; the decision one asks
   for `T4:8` per replica and two replicas.
2. **Dataset spec.** Six weakness classes, three languages, and the
   placeholder scheme that turns a template into many unique snippets.
3. **Stage 1, `build_template_bank`.** One traced gateway call per
   (language, class), asking for four short templates each. It takes a few
   seconds with Haiku, sits outside the timed window, and renders a report of
   the templates it got.
4. **Stage 2, `synthesize_snippets`.** Instantiates a chunk of snippets from
   the bank. Everything derives from the seed and the snippet index, so a chunk
   is reproducible regardless of chunking.
5. **Stage 3, `assess_snippets`.** One `DynamicBatcher` per GPU, created once
   per replica. Each (code, hypothesis) pair is a record, round-robined across
   the batchers; `Hypothesis.estimate_cost` gives them a token budget to pack.
   The task returns aggregates, never rows.
6. **Stage 4, `decision_demo`.** Fans out per chunk, chaining synthesize into
   assess, merges the chunk statistics, and renders the report.

## Run

```bash
# warm up: builds images, provisions the T4 nodes, fills the template cache
flyte run main.py decision_demo --num_decisions 70000

# the demo
flyte run main.py decision_demo
flyte run main.py decision_demo --num_decisions 1000000 --chunk_size 6000 --assess_batch_size 1000
```

| Parameter | Default | Effect |
|---|---|---|
| `num_decisions` | `1000000` | How many decisions to make; snippets = ceil(decisions / 7) |
| `chunk_size` | `6000` | Snippets per synthesize task |
| `assess_batch_size` | `1000` | Snippets per assess task (7,000 decisions). Tasks beyond a replica's `concurrency` queue up; much bigger batches starve the queues at the tail |
| `clean_fraction` | `0.3` | Share of snippets instantiated from clean templates |
| `seed` | `0` | Makes the dataset reproducible |

Run it twice. The first run pays for image builds and T4 node provisioning;
`idle_ttl=600` keeps the replicas warm for ten minutes, so the second run is
the timed one. The report's wall time covers synthesis and assessment and
excludes the template bank.

Two scheduling facts matter on a small GPU pool. A reusable environment is
tied to the code that defined it, so a code change deploys a new environment
and the old one keeps its GPU nodes until its `idle_ttl` expires; that is why
the timeout is minutes, not an hour. And the replica count is fixed rather
than an autoscaling range: a replica that scaled down between runs costs
minutes of node provisioning on the next one, which is most of the difference
between a five-minute run and a three-minute one.

## Secrets

| Secret | Env var | Purpose |
|---|---|---|
| `DEMO_GATEWAY_ANTHROPIC_API_KEY` | `LLM_GATEWAY_API_KEY` | Gateway virtual key for the Anthropic provider |

The gateway URL and model id are the `LLM_GATEWAY_URL` and `SYNTH_MODEL`
constants at the top of `main.py`. Gateway model ids are
`<provider>/<model>`, and each provider has its own virtual key, so switching
to another model (for example the open-weights `qwen38-27b-vllm/qwen38-27b`
with its own key) is a two-line change. The gateway speaks the OpenAI chat
completions API, so any OpenAI-compatible endpoint works there too.

## The report

The driver renders a Flyte report with:

- Headline cards: decisions, wall time, decisions per second, GPUs and
  replicas used, flagged, needs-review, detection accuracy, weakness-class
  accuracy.
- A confusion matrix of planted weakness vs. decided weakness.
- A calibration table for P(vulnerable).
- Per-GPU batcher stats on every replica: records, batches, average batch
  size, and utilization as reported by `DynamicBatcher.stats`.
- A few sample snippets with their decisions.

## Patterns worth copying

- **Process-level singletons on a reusable container.** The models and
  batchers (on the GPU replica) and the HTTP client (on the CPU replica) live
  at module level behind a guard. A reusable container keeps the process alive
  across task invocations, so they are created once per replica and every task
  that lands there shares them.
- **One batcher per GPU.** A `DynamicBatcher` runs one batch at a time, so a
  multi-GPU replica gets one batcher per device and tasks round-robin their
  records across them. Each batcher's forward pass runs in its own thread, so
  the eight GPUs work in parallel.
- **Aggregates, not rows.** Each assess task returns counts, a confusion
  matrix, calibration bins and a couple of samples. A million decisions never
  become a million-row task output or a million-row report.
- **Keep the creative step small.** The LLM writes 21 small template sets;
  the pipeline instantiates them into as many snippets as the run needs. The
  data stays varied enough to score against, and the gateway's latency drops
  out of the timed path.
- **Bound the padding.** A batch is padded to its longest pair, so pairs are
  capped at 256 tokens and each batch is split into length-sorted halves
  before the forward pass.
- **Chain per chunk.** Synthesize and assess are chained per chunk rather than
  staged across the whole dataset, so the GPUs start as soon as the first chunk
  exists.
- **Typed decisions with confidence.** The model only ever scores listed
  hypotheses, so the pipeline gets a probability per answer and can route
  low-confidence cases to a human instead of guessing.

## Tuning

| Knob | Where | When to change it |
|---|---|---|
| `replicas` | decision `ReusePolicy` | More T4:8 nodes, more throughput; cold replicas take minutes to provision |
| `idle_ttl` | both `ReusePolicy` | Longer keeps replicas warm between runs; shorter frees GPU nodes sooner after a code change |
| `concurrency` | `ReusePolicy` | Raise it if `utilization` in the report is low: more concurrent producers keep the queues full |
| `target_batch_cost`, `max_batch_size` | `get_decision_batchers` | Sized for short snippets on a T4 in fp16; lower them if you hit OOM |
| `MAX_PAIR_TOKENS`, `LENGTH_BUCKETS` | decision stage | Padding control; the GPUs are compute-bound, so shorter pairs mean more decisions per second |
| `chunk_size`, `assess_batch_size` | run parameters | Bigger batches mean fewer tasks and less scheduling overhead; smaller ones spread work across replicas sooner and keep task inputs inspectable |
| `DECISION_MODEL` | top of file | See the alternatives table above |
| `TEMPLATES_PER_CALL` | template bank | More templates per class means more variety per weakness |

## Requirements

- A GPU node pool with T4 accelerators, eight per node, for the decision
  model. The forward pass runs under fp16 autocast because the T4 has no
  native bfloat16; weights stay fp32 since ModernBERT keeps some buffers in
  fp32 and a pure fp16 load fails without FlashAttention.
- Access to a Union LLM Gateway (or any OpenAI-compatible endpoint) and the
  secret above.
- `unionai-reuse` in both worker images (already included). It installs the
  actor bridge that reusable containers need.
