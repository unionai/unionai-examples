# System 1 + System 2 agents on Flyte

Interleaving a **System One model** — [TypeSafe](https://docs.typesafe.ai/introduction)'s
"Jev", via the `flyteplugins-typesafe-ai` plugin — with a generative **System 2**
model inside durable Flyte tasks.

Jev does not write text. It answers narrow, typed questions — `Choice`, `Score`,
`Noul` — in parallel and in isolation, with calibrated confidence attached to
every answer. That makes it a good model-based I/O guard and decision layer to
put in front of, and between, expensive generative calls.

```
System 1  ── typed answers (Choice / Score / Noul) ──▶  guards, routing, tool plans
System 2  ── open-ended reasoning and prose ────────▶  the human-facing review
Flyte     ── durable, observable, fans tools out ───▶  the runtime underneath
```

## Layout

| Path | What it is |
|---|---|
| `battery.py` | The typed battery, the composition rules, the routing thresholds, the backend tools, and eight hand-labelled pull requests |
| `system2.py` | The generative client: writes reviews, and — for the baseline arm — answers the same battery as JSON |
| `agents.py` | Three pipelines in increasing order of how much the model decides: `guard_review`, `plan_and_execute`, `durable_review` |
| `benchmark.py` | The experiment: the same battery and the same routing code, with and without System 1 |
| `_runtime.py` | The shared task environments |

## Run

```bash
# The three pipelines
flyte run agents.py guard_review
flyte run agents.py plan_and_execute
flyte run agents.py durable_review --case_id c4

# The A/B
flyte run benchmark.py run_benchmark
flyte run benchmark.py run_benchmark --repeats 3 --num_cases 8
```

## Secrets

| Secret | Used for |
|---|---|
| `TYPESAFE_API_KEY` | System 1 — the TypeSafe API |
| `ANTHROPIC_API_KEY` | System 2 — the generative model |

```bash
flyte create secret TYPESAFE_API_KEY --value <your key>
flyte create secret ANTHROPIC_API_KEY --value <your key>
```

## The four patterns this is built around

1. **Speculative fan-out.** Every question goes in one call, including the ones
   you will not branch on — adding questions barely changes the response time,
   so the eleventh question is nearly free.
2. **Atomic decomposition, verdict in code.** Jev is never asked "what is the
   verdict?". It is asked one question per symptom, and `compose()` turns the
   symptoms into a verdict with a precedence rule you can read and unit-test.
3. **Composite scoring.** Severity and the tool plan are both composed from the
   symptoms rather than asked for directly.
4. **Confidence-gated routing.** `auto` / `review` / `escalate`, with thresholds
   that scale with risk. Escalation is a real abstention: the pipeline stops and
   hands over, and never spends a generation on a decision it is not sure about.
