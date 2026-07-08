# Customer-Support & Field-Service Resolution Agent

A Flyte v2 agent that helps resolve support tickets that require looking
**beyond the internal knowledge base** — verifying a vendor's current return
policy, checking whether a flight is delayed due to a public weather event,
confirming a vehicle recall, or pulling a manufacturer's latest spec sheet — and
drafts a customer-ready reply with sources a human agent can paste in.

## Why You.com

Support agents need to ground answers in **fresh, public, citable sources**, not
in a fine-tuned model that may be six months out of date. The You.com
**Research API** gives the agent a directly usable, cited synthesis; the
**Search API** provides transparent fallback grounding. Union's reusable,
warm-container architecture means this runs as a low-latency tool layer suitable
for real-time, human-in-the-loop support flows — not just batch.

## Requirements / spec

- **Inputs**
  - `tickets: list[Ticket]` — each ticket has a `ticket_id`, a `question`, and
    optional product/vendor `context` (e.g. "Customer bought a DeWalt DCD777
    drill from Home Depot").
  - `research_effort: str` — You.com Research depth. Default `lite` for
    low-latency, human-in-the-loop use.
- **Pipeline**
  1. `ground_answer`: calls the You.com Research API (low effort for speed)
     with the ticket + context to get a grounded, cited answer.
  2. `draft_reply`: calls Claude (`claude-haiku-4-5`) to turn the grounded
     answer into a concise, friendly, customer-ready reply that **cites its
     sources inline** so the human agent can verify before sending.
  3. `support_resolution` (driver): fans out across all tickets with
     `asyncio.gather` (each ticket runs `ground_answer` → `draft_reply`) and
     renders a Flyte report with every draft reply and its sources.
- **Outputs**
  - `ResolutionReport` — a draft reply plus the list of sources for each ticket.
  - An HTML Flyte report.
- **Durability / latency**
  - `@flyte.trace` on every external call for lineage.
  - Designed around a fast, single-pass research call so it can run inline in a
    human-in-the-loop flow.

## Secrets

| Secret key | Env var | Purpose |
| --- | --- | --- |
| `youdotcom-api-key` | `YDC_API_KEY` | You.com Research + Search APIs |
| `internal-anthropic-api-key` | `ANTHROPIC_API_KEY` | Claude via LiteLLM |

## Run it

```bash
uv run --script main.py
```
