# Continuous Competitive & Market Intelligence Agent

A Flyte v2 agent that runs continuously across a set of competitors, vendor
categories, or market signals and writes structured **deltas** (repositioning,
new feature launches, pricing changes, funding events, leadership moves) into a
knowledge-graph-ready table.

This is a [Dragonfly](https://www.union.ai/case-study)-style agent: it fans out
many small, iterative search loops and converges on the same external sources
across runs. Flyte's cross-run caching de-duplicates redundant You.com calls
when parallel runs hit the same query, which is exactly the cost-control pattern
Dragonfly built by hand.

## Why You.com

A competitive-intelligence knowledge graph must be populated with **fresh web
data that carries attributable sources** — not training-data recall, and not
brittle consumer SERP scraping. The You.com **Search API** returns ranked,
structured web *and* news results with snippets and publication timestamps; the
agent feeds those into an LLM that emits structured, source-cited deltas.

## Requirements / spec

- **Inputs**
  - `competitors: list[str]` — names of competitors/products/vendors to watch.
  - `categories: list[str]` — signal categories to watch (default: pricing,
    features, funding, leadership, partnerships).
  - `freshness: str` — recency window for You.com Search (`day`, `week`,
    `month`, `year`). Default `week`.
- **Pipeline**
  1. `watch_competitor` (fan-out, one task per competitor):
     - Calls the You.com Search API with a category-scoped query and the
       freshness filter, collecting web + news results with timestamps.
     - Calls Claude (`claude-haiku-4-5`) to extract a list of structured
       deltas, each with a category, summary, source URL, source date, and
       confidence score.
  2. `competitive_intelligence` (driver): fans out across competitors with
     `asyncio.gather`, aggregates all deltas, and renders a Flyte report.
- **Outputs**
  - `list[Delta]` — the structured deltas (knowledge-graph rows).
  - An HTML Flyte report grouping deltas by competitor and category.
- **Durability / cost control**
  - Tasks use `cache="auto"` so converging parallel/repeat runs reuse prior
    You.com + LLM results instead of paying for duplicate external calls.
  - `@flyte.trace` wraps every You.com and LLM call so the full
    prompt → query → source lineage is captured.

## Secrets

| Secret key | Env var | Purpose |
| --- | --- | --- |
| `youdotcom-api-key` | `YOU_API_KEY` | You.com Search API |
| `internal-anthropic-api-key` | `ANTHROPIC_API_KEY` | Claude via LiteLLM |

## Run it

```bash
uv run --script main.py
```

Or with the Flyte CLI:

```bash
flyte run main.py competitive_intelligence \
  --competitors '["Anthropic", "OpenAI"]'
```
