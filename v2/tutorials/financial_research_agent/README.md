# Financial Research & Earnings-Cycle Agent

A Flyte v2 agent that preps equity briefings and competitor benchmarks for the
earnings cycle. For each company it runs grounded, source-cited research and
fresh news, then synthesizes an analyst-ready briefing.

## Why You.com

Financial research demands **low-latency, ranked, source-cited results** across
both the general web and news streams. The You.com **Research API** produces a
grounded, citation-backed synthesis that a downstream LLM can turn into a
narrative briefing; the **Search API** adds a fresh-news layer. At $2–3 per
agent run and millions of runs per quarter, customers need a predictable,
enterprise-priced search backend — which fits Union's tiered, cost-controlled
task environment. Flyte caching further cuts duplicate spend when runs converge.

## Requirements / spec

- **Inputs**
  - `companies: list[str]` — companies/tickers to research.
  - `focus: str` — briefing angle (e.g. "Q4 earnings preview and competitive
    positioning").
  - `research_effort: str` — You.com Research depth (`lite`, `standard`,
    `deep`). Default `standard`.
- **Pipeline**
  1. `research_company` (fan-out, one task per company):
     - Calls the You.com Research API with the focus-scoped question and a
       `source_control` freshness window to get a grounded, cited answer.
     - Calls the You.com Search API for the latest news headlines.
     - Calls Claude (`claude-haiku-4-5`) to synthesize a structured equity
       briefing (thesis, recent developments, risks, watch items) grounded in
       the research answer + news, preserving citations.
  2. `financial_research` (driver): fans out across companies, aggregates the
     briefings, and renders a Flyte report.
- **Outputs**
  - `list[Briefing]` — structured briefings with citations.
  - An HTML Flyte report.
- **Durability / cost control**
  - `cache="auto"` so repeated/converging runs reuse prior research.
  - `@flyte.trace` on every external call for full prompt → citation lineage.

## Secrets

| Secret key | Env var | Purpose |
| --- | --- | --- |
| `youdotcom-api-key` | `YDC_API_KEY` | You.com Research + Search APIs |
| `internal-anthropic-api-key` | `ANTHROPIC_API_KEY` | Claude via LiteLLM |

## Run it

```bash
uv run --script main.py
```
