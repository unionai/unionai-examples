# Regulatory & Compliance Monitoring Agent

A long-running Flyte v2 agent that watches for changes in regulatory sources —
FDA guidance, SEC filings, state-level privacy laws, clinical-trial registries,
sanctions lists — and routes structured, **citation-precise** findings to the
right downstream team (compliance, legal, or clinical ops).

## Why You.com

Compliance use cases live or die on **citation precision and recency**. A
hallucinated regulatory citation isn't a bug, it's a liability. The You.com
**Research API** returns a grounded, synthesized answer *plus* structured
sources (URL + title + snippet), and `source_control` lets the agent restrict
research to trusted government/regulator domains and a recency window. Combined
with Flyte's audit lineage, customers get end-to-end traceability: Flyte logs
which agent issued which You.com query and got which document on which date —
a story no SERP scraper or LLM-only stack can credibly tell.

## Requirements / spec

- **Inputs**
  - `watch_items: list[WatchItem]` — each item has:
    - `topic: str` — what to monitor (e.g. "FDA guidance on AI/ML-enabled
      medical device software").
    - `trusted_domains: list[str]` — allowlist for `source_control`
      (e.g. `["fda.gov", "federalregister.gov"]`).
    - `team: str` — routing destination (`compliance`, `legal`, `clinical`).
  - `freshness: str` — recency filter for `source_control` (default `month`).
- **Pipeline**
  1. `monitor_watch_item` (fan-out, one task per watch item):
     - Calls the You.com Research API (`research_effort="standard"`) with
       `source_control` restricting to the trusted domains + freshness window,
       and an `output_schema` requesting structured findings.
     - Each finding carries a `source_url`, `published_date`, and `snippet` so
       the citation can be verified.
     - Calls Claude (`claude-haiku-4-5`) to assign a `severity`
       (`info`/`watch`/`action`) and a short routing rationale.
  2. `compliance_monitoring` (driver): fans out across watch items, aggregates
     findings, and renders a Flyte report grouped by team and severity.
- **Outputs**
  - `list[Finding]` — structured, citation-bearing findings with severity and
    routing.
  - An HTML Flyte report.
- **Durability / lineage**
  - `@flyte.trace` wraps every You.com Research and LLM call so the external
    data layer is captured in Flyte's run lineage.
  - `retries` are configured on the monitoring task for robustness.

## Secrets

| Secret key | Env var | Purpose |
| --- | --- | --- |
| `youdotcom-api-key` | `YDC_API_KEY` | You.com Research API |
| `internal-anthropic-api-key` | `ANTHROPIC_API_KEY` | Claude via LiteLLM |

## Run it

```bash
uv run --script main.py
```
