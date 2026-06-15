# Autonomous Systems & Field-Data Enrichment Agent

A Flyte v2 agent that enriches geo-tagged operational events — from autonomous
vehicles, aircraft, satellites, or field sensors — with **real-world public
context**: road closures, weather events, airspace/regulatory changes, ESG
reporting requirements, or local incidents tied to a geofence.

In Union's BYOC deployment model the sensitive operational data never leaves the
customer's cloud — only the lightweight, public-web grounding queries go out to
You.com.

## Why You.com

You.com's **Search API** (with unified web + news results, `freshness`, and
`country` targeting) lets an agent ground a geo-tagged event in current public
information without building and maintaining a per-customer crawler.

## Requirements / spec

- **Inputs**
  - `events: list[GeoEvent]` — each event has:
    - `event_id: str`
    - `location: str` — human-readable place (e.g. "Mountain View, CA").
    - `country: str` — ISO 3166-1 alpha-2 code for geo-targeting (e.g. `US`).
    - `event_type: str` — e.g. `road_closure_check`, `weather`, `incident`.
  - `freshness: str` — recency window (default `day`).
- **Pipeline**
  1. `enrich_event` (fan-out, one task per event):
     - Builds a location- and type-scoped query and calls the You.com Search
       API with `freshness` + `country`, collecting web + news results.
     - Calls Claude (`claude-haiku-4-5`) to summarize the relevant real-world
       context, extract discrete incidents, and assign an operational
       `severity` — all grounded in the returned sources.
  2. `field_data_enrichment` (driver): fans out across events with
     `asyncio.gather`, aggregates enriched events, and renders a Flyte report.
- **Outputs**
  - `list[EnrichedEvent]` — each with a context summary, extracted incidents,
    severity, and source citations.
  - An HTML Flyte report.
- **Durability / cost control**
  - `cache="auto"` so repeated geofence checks within the cache window reuse
    prior results.
  - `@flyte.trace` on every external call for lineage.

## Secrets

| Secret key | Env var | Purpose |
| --- | --- | --- |
| `youdotcom-api-key` | `YOU_API_KEY` | You.com Search API |
| `internal-anthropic-api-key` | `ANTHROPIC_API_KEY` | Claude via LiteLLM |

## Run it

```bash
uv run --script main.py
```
