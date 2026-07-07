# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.4.0",
#     "httpx>=0.27.0",
#     "litellm>=1.72.0",
# ]
# main = "field_data_enrichment"
# params = ""
# ///
"""Autonomous systems & field-data enrichment agent.

Enriches geo-tagged operational events with real-world public context (road
closures, weather, incidents) using the You.com Search API with country +
freshness targeting, then uses Claude to summarize the relevant context. Only
public-web grounding queries leave the customer's cloud, never operational data.
"""

# {{docs-fragment env}}
import asyncio
import json
import os
from dataclasses import dataclass, field

import flyte

MODEL = "anthropic/claude-haiku-4-5"

env = flyte.TaskEnvironment(
    name="field-data-enrichment",
    secrets=[
        flyte.Secret(key="youdotcom-api-key", as_env_var="YDC_API_KEY"),
        flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    ],
    image=flyte.Image.from_uv_script(__file__, name="field-data-enrichment", pre=True),
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    cache="auto",
)
# {{/docs-fragment env}}


# {{docs-fragment data_types}}
@dataclass
class GeoEvent:
    event_id: str
    location: str
    country: str
    event_type: str


@dataclass
class Incident:
    description: str
    source_url: str
    published: str
    domain: str = ""
    author: str = ""
    favicon: str = ""
    snippet: str = ""
    section: str = "web"


@dataclass
class EnrichedEvent:
    event_id: str
    location: str
    context_summary: str
    severity: str
    incidents: list[Incident] = field(default_factory=list)


@dataclass
class EnrichmentReport:
    events: list[EnrichedEvent] = field(default_factory=list)
# {{/docs-fragment data_types}}


# {{docs-fragment you_search}}
YOU_SEARCH_URL = "https://ydc-index.io/v1/search"


@dataclass
class SearchHit:
    title: str
    url: str
    domain: str
    snippet: str
    published: str
    author: str
    favicon: str
    section: str


def _domain(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _favicon(item: dict, url: str) -> str:
    return item.get("favicon_url") or (
        f"https://ydc-index.io/favicon?domain={_domain(url)}&size=128"
    )


async def _you_get(url: str, params: dict, timeout: float = 60.0) -> dict:
    """GET with exponential backoff + jitter on 429 rate limits."""
    import asyncio
    import random

    import httpx

    # YDC_API_KEY is canonical; YOU_API_KEY accepted as a backwards-compatible fallback.
    headers = {"X-API-Key": os.environ.get("YDC_API_KEY") or os.environ["YOU_API_KEY"]}
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(7):
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code == 429 and attempt < 6:
                wait = float(resp.headers.get("retry-after") or 0) or min(2**attempt, 30)
                await asyncio.sleep(wait + random.uniform(0, 2))
                continue
            resp.raise_for_status()
            return resp.json()
    resp.raise_for_status()
    return resp.json()


@flyte.trace
async def you_search(
    query: str, country: str, freshness: str = "day", count: int = 8
) -> list[SearchHit]:
    """Search the public web + news for context near a geofenced location."""
    params = {
        "query": query,
        "count": count,
        "freshness": freshness,
        "country": country,
    }
    data = await _you_get(YOU_SEARCH_URL, params)

    results = data.get("results", {})
    hits: list[SearchHit] = []
    for section in ("news", "web"):
        for item in results.get(section, []) or []:
            snippets = item.get("snippets") or []
            url = item.get("url", "")
            hits.append(
                SearchHit(
                    title=item.get("title", ""),
                    url=url,
                    domain=_domain(url),
                    snippet=(snippets[0] if snippets else item.get("description", "")),
                    published=item.get("page_age", "") or "",
                    author=", ".join(item.get("authors") or []),
                    favicon=_favicon(item, url),
                    section=section,
                )
            )
    return hits
# {{/docs-fragment you_search}}


# {{docs-fragment llm}}
@flyte.trace
async def llm_json(system: str, user: str) -> dict:
    from litellm import acompletion

    resp = await acompletion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=1536,
    )
    parsed = _parse_json(resp.choices[0].message.content)
    return parsed if isinstance(parsed, dict) else {}


def _parse_json(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=0)
    end = max(text.rfind("}"), text.rfind("]")) + 1
    return json.loads(text[start:end])
# {{/docs-fragment llm}}


ENRICH_SYSTEM = """You are an operational-context analyst for autonomous and \
field systems. Given fresh local search results near a geofenced location, \
summarize the real-world context relevant to operations, extract discrete \
incidents (road closures, weather events, regulatory/airspace changes, local \
incidents), and assign an operational severity of 'none', 'low', 'medium', or \
'high'. Each incident must reference the supporting search result by its index. \
Respond ONLY with JSON:
{"context_summary": str, "severity": str, "incidents": [{"description": str, \
"source_index": int (the [n] of the supporting search result)}]}"""


# {{docs-fragment enrich_event}}
@env.task(retries=3)
async def enrich_event(event: GeoEvent, freshness: str) -> EnrichedEvent:
    """Ground one geo-tagged event in fresh public context."""
    query = f"{event.location} {event.event_type.replace('_', ' ')} road closure weather incident"
    hits = await you_search(query, country=event.country, freshness=freshness)

    evidence = "\n\n".join(
        f"[{i + 1}] {h.title} ({h.published}) — {h.domain}\n{h.url}\n{h.snippet}"
        for i, h in enumerate(hits)
    )
    user = (
        f"Location: {event.location}\n"
        f"Event type: {event.event_type}\n\n"
        f"Search results:\n{evidence or 'No results.'}"
    )
    parsed = await llm_json(ENRICH_SYSTEM, user)

    def _incident(it: dict) -> Incident:
        idx = int(it.get("source_index", 0) or 0)
        src = hits[idx - 1] if 1 <= idx <= len(hits) else None
        return Incident(
            description=str(it.get("description", "")),
            source_url=src.url if src else "",
            published=src.published if src else "",
            domain=src.domain if src else "",
            author=src.author if src else "",
            favicon=src.favicon if src else "",
            snippet=src.snippet if src else "",
            section=src.section if src else "web",
        )

    incidents = [_incident(it) for it in (parsed.get("incidents", []) or [])]
    return EnrichedEvent(
        event_id=event.event_id,
        location=event.location,
        context_summary=str(parsed.get("context_summary", "")),
        severity=str(parsed.get("severity", "none")),
        incidents=incidents,
    )
# {{/docs-fragment enrich_event}}


# {{docs-fragment report}}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "none": 3}
_SEVERITY_STYLE = {
    "high": ("#fdecea", "#c0392b"),
    "medium": ("#fdf3e1", "#b7791f"),
    "low": ("#e3f1fb", "#2b6cb0"),
    "none": ("#eef1f4", "#627d98"),
}

REPORT_CSS = """
<style>
  .rpt { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
         Helvetica, Arial, sans-serif; color:#1f2933; max-width:1040px;
         margin:0 auto; }
  .rpt h1 { font-size:22px; margin:0 0 4px; color:#102a43; }
  .rpt .sub { color:#647488; font-size:13px; margin:0 0 18px; }
  .rpt .stats { display:flex; gap:10px; flex-wrap:wrap; margin:0 0 22px; }
  .rpt .pill { background:#f0f4f8; border-radius:999px; padding:6px 14px;
               font-size:13px; color:#334e68; }
  .rpt .pill b { color:#102a43; }
  .rpt .card { border:1px solid #e4e7eb; border-radius:12px; padding:16px 18px;
               margin:0 0 14px; box-shadow:0 1px 3px rgba(16,42,67,0.06);
               background:#fff; border-left:4px solid #cbd2d9; }
  .rpt .card.high { border-left-color:#c0392b; }
  .rpt .card.medium { border-left-color:#b7791f; }
  .rpt .card.low { border-left-color:#2b6cb0; }
  .rpt .card h2 { font-size:15px; margin:0 0 6px; color:#102a43; }
  .rpt .sev { display:inline-block; font-size:11px; font-weight:700;
              padding:3px 9px; border-radius:6px; text-transform:uppercase;
              letter-spacing:.03em; margin-right:8px; }
  .rpt .loc { font-size:13px; color:#52606d; }
  .rpt .summary { margin:8px 0; font-size:14px; line-height:1.45; }
  .rpt .inc { font-size:13px; color:#334e68; padding:6px 0; }
  .rpt .meta { color:#829ab1; font-size:12px; }
  .rpt a { color:#2b6cb0; text-decoration:none; }
  .rpt a:hover { text-decoration:underline; }
  .rpt .empty { color:#829ab1; font-style:italic; padding:8px 0; }
  .rpt .cite { display:flex; gap:9px; align-items:flex-start; background:#f7f9fb;
               border:1px solid #eef1f4; border-radius:8px; padding:7px 10px;
               margin:5px 0 2px 14px; }
  .rpt .cite img.fav { width:15px; height:15px; border-radius:3px; margin-top:2px;
                       flex:0 0 auto; background:#e4e7eb; }
  .rpt .cite .cb { font-size:12px; line-height:1.4; }
  .rpt .cite .cdom { font-weight:600; color:#334e68; }
  .rpt .cite .ctag { font-size:10px; font-weight:700; text-transform:uppercase;
                     color:#fff; background:#bcccdc; border-radius:4px;
                     padding:1px 5px; margin-left:6px; }
  .rpt .cite .ctag.news { background:#e8833a; }
  .rpt .cite .cmeta { color:#829ab1; }
  .rpt .cite .csnip { color:#52606d; font-style:italic; margin-top:2px; }
  .rpt .yoube { font-size:11px; color:#9aa5b1; margin-top:4px; }
</style>
"""


def _sev_badge(sev: str) -> str:
    bg, fg = _SEVERITY_STYLE.get(sev, ("#eef1f4", "#627d98"))
    return f"<span class='sev' style='background:{bg};color:{fg}'>{sev}</span>"


def _cite(it: Incident) -> str:
    """Render a rich You.com citation for an incident's supporting source."""
    if not it.source_url:
        return ""
    tag = (
        "<span class='ctag news'>news</span>"
        if it.section == "news"
        else "<span class='ctag'>web</span>"
    )
    meta_bits = []
    if it.published:
        meta_bits.append(it.published[:10])
    if it.author:
        meta_bits.append(f"by {it.author}")
    meta = " &middot; ".join(meta_bits)
    snip = f"<div class='csnip'>&ldquo;{it.snippet}&rdquo;</div>" if it.snippet else ""
    return (
        f"<div class='cite'><img class='fav' src='{it.favicon}' alt=''/>"
        f"<div class='cb'>"
        f"<a href='{it.source_url}'><span class='cdom'>{it.domain or 'source'}</span></a>{tag}"
        f"<div class='cmeta'>{meta}</div>{snip}</div></div>"
    )


def _render_report(report: EnrichmentReport) -> str:
    events = sorted(report.events, key=lambda e: _SEVERITY_ORDER.get(e.severity, 4))
    flagged = sum(1 for e in events if e.severity in ("high", "medium"))
    total_sources = sum(len(e.incidents) for e in events)

    cards = []
    for e in events:
        incidents = "".join(
            f"<div class='inc'>&bull; {it.description}{_cite(it)}</div>"
            for it in e.incidents
        )
        cards.append(
            f"<div class='card {e.severity}'>"
            f"<div>{_sev_badge(e.severity)}"
            f"<span class='loc'><b>{e.event_id}</b> &middot; {e.location}</span></div>"
            f"<div class='summary'>{e.context_summary or 'No relevant public context found.'}</div>"
            f"{incidents}</div>"
        )

    return f"""
    {REPORT_CSS}
    <div class="rpt">
      <h1>Field-Data Enrichment</h1>
      <p class="sub">Geo-tagged events grounded in fresh public context — each
      incident cites a timestamped You.com Search result.</p>
      <div class="stats">
        <span class="pill"><b>{len(events)}</b> events</span>
        <span class="pill" style="background:#fdecea;color:#c0392b">
          <b>{flagged}</b> flagged (high/medium)</span>
        <span class="pill"><b>{total_sources}</b> cited You.com sources</span>
      </div>
      {''.join(cards) or "<p class='empty'>No events processed.</p>"}
      <p class="yoube">Public context retrieved via the You.com Search API with
      country + freshness targeting. Operational data never leaves the BYOC
      boundary — only public-web queries go out.</p>
    </div>
    """
# {{/docs-fragment report}}


# {{docs-fragment driver}}
DEFAULT_EVENTS = [
    GeoEvent("evt-1", "Mountain View, CA", "US", "road_closure_check"),
    GeoEvent("evt-2", "Tokyo, Japan", "JP", "weather"),
    GeoEvent("evt-3", "Austin, TX", "US", "road_closure_check"),
    GeoEvent("evt-4", "Phoenix, AZ", "US", "weather"),
    GeoEvent("evt-5", "London, UK", "GB", "incident"),
    GeoEvent("evt-6", "San Francisco, CA", "US", "incident"),
    GeoEvent("evt-7", "Seattle, WA", "US", "weather"),
    GeoEvent("evt-8", "Miami, FL", "US", "weather"),
    GeoEvent("evt-9", "Denver, CO", "US", "road_closure_check"),
    GeoEvent("evt-10", "Berlin, Germany", "DE", "incident"),
]


@env.task(report=True)
async def field_data_enrichment(
    events: list[GeoEvent] = DEFAULT_EVENTS,
    freshness: str = "day",
) -> EnrichmentReport:
    """Fan out across geo-tagged events and enrich each with public context."""
    with flyte.group("enrich-events"):
        enriched = await asyncio.gather(
            *[enrich_event(e, freshness) for e in events]
        )

    report = EnrichmentReport(events=list(enriched))
    await flyte.report.replace.aio(_render_report(report), do_flush=True)
    await flyte.report.flush.aio()
    return report
# {{/docs-fragment driver}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(field_data_enrichment)
    print(run.url)
    run.wait()
# {{/docs-fragment main}}
