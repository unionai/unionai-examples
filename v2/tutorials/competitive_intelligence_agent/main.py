# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.4.0",
#     "httpx>=0.27.0",
#     "litellm>=1.72.0",
# ]
# main = "competitive_intelligence"
# params = ""
# ///
"""Continuous competitive & market intelligence agent.

A Dragonfly-style agent that fans out across competitors, pulls fresh,
source-cited web + news results from the You.com Search API, and uses Claude to
extract structured "deltas" (pricing, features, funding, leadership, etc.) into
a knowledge-graph-ready table.
"""

# {{docs-fragment env}}
import asyncio
import json
from dataclasses import dataclass, field

import flyte

MODEL = "anthropic/claude-haiku-4-5"

env = flyte.TaskEnvironment(
    name="competitive-intelligence",
    secrets=[
        flyte.Secret(key="youdotcom-api-key", as_env_var="YOU_API_KEY"),
        flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    ],
    image=flyte.Image.from_uv_script(__file__, name="competitive-intelligence", pre=True),
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    cache="auto",
)
# {{/docs-fragment env}}


# {{docs-fragment data_types}}
@dataclass
class SearchHit:
    """A You.com Search result with its full structured metadata."""

    title: str
    url: str
    domain: str
    snippet: str
    published: str  # You.com page_age timestamp
    author: str
    favicon: str  # You.com favicon_url
    thumbnail: str
    section: str  # "news" or "web" — You.com's auto classification


@dataclass
class Delta:
    competitor: str
    category: str
    summary: str
    confidence: float
    source: SearchHit | None = None


@dataclass
class CompetitorWatch:
    competitor: str
    deltas: list[Delta] = field(default_factory=list)
    sources: list[SearchHit] = field(default_factory=list)


@dataclass
class IntelReport:
    watches: list[CompetitorWatch] = field(default_factory=list)

    @property
    def deltas(self) -> list[Delta]:
        return [d for w in self.watches for d in w.deltas]
# {{/docs-fragment data_types}}


# {{docs-fragment you_search}}
YOU_SEARCH_URL = "https://ydc-index.io/v1/search"


async def _you_get(url: str, params: dict, timeout: float = 60.0) -> dict:
    """GET with exponential backoff + jitter on 429 rate limits."""
    import asyncio
    import os
    import random

    import httpx

    headers = {"X-API-Key": os.environ["YOU_API_KEY"]}
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


@flyte.trace
async def you_search(query: str, count: int = 8, freshness: str = "week") -> list[SearchHit]:
    """Call the You.com Search API and return unified web + news hits."""
    params = {"query": query, "count": count, "freshness": freshness}
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
                    thumbnail=item.get("thumbnail_url", "") or "",
                    section=section,
                )
            )
    return hits
# {{/docs-fragment you_search}}


# {{docs-fragment llm}}
@flyte.trace
async def llm_json(system: str, user: str) -> dict | list:
    """Call Claude via LiteLLM and parse a JSON response."""
    from litellm import acompletion

    resp = await acompletion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    content = resp.choices[0].message.content
    return _parse_json(content)


def _parse_json(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=0,
    )
    end = max(text.rfind("}"), text.rfind("]")) + 1
    return json.loads(text[start:end])
# {{/docs-fragment llm}}


EXTRACT_SYSTEM = """You are a competitive-intelligence analyst. Given fresh \
search results about a competitor, extract concrete, recently-changed signals \
("deltas") in the requested categories. Only report changes that are supported \
by a specific search result. Respond with a JSON object of the form:
{"deltas": [{"category": str, "summary": str, "source_index": int (the [n] of \
the supporting search result), "confidence": float between 0 and 1}]}
If there are no clear changes, return {"deltas": []}."""


# {{docs-fragment watch_competitor}}
@env.task(retries=3)
async def watch_competitor(
    competitor: str,
    categories: list[str],
    freshness: str,
) -> CompetitorWatch:
    """Search for fresh signals on one competitor and extract structured deltas."""
    query = (
        f"{competitor} "
        + " OR ".join(categories)
        + " announcement OR news OR update"
    )
    hits = await you_search(query, count=8, freshness=freshness)
    if not hits:
        return CompetitorWatch(competitor=competitor)

    evidence = "\n\n".join(
        f"[{i + 1}] {h.title} ({h.published}) — {h.domain}\n{h.url}\n{h.snippet}"
        for i, h in enumerate(hits)
    )
    user = (
        f"Competitor: {competitor}\n"
        f"Categories to watch: {', '.join(categories)}\n\n"
        f"Search results:\n{evidence}"
    )
    parsed = await llm_json(EXTRACT_SYSTEM, user)
    raw_deltas = parsed.get("deltas", []) if isinstance(parsed, dict) else []

    deltas: list[Delta] = []
    cited: list[SearchHit] = []
    for d in raw_deltas:
        idx = int(d.get("source_index", 0) or 0)
        src = hits[idx - 1] if 1 <= idx <= len(hits) else None
        if src is not None and src not in cited:
            cited.append(src)
        deltas.append(
            Delta(
                competitor=competitor,
                category=str(d.get("category", "unknown")),
                summary=str(d.get("summary", "")),
                confidence=float(d.get("confidence", 0.0) or 0.0),
                source=src,
            )
        )
    return CompetitorWatch(competitor=competitor, deltas=deltas, sources=cited)
# {{/docs-fragment watch_competitor}}


# {{docs-fragment report}}
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
               background:#fff; }
  .rpt .card h2 { font-size:16px; margin:0 0 6px; color:#102a43; }
  .rpt .row { padding:11px 0; border-top:1px solid #f0f2f5; }
  .rpt .row:first-of-type { border-top:none; }
  .rpt .chip { display:inline-block; font-size:11px; font-weight:600;
               padding:3px 9px; border-radius:6px; white-space:nowrap;
               text-transform:uppercase; letter-spacing:.03em;
               background:#e0e8f9; color:#2b4ba0; margin-right:8px; }
  .rpt .summary { margin:6px 0 4px; font-size:14px; line-height:1.45; }
  .rpt .meta { color:#829ab1; font-size:12px; }
  .rpt a { color:#2b6cb0; text-decoration:none; }
  .rpt a:hover { text-decoration:underline; }
  .rpt .bar { display:inline-block; width:60px; height:6px; border-radius:3px;
              background:#e4e7eb; vertical-align:middle; overflow:hidden;
              margin-right:6px; }
  .rpt .bar > span { display:block; height:100%; background:#3ebd93; }
  .rpt .empty { color:#829ab1; font-style:italic; padding:8px 0; }
  .rpt .cite { display:flex; gap:9px; align-items:flex-start; background:#f7f9fb;
               border:1px solid #eef1f4; border-radius:8px; padding:8px 10px;
               margin-top:8px; }
  .rpt .cite img.fav { width:16px; height:16px; border-radius:3px; margin-top:2px;
                       flex:0 0 auto; background:#e4e7eb; }
  .rpt .cite .cb { font-size:12px; line-height:1.45; }
  .rpt .cite .cdom { font-weight:600; color:#334e68; }
  .rpt .cite .ctag { font-size:10px; font-weight:700; text-transform:uppercase;
                     color:#fff; background:#bcccdc; border-radius:4px;
                     padding:1px 5px; margin-left:6px; }
  .rpt .cite .ctag.news { background:#e8833a; }
  .rpt .cite .cmeta { color:#829ab1; }
  .rpt .cite .csnip { color:#52606d; font-style:italic; margin-top:3px; }
  .rpt .src-head { font-size:11px; text-transform:uppercase; letter-spacing:.04em;
                   color:#627d98; margin:14px 0 4px; }
  .rpt .yoube { font-size:11px; color:#9aa5b1; margin-top:4px; }
</style>
"""


def _conf_bar(conf: float) -> str:
    pct = max(0, min(100, int(conf * 100)))
    return (
        f"<span class='bar'><span style='width:{pct}%'></span></span>"
        f"<span class='meta'>{conf:.0%} confidence</span>"
    )


def _cite(src: SearchHit) -> str:
    """Render a rich You.com citation: favicon, domain, date, author, snippet."""
    if src is None:
        return ""
    tag = (
        f"<span class='ctag news'>news</span>"
        if src.section == "news"
        else "<span class='ctag'>web</span>"
    )
    meta_bits = []
    if src.published:
        meta_bits.append(src.published[:10])
    if src.author:
        meta_bits.append(f"by {src.author}")
    meta = " &middot; ".join(meta_bits)
    snip = f"<div class='csnip'>&ldquo;{src.snippet}&rdquo;</div>" if src.snippet else ""
    return (
        f"<div class='cite'>"
        f"<img class='fav' src='{src.favicon}' alt=''/>"
        f"<div class='cb'>"
        f"<a href='{src.url}'><span class='cdom'>{src.domain or 'source'}</span></a>{tag}"
        f"<div class='cmeta'>{meta}</div>{snip}</div></div>"
    )


def _render_report(report: IntelReport) -> str:
    watches = sorted(report.watches, key=lambda w: w.competitor)
    total_sources = sum(len(w.sources) for w in watches)

    cards = []
    for w in watches:
        deltas = sorted(w.deltas, key=lambda d: -d.confidence)
        rows = "".join(
            f"<div class='row'><span class='chip'>{d.category}</span>"
            f"<div class='summary'>{d.summary}</div>"
            f"{_conf_bar(d.confidence)}"
            f"{_cite(d.source)}"
            "</div>"
            for d in deltas
        )
        cards.append(
            f"<div class='card'><h2>{w.competitor}</h2>"
            f"<span class='meta'>{len(deltas)} signal(s) &middot; "
            f"{len(w.sources)} You.com source(s)</span>{rows or ''}</div>"
        )

    return f"""
    {REPORT_CSS}
    <div class="rpt">
      <h1>Competitive Intelligence Deltas</h1>
      <p class="sub">Fresh, source-cited market signals — every delta links back
      to a ranked, timestamped You.com Search result.</p>
      <div class="stats">
        <span class="pill"><b>{len(report.deltas)}</b> signals</span>
        <span class="pill"><b>{len(watches)}</b> competitors tracked</span>
        <span class="pill"><b>{total_sources}</b> cited You.com sources</span>
      </div>
      {''.join(cards) or "<p class='empty'>No signals detected in this window.</p>"}
      <p class="yoube">Sources retrieved and ranked by the You.com Search API
      (web + auto-classified news), with publication timestamps, authors, and
      snippet provenance preserved for full prompt &rarr; citation lineage.</p>
    </div>
    """
# {{/docs-fragment report}}


# {{docs-fragment driver}}
@env.task(report=True)
async def competitive_intelligence(
    competitors: list[str] = [
        "Anthropic",
        "OpenAI",
        "Mistral AI",
        "Google DeepMind",
        "Cohere",
        "Perplexity AI",
        "xAI",
        "Hugging Face",
        "Databricks",
        "Together AI",
    ],
    categories: list[str] = [
        "pricing",
        "product launch",
        "model release",
        "funding",
        "leadership",
        "partnership",
    ],
    freshness: str = "week",
) -> IntelReport:
    """Fan out across competitors and aggregate structured deltas."""
    with flyte.group("watch-competitors"):
        results = await asyncio.gather(
            *[watch_competitor(c, categories, freshness) for c in competitors]
        )

    report = IntelReport(watches=list(results))

    await flyte.report.replace.aio(_render_report(report), do_flush=True)
    await flyte.report.flush.aio()
    return report
# {{/docs-fragment driver}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(competitive_intelligence)
    print(run.url)
    run.wait()
# {{/docs-fragment main}}
