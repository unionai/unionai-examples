# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.4.0",
#     "httpx>=0.27.0",
#     "litellm>=1.72.0",
# ]
# main = "financial_research"
# params = ""
# ///
"""Financial research & earnings-cycle agent.

For each company, runs grounded, source-cited research via the You.com Research
API plus a fresh-news layer via the Search API, then uses Claude to synthesize
an analyst-ready equity briefing that preserves citations. Flyte caching cuts
duplicate spend when runs converge.
"""

# {{docs-fragment env}}
import asyncio
import json
import os
from dataclasses import dataclass, field

import flyte

MODEL = "anthropic/claude-haiku-4-5"

env = flyte.TaskEnvironment(
    name="financial-research",
    secrets=[
        flyte.Secret(key="youdotcom-api-key", as_env_var="YOU_API_KEY"),
        flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    ],
    image=flyte.Image.from_uv_script(__file__, name="financial-research", pre=True),
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    cache="auto",
)
# {{/docs-fragment env}}


# {{docs-fragment data_types}}
@dataclass
class Source:
    title: str
    url: str
    domain: str = ""
    snippet: str = ""
    published: str = ""
    favicon: str = ""
    section: str = "research"  # "research", "news", or "web"


def _domain(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _favicon_for(url: str) -> str:
    return f"https://ydc-index.io/favicon?domain={_domain(url)}&size=128"


@dataclass
class Briefing:
    company: str
    thesis: str
    recent_developments: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    watch_items: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)


@dataclass
class ResearchReport:
    briefings: list[Briefing] = field(default_factory=list)
# {{/docs-fragment data_types}}


# {{docs-fragment you_apis}}
YOU_RESEARCH_URL = "https://api.you.com/v1/research"
YOU_SEARCH_URL = "https://ydc-index.io/v1/search"


async def _you_request(method: str, url: str, timeout: float, **kwargs) -> dict:
    """HTTP wrapper with exponential backoff + jitter on 429 rate limits.

    Fanned-out tasks run in separate pods, so we retry on the client side to
    smooth out bursts against the You.com API rate limit.
    """
    import asyncio
    import random

    import httpx

    headers = {"X-API-Key": os.environ["YOU_API_KEY"]}
    if method == "POST":
        headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(7):
            resp = await client.request(method, url, headers=headers, **kwargs)
            if resp.status_code == 429 and attempt < 6:
                wait = float(resp.headers.get("retry-after") or 0) or min(2**attempt, 30)
                await asyncio.sleep(wait + random.uniform(0, 2))
                continue
            resp.raise_for_status()
            return resp.json()
    resp.raise_for_status()
    return resp.json()


@flyte.trace
async def you_research(question: str, research_effort: str, freshness: str) -> dict:
    """Grounded, citation-backed research answer."""
    body = {
        "input": question,
        "research_effort": research_effort,
        "source_control": {"freshness": freshness},
    }
    return await _you_request("POST", YOU_RESEARCH_URL, 300.0, json=body)


@flyte.trace
async def you_news(query: str, count: int = 6, freshness: str = "week") -> list[dict]:
    """Fresh news headlines for a company."""
    params = {"query": query, "count": count, "freshness": freshness}
    data = await _you_request("GET", YOU_SEARCH_URL, 60.0, params=params)

    results = data.get("results", {})
    out: list[dict] = []
    for section in ("news", "web"):
        for item in results.get(section, []) or []:
            snippets = item.get("snippets") or []
            url = item.get("url", "")
            out.append(
                {
                    "title": item.get("title", ""),
                    "url": url,
                    "domain": _domain(url),
                    "snippet": snippets[0] if snippets else item.get("description", ""),
                    "published": item.get("page_age", "") or "",
                    "favicon": item.get("favicon_url")
                    or _favicon_for(url),
                    "section": section,
                }
            )
    return out
# {{/docs-fragment you_apis}}


# {{docs-fragment llm}}
@flyte.trace
async def synthesize_briefing(company: str, focus: str, research: str, news: str) -> dict:
    """Use Claude to synthesize a structured equity briefing."""
    from litellm import acompletion

    system = (
        "You are an equity research analyst. Using ONLY the grounded research "
        "and news provided, write a concise briefing. Respond ONLY with JSON: "
        '{"thesis": str, "recent_developments": [str], "risks": [str], '
        '"watch_items": [str]}. Keep each list to 3-5 short, specific bullets.'
    )
    user = (
        f"Company: {company}\nFocus: {focus}\n\n"
        f"Grounded research:\n{research}\n\nRecent news:\n{news}"
    )
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


# {{docs-fragment research_company}}
@env.task(retries=3)
async def research_company(
    company: str,
    focus: str,
    research_effort: str,
    freshness: str,
) -> Briefing:
    """Research one company and synthesize a cited briefing."""
    question = (
        f"Provide a grounded analysis of {company} with respect to: {focus}. "
        f"Cover recent financial performance, strategic moves, competitive "
        f"positioning, and risks."
    )
    research_result, news = await asyncio.gather(
        you_research(question, research_effort, freshness),
        you_news(f"{company} earnings news", freshness=freshness),
    )

    output = research_result.get("output", {})
    research_text = output.get("content", "")
    if not isinstance(research_text, str):
        research_text = json.dumps(research_text)

    sources: list[Source] = []
    for s in output.get("sources", []) or []:
        url = str(s.get("url", ""))
        sources.append(
            Source(
                title=str(s.get("title", "") or url),
                url=url,
                domain=_domain(url),
                snippet=str((s.get("snippets") or [""])[0]),
                favicon=_favicon_for(url),
                section="research",
            )
        )
    for n in news:
        sources.append(
            Source(
                title=str(n.get("title", "")),
                url=str(n.get("url", "")),
                domain=str(n.get("domain", "")),
                snippet=str(n.get("snippet", "")),
                published=str(n.get("published", "")),
                favicon=str(n.get("favicon", "")),
                section=str(n.get("section", "web")),
            )
        )
    news_text = "\n".join(
        f"- {n['title']} ({n['published']}) {n['domain']}: {n['snippet'][:120]}"
        for n in news
    )

    parsed = await synthesize_briefing(company, focus, research_text, news_text)

    def _list(key: str) -> list[str]:
        return [str(x) for x in (parsed.get(key) or [])]

    return Briefing(
        company=company,
        thesis=str(parsed.get("thesis", "")),
        recent_developments=_list("recent_developments"),
        risks=_list("risks"),
        watch_items=_list("watch_items"),
        sources=sources,
    )
# {{/docs-fragment research_company}}


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
  .rpt .card { border:1px solid #e4e7eb; border-radius:12px; padding:18px 20px;
               margin:0 0 16px; box-shadow:0 1px 3px rgba(16,42,67,0.06);
               background:#fff; }
  .rpt .card h2 { font-size:18px; margin:0 0 8px; color:#102a43; }
  .rpt .thesis { font-size:14px; line-height:1.5; background:#f7f9fb;
                 border-radius:8px; padding:10px 12px; margin:0 0 14px; }
  .rpt .cols { display:flex; gap:18px; flex-wrap:wrap; }
  .rpt .col { flex:1; min-width:220px; }
  .rpt .col h3 { font-size:12px; text-transform:uppercase; letter-spacing:.04em;
                 color:#627d98; margin:0 0 6px; }
  .rpt .col.risks h3 { color:#c0392b; }
  .rpt ul { margin:0; padding-left:18px; }
  .rpt li { font-size:13px; line-height:1.5; margin:0 0 4px; }
  .rpt .sources { margin-top:14px; border-top:1px solid #f0f2f5; padding-top:10px; }
  .rpt .sources h3 { font-size:12px; text-transform:uppercase; color:#627d98;
                     margin:0 0 8px; }
  .rpt a { color:#2b6cb0; text-decoration:none; }
  .rpt a:hover { text-decoration:underline; }
  .rpt .empty { color:#829ab1; font-style:italic; padding:8px 0; }
  .rpt .cite { display:flex; gap:9px; align-items:flex-start; background:#f7f9fb;
               border:1px solid #eef1f4; border-radius:8px; padding:7px 10px;
               margin:0 0 6px; }
  .rpt .cite img.fav { width:15px; height:15px; border-radius:3px; margin-top:2px;
                       flex:0 0 auto; background:#e4e7eb; }
  .rpt .cite .cb { font-size:12px; line-height:1.4; }
  .rpt .cite .cdom { font-weight:600; color:#334e68; }
  .rpt .cite .ctag { font-size:10px; font-weight:700; text-transform:uppercase;
                     color:#fff; background:#bcccdc; border-radius:4px;
                     padding:1px 5px; margin-left:6px; }
  .rpt .cite .ctag.research { background:#5b8def; }
  .rpt .cite .ctag.news { background:#e8833a; }
  .rpt .cite .cmeta { color:#829ab1; }
  .rpt .cite .csnip { color:#52606d; font-style:italic; margin-top:2px; }
  .rpt .yoube { font-size:11px; color:#9aa5b1; margin-top:4px; }
</style>
"""


def _cite(s: Source) -> str:
    """Render a rich You.com citation (Research or Search source)."""
    if not s.url:
        return ""
    tag_cls = s.section if s.section in ("research", "news") else "web"
    meta_bits = []
    if s.published:
        meta_bits.append(s.published[:10])
    if s.title:
        meta_bits.append(s.title)
    meta = " &middot; ".join(meta_bits)
    snip = f"<div class='csnip'>&ldquo;{s.snippet}&rdquo;</div>" if s.snippet else ""
    return (
        f"<div class='cite'><img class='fav' src='{s.favicon}' alt=''/>"
        f"<div class='cb'>"
        f"<a href='{s.url}'><span class='cdom'>{s.domain or 'source'}</span></a>"
        f"<span class='ctag {tag_cls}'>{s.section}</span>"
        f"<div class='cmeta'>{meta}</div>{snip}</div></div>"
    )


def _render_report(report: ResearchReport) -> str:
    def _ul(items: list[str]) -> str:
        if not items:
            return "<p class='empty'>None reported.</p>"
        return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"

    cards = []
    for b in report.briefings:
        src = "".join(_cite(s) for s in b.sources[:10])
        cards.append(
            f"<div class='card'><h2>{b.company}</h2>"
            f"<div class='thesis'>{b.thesis or 'No thesis generated.'}</div>"
            f"<div class='cols'>"
            f"<div class='col'><h3>Recent developments</h3>{_ul(b.recent_developments)}</div>"
            f"<div class='col risks'><h3>Risks</h3>{_ul(b.risks)}</div>"
            f"<div class='col'><h3>Watch items</h3>{_ul(b.watch_items)}</div>"
            f"</div>"
            + (f"<div class='sources'><h3>You.com sources ({len(b.sources)})</h3>{src}</div>" if src else "")
            + "</div>"
        )

    total_sources = sum(len(b.sources) for b in report.briefings)
    return f"""
    {REPORT_CSS}
    <div class="rpt">
      <h1>Financial Research Briefings</h1>
      <p class="sub">Grounded, citation-backed equity briefings — each company
      backed by You.com Research synthesis plus fresh Search news.</p>
      <div class="stats">
        <span class="pill"><b>{len(report.briefings)}</b> companies</span>
        <span class="pill"><b>{total_sources}</b> You.com sources cited</span>
      </div>
      {''.join(cards) or "<p class='empty'>No briefings generated.</p>"}
      <p class="yoube">Research answers from the You.com Research API (grounded
      synthesis with inline citations) plus fresh headlines from the You.com
      Search API (web + auto-classified news with timestamps and snippets).</p>
    </div>
    """
# {{/docs-fragment report}}


# {{docs-fragment driver}}
@env.task(report=True)
async def financial_research(
    companies: list[str] = [
        "NVIDIA",
        "Advanced Micro Devices",
        "Microsoft",
        "Alphabet",
        "Amazon",
        "Meta Platforms",
        "Broadcom",
        "Taiwan Semiconductor Manufacturing",
    ],
    focus: str = "Q4 earnings preview and competitive positioning",
    research_effort: str = "standard",
    freshness: str = "month",
) -> ResearchReport:
    """Fan out across companies and aggregate cited equity briefings."""
    with flyte.group("research-companies"):
        briefings = await asyncio.gather(
            *[
                research_company(c, focus, research_effort, freshness)
                for c in companies
            ]
        )

    report = ResearchReport(briefings=list(briefings))
    await flyte.report.replace.aio(_render_report(report), do_flush=True)
    await flyte.report.flush.aio()
    return report
# {{/docs-fragment driver}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(financial_research)
    print(run.url)
    run.wait()
# {{/docs-fragment main}}
