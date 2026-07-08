# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.4.0",
#     "httpx>=0.27.0",
#     "litellm>=1.72.0",
# ]
# main = "compliance_monitoring"
# params = ""
# ///
"""Regulatory & compliance monitoring agent.

Watches trusted regulatory sources via the You.com Research API (with
domain/freshness source controls and a structured output schema), then uses
Claude to assign severity and route citation-precise findings to the right team.
Every external call is traced so Flyte's audit lineage extends to the web layer.
"""

# {{docs-fragment env}}
import asyncio
import json
import os
from dataclasses import dataclass, field

import flyte

MODEL = "anthropic/claude-haiku-4-5"

env = flyte.TaskEnvironment(
    name="compliance-monitoring",
    secrets=[
        flyte.Secret(key="youdotcom-api-key", as_env_var="YDC_API_KEY"),
        flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    ],
    image=flyte.Image.from_uv_script(__file__, name="compliance-monitoring", pre=True),
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)
# {{/docs-fragment env}}


# {{docs-fragment data_types}}
@dataclass
class WatchItem:
    topic: str
    trusted_domains: list[str]
    team: str


@dataclass
class Finding:
    topic: str
    team: str
    title: str
    summary: str
    source_url: str
    published_date: str
    snippet: str
    domain: str = ""
    favicon: str = ""
    severity: str = "info"
    rationale: str = ""


def _domain(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _favicon_for(url: str) -> str:
    return f"https://ydc-index.io/favicon?domain={_domain(url)}&size=128"


@dataclass
class ComplianceReport:
    findings: list[Finding] = field(default_factory=list)
# {{/docs-fragment data_types}}


# {{docs-fragment you_research}}
YOU_RESEARCH_URL = "https://api.you.com/v1/research"

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "source_url": {"type": "string"},
                    "published_date": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": [
                    "title",
                    "summary",
                    "source_url",
                    "published_date",
                    "snippet",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


async def _you_post(url: str, body: dict, timeout: float = 300.0) -> dict:
    """POST with exponential backoff + jitter on 429 rate limits."""
    import asyncio
    import random

    import httpx

    # YDC_API_KEY is canonical; YOU_API_KEY accepted as a backwards-compatible fallback.
    headers = {
        "X-API-Key": os.environ.get("YDC_API_KEY") or os.environ["YOU_API_KEY"],
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(7):
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code == 429 and attempt < 6:
                wait = float(resp.headers.get("retry-after") or 0) or min(2**attempt, 30)
                await asyncio.sleep(wait + random.uniform(0, 2))
                continue
            resp.raise_for_status()
            return resp.json()
    resp.raise_for_status()
    return resp.json()


@flyte.trace
async def you_research(
    question: str,
    include_domains: list[str],
    freshness: str,
    research_effort: str = "standard",
) -> dict:
    """Call the You.com Research API with domain + freshness source controls."""
    body = {
        "input": question,
        "research_effort": research_effort,
        "source_control": {
            "include_domains": include_domains,
            "freshness": freshness,
        },
        "output_schema": FINDINGS_SCHEMA,
    }
    return await _you_post(YOU_RESEARCH_URL, body)
# {{/docs-fragment you_research}}


# {{docs-fragment llm}}
@flyte.trace
async def triage(topic: str, findings: list[dict]) -> list[dict]:
    """Use Claude to assign a severity + rationale to each finding."""
    from litellm import acompletion

    if not findings:
        return []

    system = (
        "You are a regulatory-compliance triage analyst. For each finding, "
        "assign a severity of 'info' (FYI), 'watch' (monitor closely), or "
        "'action' (requires a concrete compliance/legal response), and a one-"
        "sentence rationale. Respond ONLY with JSON: "
        '{"triage": [{"severity": str, "rationale": str}]} with one entry per '
        "finding, in order."
    )
    listing = "\n".join(
        f"[{i + 1}] {f.get('title', '')}: {f.get('summary', '')}"
        for i, f in enumerate(findings)
    )
    resp = await acompletion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Topic: {topic}\n\nFindings:\n{listing}"},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    parsed = _parse_json(resp.choices[0].message.content)
    return parsed.get("triage", []) if isinstance(parsed, dict) else []


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


# {{docs-fragment monitor_watch_item}}
@env.task(retries=3)
async def monitor_watch_item(item: WatchItem, freshness: str) -> list[Finding]:
    """Research one regulatory topic and produce triaged, cited findings."""
    question = (
        f"What are the most recent changes, updates, or new guidance regarding "
        f"'{item.topic}'? Report concrete, dated changes with their sources."
    )
    result = await you_research(question, item.trusted_domains, freshness)
    output = result.get("output", {})

    # Build a lookup from the Research API's full source list (url -> metadata).
    src_by_url: dict[str, dict] = {}
    for s in output.get("sources", []) or []:
        url = str(s.get("url", ""))
        if url:
            src_by_url[url] = s

    content = output.get("content", {})
    if isinstance(content, str):
        content = _parse_json(content) if content.strip() else {}
    raw_findings = content.get("findings", []) if isinstance(content, dict) else []

    triage_results = await triage(item.topic, raw_findings)

    findings: list[Finding] = []
    for i, f in enumerate(raw_findings):
        t = triage_results[i] if i < len(triage_results) else {}
        url = str(f.get("source_url", ""))
        meta = src_by_url.get(url, {})
        snippet = str(f.get("snippet", "")) or str((meta.get("snippets") or [""])[0])
        findings.append(
            Finding(
                topic=item.topic,
                team=item.team,
                title=str(f.get("title", "") or meta.get("title", "")),
                summary=str(f.get("summary", "")),
                source_url=url,
                published_date=str(f.get("published_date", "")),
                snippet=snippet,
                domain=_domain(url),
                favicon=_favicon_for(url),
                severity=str(t.get("severity", "info")),
                rationale=str(t.get("rationale", "")),
            )
        )
    return findings
# {{/docs-fragment monitor_watch_item}}


# {{docs-fragment report}}
_SEVERITY_ORDER = {"action": 0, "watch": 1, "info": 2}
_SEVERITY_STYLE = {
    "action": ("#fdecea", "#c0392b"),
    "watch": ("#fdf3e1", "#b7791f"),
    "info": ("#e3f1fb", "#2b6cb0"),
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
  .rpt .card.action { border-left-color:#c0392b; }
  .rpt .card.watch { border-left-color:#b7791f; }
  .rpt .card.info { border-left-color:#2b6cb0; }
  .rpt .card h2 { font-size:15px; margin:0 0 6px; color:#102a43; }
  .rpt .sev { display:inline-block; font-size:11px; font-weight:700;
              padding:3px 9px; border-radius:6px; text-transform:uppercase;
              letter-spacing:.03em; margin-right:8px; }
  .rpt .team { display:inline-block; font-size:11px; font-weight:600;
               padding:3px 9px; border-radius:6px; background:#edf0f3;
               color:#52606d; text-transform:uppercase; }
  .rpt .summary { margin:8px 0; font-size:14px; line-height:1.45; }
  .rpt .rationale { font-size:13px; color:#486581; font-style:italic; }
  .rpt .meta { color:#829ab1; font-size:12px; }
  .rpt a { color:#2b6cb0; text-decoration:none; }
  .rpt a:hover { text-decoration:underline; }
  .rpt .empty { color:#829ab1; font-style:italic; padding:8px 0; }
  .rpt .cite { display:flex; gap:9px; align-items:flex-start; background:#f7f9fb;
               border:1px solid #eef1f4; border-radius:8px; padding:8px 10px;
               margin-top:10px; }
  .rpt .cite img.fav { width:16px; height:16px; border-radius:3px; margin-top:2px;
                       flex:0 0 auto; background:#e4e7eb; }
  .rpt .cite .cb { font-size:12px; line-height:1.45; }
  .rpt .cite .cdom { font-weight:600; color:#334e68; }
  .rpt .cite .ctag { font-size:10px; font-weight:700; text-transform:uppercase;
                     color:#fff; background:#5b8def; border-radius:4px;
                     padding:1px 5px; margin-left:6px; }
  .rpt .cite .cmeta { color:#829ab1; }
  .rpt .cite .csnip { color:#52606d; font-style:italic; margin-top:3px; }
  .rpt .yoube { font-size:11px; color:#9aa5b1; margin-top:4px; }
</style>
"""


def _sev_badge(sev: str) -> str:
    bg, fg = _SEVERITY_STYLE.get(sev, ("#edf0f3", "#52606d"))
    return f"<span class='sev' style='background:{bg};color:{fg}'>{sev}</span>"


def _cite(f: Finding) -> str:
    """Render a rich You.com Research citation with domain, date, and snippet."""
    if not f.source_url:
        return ""
    meta = f.published_date[:10] if f.published_date else ""
    snip = f"<div class='csnip'>&ldquo;{f.snippet}&rdquo;</div>" if f.snippet else ""
    return (
        f"<div class='cite'><img class='fav' src='{f.favicon}' alt=''/>"
        f"<div class='cb'>"
        f"<a href='{f.source_url}'><span class='cdom'>{f.domain or 'source'}</span></a>"
        f"<span class='ctag'>research</span>"
        f"<div class='cmeta'>{meta} &middot; {f.title}</div>{snip}</div></div>"
    )


def _render_report(report: ComplianceReport) -> str:
    findings = sorted(
        report.findings,
        key=lambda f: (_SEVERITY_ORDER.get(f.severity, 3), f.team),
    )
    counts = {s: sum(1 for f in findings if f.severity == s) for s in _SEVERITY_ORDER}
    cited = sum(1 for f in findings if f.source_url)

    cards = []
    for f in findings:
        cards.append(
            f"<div class='card {f.severity}'>"
            f"<div>{_sev_badge(f.severity)}<span class='team'>{f.team}</span></div>"
            f"<h2>{f.title or f.topic}</h2>"
            f"<div class='summary'>{f.summary}</div>"
            f"<div class='rationale'>{f.rationale}</div>"
            f"<div class='meta' style='margin-top:6px'>{f.topic}</div>"
            f"{_cite(f)}</div>"
        )

    return f"""
    {REPORT_CSS}
    <div class="rpt">
      <h1>Compliance Monitoring Findings</h1>
      <p class="sub">Citation-precise regulatory changes from trusted domains —
      every finding links to a You.com Research source with snippet provenance.</p>
      <div class="stats">
        <span class="pill"><b>{len(findings)}</b> findings</span>
        <span class="pill"><b>{cited}</b> cited You.com sources</span>
        <span class="pill" style="background:#fdecea;color:#c0392b">
          <b>{counts['action']}</b> action</span>
        <span class="pill" style="background:#fdf3e1;color:#b7791f">
          <b>{counts['watch']}</b> watch</span>
        <span class="pill" style="background:#e3f1fb;color:#2b6cb0">
          <b>{counts['info']}</b> info</span>
      </div>
      {''.join(cards) or "<p class='empty'>No findings in this window.</p>"}
      <p class="yoube">Findings retrieved via the You.com Research API with
      <code>source_control</code> domain allowlists and freshness filters.
      Flyte logs which agent called which query and got which document — full
      prompt &rarr; citation lineage for audit.</p>
    </div>
    """
# {{/docs-fragment report}}


# {{docs-fragment driver}}
def _default_watch_items() -> list[WatchItem]:
    return [
        WatchItem(
            topic="FDA guidance on AI/ML-enabled medical device software",
            trusted_domains=["fda.gov", "federalregister.gov"],
            team="clinical",
        ),
        WatchItem(
            topic="SEC climate-related disclosure rules for public companies",
            trusted_domains=["sec.gov", "federalregister.gov"],
            team="legal",
        ),
        WatchItem(
            topic="OFAC sanctions list additions and updates",
            trusted_domains=["treasury.gov", "ofac.treasury.gov"],
            team="compliance",
        ),
        WatchItem(
            topic="State-level consumer data privacy laws and amendments",
            trusted_domains=["iapp.org", "oag.ca.gov"],
            team="legal",
        ),
        WatchItem(
            topic="FDA drug recalls and safety communications",
            trusted_domains=["fda.gov"],
            team="clinical",
        ),
        WatchItem(
            topic="HIPAA enforcement actions and guidance updates",
            trusted_domains=["hhs.gov"],
            team="compliance",
        ),
    ]


@env.task(report=True)
async def compliance_monitoring(
    watch_items: list[WatchItem] | None = None,
    freshness: str = "month",
) -> ComplianceReport:
    """Fan out across regulatory watch items and aggregate triaged findings."""
    if watch_items is None:
        watch_items = _default_watch_items()

    with flyte.group("monitor-watch-items"):
        results = await asyncio.gather(
            *[monitor_watch_item(item, freshness) for item in watch_items]
        )

    report = ComplianceReport(findings=[f for fs in results for f in fs])

    await flyte.report.replace.aio(_render_report(report), do_flush=True)
    await flyte.report.flush.aio()
    return report
# {{/docs-fragment driver}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(compliance_monitoring)
    print(run.url)
    run.wait()
# {{/docs-fragment main}}
