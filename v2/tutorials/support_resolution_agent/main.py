# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.4.0",
#     "httpx>=0.27.0",
#     "litellm>=1.72.0",
# ]
# main = "support_resolution"
# params = ""
# ///
"""Customer-support & field-service resolution agent.

Grounds a support ticket in fresh, public, citable sources via the You.com
Research API (low effort for low latency, human-in-the-loop use), then uses
Claude to draft a customer-ready reply that cites its sources inline so a human
agent can verify before sending.
"""

# {{docs-fragment env}}
import asyncio
import json
import os
from dataclasses import dataclass, field

import flyte

MODEL = "anthropic/claude-haiku-4-5"

env = flyte.TaskEnvironment(
    name="support-resolution",
    secrets=[
        flyte.Secret(key="youdotcom-api-key", as_env_var="YOU_API_KEY"),
        flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    ],
    image=flyte.Image.from_uv_script(__file__, name="support-resolution", pre=True),
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)
# {{/docs-fragment env}}


# {{docs-fragment data_types}}
@dataclass
class Source:
    title: str
    url: str
    snippet: str
    domain: str = ""
    favicon: str = ""


def _domain(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _favicon_for(url: str) -> str:
    return f"https://ydc-index.io/favicon?domain={_domain(url)}&size=128"


@dataclass
class Ticket:
    ticket_id: str
    question: str
    context: str = ""


@dataclass
class Grounding:
    answer: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class Resolution:
    ticket_id: str
    ticket: str
    grounded_answer: str
    draft_reply: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class ResolutionReport:
    resolutions: list[Resolution] = field(default_factory=list)
# {{/docs-fragment data_types}}


# {{docs-fragment you_research}}
YOU_RESEARCH_URL = "https://api.you.com/v1/research"


async def _you_post(url: str, body: dict, timeout: float = 120.0) -> dict:
    """POST with exponential backoff + jitter on 429 rate limits."""
    import random

    import httpx

    headers = {
        "X-API-Key": os.environ["YOU_API_KEY"],
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
async def you_research(question: str, research_effort: str = "lite") -> dict:
    """Fast, citation-backed grounding for a support question."""
    body = {"input": question, "research_effort": research_effort}
    return await _you_post(YOU_RESEARCH_URL, body)
# {{/docs-fragment you_research}}


# {{docs-fragment ground_answer}}
@env.task(retries=3)
async def ground_answer(ticket: str, context: str, research_effort: str) -> Grounding:
    """Ground the ticket in fresh public sources via the Research API."""
    question = ticket if not context else f"{ticket}\n\nContext: {context}"
    result = await you_research(question, research_effort)

    output = result.get("output", {})
    answer = output.get("content", "")
    if not isinstance(answer, str):
        answer = json.dumps(answer)

    sources = []
    for s in output.get("sources", []) or []:
        url = str(s.get("url", ""))
        sources.append(
            Source(
                title=str(s.get("title", "") or url),
                url=url,
                snippet=str((s.get("snippets") or [""])[0]),
                domain=_domain(url),
                favicon=_favicon_for(url),
            )
        )
    return Grounding(answer=answer, sources=sources)
# {{/docs-fragment ground_answer}}


# {{docs-fragment draft_reply}}
@flyte.trace
async def _draft(ticket: str, answer: str, sources_text: str) -> str:
    from litellm import acompletion

    system = (
        "You are a senior customer-support agent. Using ONLY the grounded "
        "answer and sources provided, draft a concise, friendly, customer-ready "
        "reply. Cite the relevant source URL inline in parentheses after any "
        "factual claim so a human agent can verify before sending. If the "
        "sources do not answer the question, say so plainly."
    )
    user = (
        f"Customer ticket: {ticket}\n\n"
        f"Grounded answer:\n{answer}\n\nSources:\n{sources_text}"
    )
    resp = await acompletion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


@env.task
async def draft_reply(ticket: Ticket, grounding: Grounding) -> Resolution:
    """Turn the grounded answer into a cited, customer-ready reply."""
    sources_text = "\n".join(
        f"- {s.title} ({s.domain}): {s.url}\n  \"{s.snippet}\""
        for s in grounding.sources
    )
    reply = await _draft(ticket.question, grounding.answer, sources_text)

    return Resolution(
        ticket_id=ticket.ticket_id,
        ticket=ticket.question,
        grounded_answer=grounding.answer,
        draft_reply=reply,
        sources=grounding.sources,
    )
# {{/docs-fragment draft_reply}}


# {{docs-fragment resolve_ticket}}
async def resolve_ticket(ticket: Ticket, research_effort: str) -> Resolution:
    """Ground one ticket then draft its reply."""
    grounding = await ground_answer(ticket.question, ticket.context, research_effort)
    return await draft_reply(ticket, grounding)
# {{/docs-fragment resolve_ticket}}


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
  .rpt .tid { display:inline-block; font-size:11px; font-weight:700;
              padding:3px 9px; border-radius:6px; background:#e0e8f9;
              color:#2b4ba0; margin-right:8px; }
  .rpt .q { font-size:15px; font-weight:600; color:#102a43; margin:8px 0 12px; }
  .rpt .reply { background:#f7faf7; border:1px solid #e1ece1; border-radius:8px;
                padding:12px 14px; font-size:14px; line-height:1.55; }
  .rpt .reply h3 { font-size:11px; text-transform:uppercase; letter-spacing:.04em;
                   color:#3c8a5e; margin:0 0 8px; }
  .rpt .sources { margin-top:12px; }
  .rpt .sources h3 { font-size:11px; text-transform:uppercase; color:#627d98;
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
                     color:#fff; background:#5b8def; border-radius:4px;
                     padding:1px 5px; margin-left:6px; }
  .rpt .cite .cmeta { color:#829ab1; }
  .rpt .cite .csnip { color:#52606d; font-style:italic; margin-top:2px; }
  .rpt .yoube { font-size:11px; color:#9aa5b1; margin-top:4px; }
</style>
"""


def _cite(s: Source) -> str:
    """Render a rich You.com Research citation for a support source."""
    if not s.url:
        return ""
    snip = f"<div class='csnip'>&ldquo;{s.snippet}&rdquo;</div>" if s.snippet else ""
    return (
        f"<div class='cite'><img class='fav' src='{s.favicon}' alt=''/>"
        f"<div class='cb'>"
        f"<a href='{s.url}'><span class='cdom'>{s.domain or 'source'}</span></a>"
        f"<span class='ctag'>research</span>"
        f"<div class='cmeta'>{s.title}</div>{snip}</div></div>"
    )


def _render_report(report: ResolutionReport) -> str:
    cards = []
    for res in report.resolutions:
        src = "".join(_cite(s) for s in res.sources[:8])
        reply_html = res.draft_reply.replace("\n", "<br/>")
        cards.append(
            f"<div class='card'>"
            f"<div><span class='tid'>{res.ticket_id}</span></div>"
            f"<div class='q'>{res.ticket}</div>"
            f"<div class='reply'><h3>Draft reply (for human review)</h3>{reply_html}</div>"
            + (f"<div class='sources'><h3>You.com sources ({len(res.sources)})</h3>{src}</div>" if src else "")
            + "</div>"
        )

    total_sources = sum(len(r.sources) for r in report.resolutions)
    return f"""
    {REPORT_CSS}
    <div class="rpt">
      <h1>Support Resolutions</h1>
      <p class="sub">Tickets grounded in fresh public sources via the You.com
      Research API — draft replies cite sources a human agent can verify.</p>
      <div class="stats">
        <span class="pill"><b>{len(report.resolutions)}</b> tickets</span>
        <span class="pill"><b>{total_sources}</b> You.com sources cited</span>
      </div>
      {''.join(cards) or "<p class='empty'>No tickets processed.</p>"}
      <p class="yoube">Each ticket grounded by the You.com Research API
      (<code>lite</code> effort for low-latency, human-in-the-loop use). Sources
      include domain, title, and snippet provenance — ready to paste into a
      customer reply with verification links.</p>
    </div>
    """
# {{/docs-fragment report}}


# {{docs-fragment driver}}
def _default_tickets() -> list[Ticket]:
    return [
        Ticket(
            "tkt-1",
            "Is there a recall on the DeWalt DCD777 cordless drill, and what should "
            "the customer do if there is?",
            "Customer purchased the drill recently and is asking about safety recalls.",
        ),
        Ticket(
            "tkt-2",
            "What is Sony's current return policy for the WH-1000XM5 headphones?",
            "Customer wants to return an opened pair bought 20 days ago.",
        ),
        Ticket(
            "tkt-3",
            "Are there any current weather advisories that could delay flights out of "
            "Denver International Airport today?",
            "Customer is worried about a connecting flight.",
        ),
        Ticket(
            "tkt-4",
            "What are the dimensions and weight capacity of the IKEA BEKANT desk?",
            "Customer is checking if it fits their space before resolving a complaint.",
        ),
        Ticket(
            "tkt-5",
            "Has Samsung issued any recall or safety notice for the Galaxy Z Fold5?",
            "Customer reports overheating and wants to know about known issues.",
        ),
        Ticket(
            "tkt-6",
            "What is the warranty period for a Dyson V15 Detect vacuum in the US?",
            "Customer's vacuum stopped working and asks about coverage.",
        ),
    ]


@env.task(report=True)
async def support_resolution(
    tickets: list[Ticket] | None = None,
    research_effort: str = "lite",
) -> ResolutionReport:
    """Fan out across support tickets, grounding and drafting cited replies."""
    if tickets is None:
        tickets = _default_tickets()

    with flyte.group("resolve-tickets"):
        resolutions = await asyncio.gather(
            *[resolve_ticket(t, research_effort) for t in tickets]
        )

    report = ResolutionReport(resolutions=list(resolutions))
    await flyte.report.replace.aio(_render_report(report), do_flush=True)
    await flyte.report.flush.aio()
    return report
# {{/docs-fragment driver}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(support_resolution)
    print(run.url)
    run.wait()
# {{/docs-fragment main}}
