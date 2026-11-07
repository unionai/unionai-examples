"""Building blocks for the tutorial's Flyte reports and artifact cards.

Every report and card is self-contained HTML: inline CSS and SVG, no scripts or external
assets, so it renders the same in the Flyte UI, in an artifact card, and in a browser.
"""

import json
from html import escape

PALETTE = ["#6f5bd8", "#2f9e8f", "#e08a2e", "#d4507a", "#4a8fe0", "#8a8f98"]

CSS = """
<style>
  .sd { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color: #1d1b26;
        max-width: 1120px; line-height: 1.45; }
  .sd h2 { font-size: 22px; margin: 4px 0 2px; letter-spacing: -0.3px; }
  .sd h3 { font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #6f5bd8;
           margin: 22px 0 10px; }
  .sd .lede { color: #5b5868; margin: 0 0 14px; font-size: 14px; }
  .sd .hero { background: linear-gradient(135deg, #6f5bd8, #3d2f8f); color: #fff;
              border-radius: 14px; padding: 20px 24px; margin-bottom: 16px; }
  .sd .hero h2 { color: #fff; }
  .sd .hero .lede { color: rgba(255,255,255,0.85); margin: 4px 0 0; }
  .sd .badges { margin-top: 12px; display: flex; gap: 6px; flex-wrap: wrap; }
  .sd .badge { background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.3);
               border-radius: 999px; padding: 2px 10px; font-size: 12px; }
  .sd .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
               gap: 10px; margin: 10px 0; }
  .sd .tile { background: #f6f4fd; border: 1px solid #e4dff7; border-radius: 10px;
              padding: 10px 14px; }
  .sd .tile .v { font-size: 22px; font-weight: 650; letter-spacing: -0.4px; }
  .sd .tile .k { font-size: 12px; color: #6a6678; }
  .sd .tile.good .v { color: #1f8a5b; } .sd .tile.bad .v { color: #c0392b; }
  .sd .charts { display: flex; flex-wrap: wrap; gap: 18px; }
  .sd .panel { background: #fff; border: 1px solid #ebe8f3; border-radius: 10px; padding: 12px; }
  .sd table { border-collapse: collapse; font-size: 13px; margin: 6px 0 12px; }
  .sd th { text-align: left; background: #f6f4fd; font-weight: 600; }
  .sd td, .sd th { border-bottom: 1px solid #ebe8f3; padding: 5px 10px; }
  .sd td.num, .sd th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .sd .chat { border: 1px solid #ebe8f3; border-radius: 10px; padding: 10px; margin: 8px 0;
              background: #fcfbff; }
  .sd .msg { margin: 6px 0; padding: 8px 10px; border-radius: 8px; font-size: 13px;
             white-space: pre-wrap; word-break: break-word; max-height: 220px; overflow: auto; }
  .sd .role { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .6px;
              margin-bottom: 2px; opacity: .7; }
  .sd .user { background: #eef3ff; } .sd .assistant { background: #f1edff; }
  .sd .tool { background: #eefaf6; font-family: ui-monospace, Menlo, monospace; }
  .sd .system { background: #f4f4f6; color: #5b5868; }
  .sd code, .sd .mono { font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
  .sd .decision { font-size: 16px; font-weight: 650; padding: 10px 14px; border-radius: 10px; }
  .sd .decision.good { background: #e8f7ef; color: #1f6e49; }
  .sd .decision.neutral { background: #f4f4f6; color: #4a4858; }
</style>
"""


def page(*parts: str) -> str:
    return CSS + f"<div class='sd'>{''.join(parts)}</div>"


def hero(title: str, subtitle: str = "", badges: list[str] | None = None) -> str:
    chips = "".join(f"<span class='badge'>{escape(b)}</span>" for b in badges or [])
    sub = f"<p class='lede'>{escape(subtitle)}</p>" if subtitle else ""
    return f"<div class='hero'><h2>{escape(title)}</h2>{sub}<div class='badges'>{chips}</div></div>"


def section(title: str, *parts: str, lede: str = "") -> str:
    intro = f"<p class='lede'>{escape(lede)}</p>" if lede else ""
    return f"<h3>{escape(title)}</h3>{intro}{''.join(parts)}"


def tiles(items: list[tuple[str, str]] | dict[str, str], tone: dict[str, str] | None = None) -> str:
    """Big-number tiles. `tone` maps a label to "good" or "bad"."""
    pairs = items.items() if isinstance(items, dict) else items
    tone = tone or {}
    return "<div class='tiles'>" + "".join(
        f"<div class='tile {tone.get(k, '')}'><div class='v'>{escape(str(v))}</div>"
        f"<div class='k'>{escape(k)}</div></div>"
        for k, v in pairs
    ) + "</div>"


def charts(*svgs: str) -> str:
    return "<div class='charts'>" + "".join(f"<div class='panel'>{s}</div>" for s in svgs) + "</div>"


def table(headers: list[str], rows: list[list], numeric: set[int] | None = None) -> str:
    numeric = numeric if numeric is not None else set(range(1, len(headers)))
    head = "".join(
        f"<th class='{'num' if i in numeric else ''}'>{escape(h)}</th>" for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>" + "".join(
            f"<td class='{'num' if i in numeric else ''}'>{escape(str(c))}</td>" for i, c in enumerate(r)
        ) + "</tr>"
        for r in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def bar_chart(
    title: str,
    series: dict[str, float],
    fmt: str = "{:.2f}",
    reference: float | None = None,
    width: int = 500,
) -> str:
    """Horizontal bars, one per label. `reference` draws a dashed line, e.g. 1.0x."""
    bar_h, gap, label_w = 20, 8, 140
    values = list(series.values()) or [0]
    peak = max(values + [reference or 0]) or 1
    height = len(series) * (bar_h + gap) + 30
    span = width - label_w - 70
    out = []
    for i, (name, value) in enumerate(series.items()):
        y = 24 + i * (bar_h + gap)
        w = max(span * value / peak, 1)
        out.append(
            f"<text x='{label_w - 8}' y='{y + 14}' text-anchor='end' font-size='12'>{escape(name)}</text>"
            f"<rect x='{label_w}' y='{y}' width='{w:.1f}' height='{bar_h}' rx='4' fill='{PALETTE[i % len(PALETTE)]}'/>"
            f"<text x='{label_w + w + 6:.1f}' y='{y + 14}' font-size='12'>{escape(fmt.format(value))}</text>"
        )
    if reference is not None:
        x = label_w + span * reference / peak
        out.append(f"<line x1='{x:.1f}' y1='18' x2='{x:.1f}' y2='{height - 4}' stroke='#555' stroke-dasharray='4 3'/>")
    return _svg(title, width, height, "".join(out))


def histogram(title: str, values: list[float], bins: int = 12, unit: str = "", width: int = 500) -> str:
    """Vertical histogram of a list of numbers."""
    height, pad_l, pad_b = 190, 36, 30
    if not values:
        return _svg(title, width, 60, "<text x='0' y='40' font-size='12'>no data</text>")
    lo, hi = min(values), max(values)
    step = (hi - lo) / bins or 1
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / step), bins - 1)] += 1
    peak = max(counts)
    bw = (width - pad_l - 10) / bins
    plot_h = height - pad_b - 24
    out = []
    for i, c in enumerate(counts):
        h = plot_h * c / peak
        x = pad_l + i * bw
        out.append(f"<rect x='{x + 1:.1f}' y='{24 + plot_h - h:.1f}' width='{bw - 2:.1f}' height='{h:.1f}' rx='2' fill='{PALETTE[0]}'/>")
    out.append(f"<text x='{pad_l}' y='{height - 10}' font-size='11'>{lo:,.0f}{unit}</text>")
    out.append(f"<text x='{width - 10}' y='{height - 10}' font-size='11' text-anchor='end'>{hi:,.0f}{unit}</text>")
    out.append(f"<text x='{pad_l - 6}' y='30' font-size='11' text-anchor='end'>{peak}</text>")
    return _svg(title, width, height, "".join(out))


def line_chart(
    title: str,
    series: dict[str, list[tuple[float, float]]],
    width: int = 520,
    height: int = 220,
    y_label: str = "",
) -> str:
    """Multi-series line chart of (x, y) points, with a legend."""
    pad_l, pad_r, pad_t, pad_b = 46, 12, 44, 34
    points = [p for s in series.values() for p in s]
    if not points:
        return _svg(title, width, 60, "<text x='0' y='40' font-size='12'>waiting for data</text>")
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    if y1 == y0:
        y1 = y0 + 1
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def sx(x):
        return pad_l + pw * (x - x0) / ((x1 - x0) or 1)

    def sy(y):
        return pad_t + ph * (1 - (y - y0) / (y1 - y0))

    out = [
        f"<line x1='{pad_l}' y1='{pad_t + ph}' x2='{pad_l + pw}' y2='{pad_t + ph}' stroke='#ccc'/>",
        f"<text x='{pad_l - 6}' y='{pad_t + 4}' font-size='11' text-anchor='end'>{y1:.3g}</text>",
        f"<text x='{pad_l - 6}' y='{pad_t + ph}' font-size='11' text-anchor='end'>{y0:.3g}</text>",
        f"<text x='{pad_l}' y='{height - 8}' font-size='11'>{x0:,.0f}</text>",
        f"<text x='{pad_l + pw}' y='{height - 8}' font-size='11' text-anchor='end'>{x1:,.0f}{(' ' + y_label) if y_label else ''}</text>",
    ]
    for i, (name, pts) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in sorted(pts))
        out.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{path}'/>")
        lx = pad_l + i * 120
        out.append(f"<rect x='{lx}' y='24' width='10' height='3' fill='{color}'/>")
        out.append(f"<text x='{lx + 14}' y='29' font-size='11'>{escape(name)}</text>")
    return _svg(title, width, height, "".join(out))


def _svg(title: str, width: int, height: int, body: str) -> str:
    return (
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg' "
        f"font-family='-apple-system, Segoe UI, Roboto, sans-serif' fill='#1d1b26'>"
        f"<text x='0' y='14' font-size='13' font-weight='600'>{escape(title)}</text>{body}</svg>"
    )


def chat(messages: list[dict], max_chars: int = 1200) -> str:
    """Render an OpenAI-format conversation, tool calls included."""
    out = []
    for m in messages:
        role = m.get("role", "")
        text = m.get("content") or ""
        if m.get("tool_calls"):
            calls = "\n".join(
                f"{c['function']['name']}({c['function'].get('arguments', '')})" for c in m["tool_calls"]
            )
            text = (text + "\n" if text else "") + "→ " + calls
        if len(text) > max_chars:
            text = text[:max_chars] + " …"
        css = role if role in {"user", "assistant", "tool", "system"} else "system"
        out.append(f"<div class='msg {css}'><div class='role'>{escape(role)}</div>{escape(text)}</div>")
    return "<div class='chat'>" + "".join(out) + "</div>"


def pretty_json(value) -> str:
    return f"<pre class='mono'>{escape(json.dumps(value, indent=2))}</pre>"
