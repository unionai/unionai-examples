"""Report and artifact-card visuals: stat tiles, charts, tables and a card shell, in plain HTML + SVG.

No dependencies, so it works in every task image. Everything renders inside a ``.viz`` root
whose colors are CSS custom properties with selected light and dark values, and every mark
carries an SVG ``<title>`` so hovering shows its value. Each tutorial
folder has its own copy, so every example is self-contained.

    html = viz.page("Title", "subtitle", viz.tiles([...]), viz.line_chart([...]))
    await flyte.report.replace.aio(html, do_flush=True)
"""

from __future__ import annotations

import html as _html
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# Categorical slots, fixed order (validated light/dark pairs). Status colors are separate.
SLOTS = 8
STYLE = """
<style>
.viz { color-scheme: light;
  --surface: #fcfcfb; --plane: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --good-text: #006300;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --q1: #cde2fb; --q2: #9ec5f4; --q3: #6da7ec; --q4: #3987e5; --q5: #256abf; --q6: #184f95; --q7: #0d366b;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b; --neutral: #c3c2b7;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--plane);
  padding: 20px; border-radius: 12px; line-height: 1.45; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz { color-scheme: dark;
  --surface: #1a1a19; --plane: #0d0d0d; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --good-text: #0ca30c;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
  --q1: #0d366b; --q2: #184f95; --q3: #256abf; --q4: #3987e5; --q5: #6da7ec; --q6: #9ec5f4; --q7: #cde2fb; --neutral: #383835; } }
:root[data-theme="dark"] .viz { color-scheme: dark;
  --surface: #1a1a19; --plane: #0d0d0d; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --good-text: #0ca30c;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
  --q1: #0d366b; --q2: #184f95; --q3: #256abf; --q4: #3987e5; --q5: #6da7ec; --q6: #9ec5f4; --q7: #cde2fb; --neutral: #383835; }
.viz * { box-sizing: border-box; }
.viz h2 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.2px; }
.viz h3 { font-size: 14px; margin: 0 0 10px; color: var(--ink); }
.viz .sub { color: var(--ink-2); font-size: 13px; margin: 0 0 16px; }
.viz .note { color: var(--ink-2); font-size: 12px; margin: 8px 0 0; }
.viz .panel { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 16px; margin: 0 0 14px; }
.viz .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; }
.viz .row > .panel { margin: 0; }
.viz .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 0 0 14px; }
.viz .tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 12px 14px; }
.viz .tile .label { font-size: 12px; color: var(--ink-2); }
.viz .tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
.viz .tile .tnote { font-size: 11px; color: var(--muted); margin-top: 2px; }
.viz .hero .value { font-size: 48px; line-height: 1.1; }
.viz svg { display: block; width: 100%; height: auto; overflow: visible; }
.viz svg text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.viz svg text.ink { fill: var(--ink); } .viz svg text.ink2 { fill: var(--ink-2); }
.viz .gl { stroke: var(--grid); stroke-width: 1; } .viz .ax { stroke: var(--axis); stroke-width: 1; }
.viz .ref { stroke: var(--ink-2); stroke-width: 1; }
.viz .legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--ink-2); margin: 0 0 8px; }
.viz .key { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.viz table { border-collapse: collapse; width: 100%; font-size: 12px; }
.viz th { text-align: left; color: var(--ink-2); font-weight: 600; border-bottom: 1px solid var(--axis); padding: 6px 8px; }
.viz td { border-bottom: 1px solid var(--grid); padding: 6px 8px; vertical-align: top; font-variant-numeric: tabular-nums; }
.viz td:not(:last-child) { white-space: nowrap; }
.viz code, .viz pre { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; }
.viz pre { background: var(--plane); border: 1px solid var(--ring); border-radius: 8px; padding: 10px; overflow-x: auto; margin: 0; }
.viz .badge { display: inline-block; border: 1px solid var(--ring); border-radius: 999px; padding: 2px 10px; font-size: 12px; color: var(--ink-2); margin: 0 6px 6px 0; }
.viz .status { white-space: nowrap; }
.viz details summary { cursor: pointer; color: var(--ink-2); font-size: 12px; }
</style>
"""

# Status: (css var, icon). Always shown with its label, never color alone.
STATUS = {
    "good": ("--good", "✓"),
    "warning": ("--warning", "▲"),
    "serious": ("--serious", "◆"),
    "critical": ("--critical", "✕"),
    "neutral": ("--neutral", "•"),
}


def esc(s: object) -> str:
    return _html.escape(str(s))


def slot(i: int) -> str:
    """CSS color for categorical slot ``i`` (1-based)."""
    return f"var(--s{(i - 1) % SLOTS + 1})"


def color(c: str | int) -> str:
    """An int is a categorical slot; a status name maps to its status color; anything else passes through."""
    if isinstance(c, int):
        return slot(c)
    if c in STATUS:
        return f"var({STATUS[c][0]})"
    return c


def compact(v: float, digits: int = 1) -> str:
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:.{digits}f}".rstrip("0").rstrip(".") + suf
    if a >= 100 or float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.{2 if a < 10 else 1}f}".rstrip("0").rstrip(".")


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    out, t = [], start
    while not out or out[-1] < hi - 1e-9:  # always end on a tick at or above ``hi``
        out.append(round(t, 10))
        t += step
    return out


# ── layout ──────────────────────────────────────────────────────────────────


def page(title: str, subtitle: str = "", *blocks: str) -> str:
    """A full report body: style, title, subtitle and the blocks in order."""
    sub = f"<p class='sub'>{subtitle}</p>" if subtitle else ""
    return f"{STYLE}<div class='viz'><h2>{esc(title)}</h2>{sub}{''.join(blocks)}</div>"


def panel(title: str, body: str, note: str = "") -> str:
    head = f"<h3>{esc(title)}</h3>" if title else ""
    foot = f"<p class='note'>{note}</p>" if note else ""
    return f"<div class='panel'>{head}{body}{foot}</div>"


def row(*panels: str) -> str:
    return f"<div class='row'>{''.join(panels)}</div>"


def tiles(items: Iterable[tuple[str, str] | tuple[str, str, str]], hero: bool = False) -> str:
    """Stat tiles: (label, value[, note]). ``hero=True`` makes the first tile the view's headline."""
    out = []
    for i, it in enumerate(items):
        label, value, note = (*it, "")[:3]
        cls = "tile hero" if hero and i == 0 else "tile"
        tn = f"<div class='tnote'>{note}</div>" if note else ""
        out.append(f"<div class='{cls}'><div class='label'>{esc(label)}</div><div class='value'>{value}</div>{tn}</div>")
    return f"<div class='tiles'>{''.join(out)}</div>"


def legend(items: Iterable[tuple[str, str | int]]) -> str:
    return "<div class='legend'>" + "".join(
        f"<span><span class='key' style='background:{color(c)}'></span>{esc(name)}</span>" for name, c in items
    ) + "</div>"


def status(kind: str, label: str) -> str:
    var, icon = STATUS[kind]
    return f"<span class='status'><span style='color:var({var})'>{icon}</span> {esc(label)}</span>"


def table(header: Sequence[str], rows: Iterable[Sequence[object]], raw: bool = True) -> str:
    """Rows are rendered as given (``raw``) so cells can hold status() or <code>; escape inputs yourself."""
    th = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c if raw else esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def code(text: str) -> str:
    return f"<pre>{esc(text.strip())}</pre>"


# ── charts ──────────────────────────────────────────────────────────────────


@dataclass
class Series:
    name: str
    values: Sequence[float | None]
    color: str | int = 1
    xs: Sequence[float] | None = None  # defaults to 0..n-1
    step: bool = False  # draw as a step function (counts over time)


def line_chart(
    series: Sequence[Series],
    *,
    height: int = 220,
    width: int = 640,
    y_range: tuple[float, float] | None = None,
    x_label: str = "",
    y_fmt=compact,
    x_fmt=compact,
    markers: dict[float, str] | None = None,
    refs: Sequence[tuple[float, str]] = (),
    area: bool = False,
) -> str:
    """Lines on one y-axis. ``markers`` are vertical event lines at x; ``refs`` horizontal reference lines."""
    pts = [(x, v) for s in series for x, v in zip(s.xs or range(len(s.values)), s.values) if v is not None]
    if not pts:
        return "<p class='note'>no data yet</p>"
    xs = [p[0] for p in pts]
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    ys = [p[1] for p in pts] + [r[0] for r in refs]
    lo, hi = y_range or (min(ys), max(ys))
    ticks = nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    L, R, T, B = 44, 70, 10, 26
    W, H = width - L - R, height - T - B

    def X(x: float) -> float:
        return L + W * (x - x0) / (x1 - x0)

    def Y(v: float) -> float:
        return T + H * (1 - (v - lo) / ((hi - lo) or 1))

    g = "".join(
        f"<line class='gl' x1='{L}' x2='{L + W}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>"
        f"<text x='{L - 6}' y='{Y(t) + 4:.1f}' text-anchor='end'>{y_fmt(t)}</text>"
        for t in ticks
    )
    xt = nice_ticks(x0, x1, 5)
    if all(float(x).is_integer() for x in xs):
        xt = [t for t in xt if float(t).is_integer()]  # no "step 1.5" ticks on an integer axis
    g += "".join(
        f"<text x='{X(t):.1f}' y='{T + H + 16}' text-anchor='middle'>{x_fmt(t)}</text>" for t in xt if x0 <= t <= x1
    )
    if x_label:
        g += f"<text x='{L + W}' y='{T + H + 16}' text-anchor='end' dx='{R - 4}'>{esc(x_label)}</text>"
    for v, label in refs:
        g += (
            f"<line class='ref' x1='{L}' x2='{L + W}' y1='{Y(v):.1f}' y2='{Y(v):.1f}'/>"
            f"<text class='ink2' x='{L + W + 4}' y='{Y(v) + 4:.1f}'>{esc(label)}</text>"
        )
    for k, (x, label) in enumerate(sorted((markers or {}).items())):
        g += (
            f"<line class='ref' x1='{X(x):.1f}' x2='{X(x):.1f}' y1='{T}' y2='{T + H}'/>"
            f"<text class='ink2' x='{X(x) + 4:.1f}' y='{T + 10 + 14 * (k % 2)}'>{esc(label)}</text>"
        )
    marks, ends = "", ""
    for s in series:
        p = [(X(x), Y(v), x, v) for x, v in zip(s.xs or range(len(s.values)), s.values) if v is not None]
        if not p:
            continue
        c = color(s.color)
        if s.step:
            d = f"M{p[0][0]:.1f},{p[0][1]:.1f}" + "".join(f" H{a:.1f} V{b:.1f}" for a, b, _, _ in p[1:])
        else:
            d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b, _, _ in p)
        if area:
            marks += f"<path d='{d} L{p[-1][0]:.1f},{T + H} L{p[0][0]:.1f},{T + H} Z' style='fill:{c};opacity:.1'/>"
        marks += f"<path d='{d}' style='fill:none;stroke:{c};stroke-width:2;stroke-linejoin:round;stroke-linecap:round'/>"
        # Hover targets: a generous invisible hit circle per point.
        marks += "".join(
            f"<circle cx='{a:.1f}' cy='{b:.1f}' r='7' style='fill:transparent'><title>{esc(s.name)} · {x_fmt(x)}: {y_fmt(v)}</title></circle>"
            for a, b, x, v in p
        )
        a, b, _, v = p[-1]
        ends += (
            f"<circle cx='{a:.1f}' cy='{b:.1f}' r='4' style='fill:{c};stroke:var(--surface);stroke-width:2'/>"
            f"<text class='ink' x='{a + 8:.1f}' y='{b + 4:.1f}'>{y_fmt(v)}</text>"
        )
    head = legend((s.name, s.color) for s in series) if len(series) > 1 else ""
    return f"{head}<svg viewBox='0 0 {width} {height}' style='max-width:{width}px' role='img'>{g}{marks}{ends}</svg>"


@dataclass
class Bar:
    label: str
    value: float
    color: str | int = 1
    note: str = ""  # shown in the tooltip and after the value


def bar_chart(
    bars: Sequence[Bar],
    *,
    width: int = 640,
    unit: str = "",
    refs: Sequence[tuple[float, str]] = (),
    max_value: float | None = None,
    label_width: int = 170,
    fmt=compact,
    legend_items: Sequence[tuple[str, str | int]] = (),
) -> str:
    """Horizontal bars from one baseline, value at the tip; ``refs`` are vertical reference lines (caps, ladders)."""
    if not bars:
        return "<p class='note'>no data</p>"
    band, thick = 28, 18
    L, R, T = label_width, 110, 6
    top = max_value or max([b.value for b in bars] + [r[0] for r in refs] + [1e-9]) * 1.02
    W = width - L - R
    H = band * len(bars)
    height = T + H + 22

    def X(v: float) -> float:
        return L + W * max(v, 0) / top

    g = "".join(
        f"<line class='gl' x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{T}' y2='{T + H}'/>"
        f"<text x='{X(t):.1f}' y='{T + H + 14}' text-anchor='middle'>{fmt(t)}{esc(unit)}</text>"
        for t in nice_ticks(0, top, 4)
        if t <= top
    )
    marks = ""
    for i, b in enumerate(bars):
        y = T + i * band + (band - thick) / 2
        w = max(X(b.value) - L, 2)
        r = min(4, w / 2)
        c = color(b.color)
        # 4px rounded data-end, square at the baseline.
        d = f"M{L},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{thick - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} Z"
        tip = f"{esc(b.label)}: {fmt(b.value)}{esc(unit)}" + (f" · {esc(b.note)}" if b.note else "")
        marks += (
            f"<text class='ink2' x='{L - 8}' y='{y + thick / 2 + 4:.1f}' text-anchor='end'>{esc(b.label)}</text>"
            f"<path d='{d}' style='fill:{c}'><title>{tip}</title></path>"
            f"<text class='ink' x='{L + w + 6:.1f}' y='{y + thick / 2 + 4:.1f}'>{fmt(b.value)}{esc(unit)}"
            f"{'  ' + esc(b.note) if b.note else ''}</text>"
        )
    for v, label in refs:
        marks += (
            f"<line class='ref' x1='{X(v):.1f}' x2='{X(v):.1f}' y1='{T - 4}' y2='{T + H}' style='stroke-dasharray:0'/>"
            f"<text class='ink2' x='{X(v):.1f}' y='{T - 6}' text-anchor='middle'>{esc(label)}</text>"
        )
    head = legend(legend_items) if legend_items else ""
    pad = 14 if refs else 0
    return f"{head}<svg viewBox='0 {-pad} {width} {height + pad}' style='max-width:{width}px' role='img'>{g}{marks}</svg>"


def stacked_columns(
    categories: Sequence[str],
    stacks: Sequence[Series],
    *,
    height: int = 220,
    width: int = 640,
    unit: str = "",
    fmt=compact,
) -> str:
    """Vertical stacked columns, one per category, 2px surface gap between segments; total on the cap."""
    n = len(categories)
    if not n:
        return "<p class='note'>no data</p>"
    totals = [sum((s.values[i] or 0) for s in stacks) for i in range(n)]
    ticks = nice_ticks(0, max(totals + [1e-9]))
    top = ticks[-1]
    L, R, T, B = 44, 10, 16, 26
    W, H = width - L - R, height - T - B
    slotw = W / n
    thick = min(24, slotw * 0.6)
    # Many columns: label every k-th category, and put totals only on the first and tallest column.
    every = max(1, -(-n * 28 // max(int(W), 1)))
    crowded = slotw < 22
    peak = max(range(n), key=totals.__getitem__)

    def Y(v: float) -> float:
        return T + H * (1 - v / top)

    g = "".join(
        f"<line class='gl' x1='{L}' x2='{L + W}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>"
        f"<text x='{L - 6}' y='{Y(t) + 4:.1f}' text-anchor='end'>{fmt(t)}{esc(unit)}</text>"
        for t in ticks
    )
    marks = ""
    for i, cat in enumerate(categories):
        x = L + i * slotw + (slotw - thick) / 2
        base = 0.0
        segs = [(s, s.values[i] or 0) for s in stacks if (s.values[i] or 0) > 0]
        for j, (s, v) in enumerate(segs):
            y1, y0 = Y(base + v), Y(base)
            h = max(y0 - y1 - (2 if j else 0), 1)  # 2px surface gap
            last = j == len(segs) - 1
            r = min(4, h / 2, thick / 2) if last else 0
            if r:
                d = (f"M{x:.1f},{y0 - (2 if j else 0):.1f} V{y1 + r:.1f} a{r},{r} 0 0 1 {r},-{r} H{x + thick - r:.1f} "
                     f"a{r},{r} 0 0 1 {r},{r} V{y0 - (2 if j else 0):.1f} Z")
            else:
                d = f"M{x:.1f},{y0 - (2 if j else 0):.1f} V{y0 - (2 if j else 0) - h:.1f} h{thick:.1f} V{y0 - (2 if j else 0):.1f} Z"
            marks += f"<path d='{d}' style='fill:{color(s.color)}'><title>{esc(cat)} · {esc(s.name)}: {fmt(v)}{esc(unit)}</title></path>"
            base += v
        if not crowded or i in (0, peak):
            marks += f"<text class='ink' x='{x + thick / 2:.1f}' y='{Y(totals[i]) - 4:.1f}' text-anchor='middle'>{fmt(totals[i])}</text>"
        if i % every == 0:
            marks += f"<text x='{x + thick / 2:.1f}' y='{T + H + 16}' text-anchor='middle'>{esc(cat)}</text>"
    head = legend((s.name, s.color) for s in stacks) if len(stacks) > 1 else ""
    return f"{head}<svg viewBox='0 0 {width} {height}' style='max-width:{width}px' role='img'>{g}{marks}</svg>"


def heatmap(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    width: int = 640,
    lo: float | None = None,
    hi: float | None = None,
    fmt=compact,
    label_width: int = 70,
    cell_h: int = 16,
    show_every: int = 1,
) -> str:
    """Sequential one-hue heatmap (blue, 7 steps); hover a cell for its value."""
    flat = [v for r in matrix for v in r]
    lo = min(flat) if lo is None else lo
    hi = max(flat) if hi is None else hi
    nr, nc = len(matrix), len(col_labels)
    L, T = label_width, 4
    cw = (width - L - 8) / max(nc, 1)
    gap, rx = (2, 3) if cw >= 8 else (0, 0)  # hundreds of columns: a continuous strip
    out = ""
    for i, r in enumerate(matrix):
        y = T + i * cell_h
        out += f"<text class='ink2' x='{L - 6}' y='{y + cell_h / 2 + 4:.1f}' text-anchor='end'>{esc(row_labels[i])}</text>"
        for j, v in enumerate(r):
            k = 1 + min(6, int(7 * (v - lo) / ((hi - lo) or 1)))
            out += (
                f"<rect x='{L + j * cw + 1:.1f}' y='{y + 1}' width='{cw - 2:.1f}' height='{cell_h - 2}' rx='2' "
                f"style='fill:var(--q{k})'><title>{esc(row_labels[i])} · {esc(col_labels[j])}: {fmt(v)}</title></rect>"
            )
    out += "".join(
        f"<text x='{L + j * cw + cw / 2:.1f}' y='{T + nr * cell_h + 14}' text-anchor='middle'>{esc(c)}</text>"
        for j, c in enumerate(col_labels)
        if j % show_every == 0
    )
    scale = "".join(f"<span class='key' style='background:var(--q{k});margin:0'></span>" for k in range(1, 8))
    key = f"<div class='legend'><span>{fmt(lo)}</span><span>{scale}</span><span>{fmt(hi)}</span></div>"
    h = T + nr * cell_h + 22
    return f"{key}<svg viewBox='0 0 {width} {h}' style='max-width:{width}px' role='img'>{out}</svg>"


def cell_grid(
    cells: Sequence[Sequence[str]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    kinds: dict[str, tuple[str | int, str]],
    *,
    width: int = 640,
    label_width: int = 70,
    cell_h: int = 18,
    show_every: int = 1,
) -> str:
    """A grid of categorical states, e.g. step x shard. ``kinds`` maps a state to (color, legend label)."""
    nr, nc = len(cells), len(col_labels)
    L, T = label_width, 4
    cw = (width - L - 8) / max(nc, 1)
    gap, rx = (2, 3) if cw >= 8 else (0, 0)  # hundreds of columns: a continuous strip
    out = ""
    for i, r in enumerate(cells):
        y = T + i * cell_h
        out += f"<text class='ink2' x='{L - 6}' y='{y + cell_h / 2 + 4:.1f}' text-anchor='end'>{esc(row_labels[i])}</text>"
        for j, k in enumerate(r):
            c, name = kinds[k]
            out += (
                f"<rect x='{L + j * cw + gap / 2:.1f}' y='{y + 1}' width='{cw - gap:.1f}' height='{cell_h - 2}' rx='{rx}' "
                f"style='fill:{color(c)}'><title>{esc(row_labels[i])} · {esc(col_labels[j])}: {esc(name)}</title></rect>"
            )
    out += "".join(
        f"<text x='{L + j * cw + cw / 2:.1f}' y='{T + nr * cell_h + 14}' text-anchor='middle'>{esc(c)}</text>"
        for j, c in enumerate(col_labels)
        if j % show_every == 0
    )
    h = T + nr * cell_h + 22
    return legend((name, c) for c, name in kinds.values()) + f"<svg viewBox='0 0 {width} {h}' style='max-width:{width}px' role='img'>{out}</svg>"


@dataclass
class Span:
    lane: str
    label: str
    start: float
    end: float
    color: str | int | None = None  # defaults to the lane's slot


def gantt(
    spans: Sequence[Span], *, width: int = 720, label_width: int = 190, unit: str = "s", packed: bool | None = None
) -> str:
    """Spans on a shared time axis from the first start, grouped by lane.

    One row per span, or one row per lane when ``packed`` (the default past 30 spans, so a long run
    stays one screen tall).
    """
    if not spans:
        return "<p class='note'>no spans</p>"
    t0 = min(s.start for s in spans)
    t1 = max(s.end for s in spans)
    lanes = sorted({s.lane for s in spans}, key=lambda ln: min(s.start for s in spans if s.lane == ln))
    if packed is None:
        packed = len(spans) > 30
    if packed:
        rows = [(ln, [s for s in spans if s.lane == ln]) for ln in lanes]
    else:
        rows = [(f"{s.lane} · {s.label}", [s]) for s in sorted(spans, key=lambda s: (lanes.index(s.lane), s.start))]
    rh, thick = (26, 18) if packed else (18, 12)
    L, T = label_width, 4
    W = width - L - 16
    scale = W / max(t1 - t0, 1)
    ticks = nice_ticks(0, t1 - t0, 6)
    g = "".join(
        f"<line class='gl' x1='{L + t * scale:.1f}' x2='{L + t * scale:.1f}' y1='{T}' y2='{T + rh * len(rows)}'/>"
        f"<text x='{L + t * scale:.1f}' y='{T + rh * len(rows) + 14}' text-anchor='middle'>{compact(t)}{unit}</text>"
        for t in ticks
        if t <= t1 - t0
    )
    marks, prev = "", None
    for i, (label, row) in enumerate(rows):
        y = T + i * rh
        lane = row[0].lane
        if lane != prev and prev is not None:
            marks += f"<line class='ax' x1='0' x2='{width}' y1='{y:.1f}' y2='{y:.1f}'/>"
        prev = lane
        marks += f"<text class='ink2' x='{L - 8}' y='{y + rh / 2 + 4:.1f}' text-anchor='end'>{esc(label)}</text>"
        for s in row:
            x = L + (s.start - t0) * scale
            w = max((s.end - s.start) * scale, 1 if packed else 3)
            c = color(s.color if s.color is not None else lanes.index(s.lane) + 1)
            marks += (
                f"<rect x='{x:.1f}' y='{y + (rh - thick) / 2:.1f}' width='{w:.1f}' height='{thick}' rx='{0 if packed else 3}' style='fill:{c}'>"
                f"<title>{esc(s.lane)} · {esc(s.label)}: {s.start - t0:.0f}{unit} → {s.end - t0:.0f}{unit} ({s.end - s.start:.0f}{unit})</title></rect>"
            )
    h = T + rh * len(rows) + 22
    return f"<svg viewBox='0 0 {width} {h}' style='max-width:{width}px' role='img'>{g}{marks}</svg>"


def concurrency(spans: Sequence[Span], lane: str, step: float = 1.0) -> tuple[list[float], list[float]]:
    """(times, running count) for one lane, relative to the earliest start across ``spans``."""
    t0 = min(s.start for s in spans)
    mine = [s for s in spans if s.lane == lane]
    t1 = max(s.end for s in spans)
    xs, ys, t = [], [], t0
    while t <= t1 + step:
        xs.append(t - t0)
        ys.append(sum(1 for s in mine if s.start <= t < s.end))
        t += step
    return xs, ys


# ── artifact cards ──────────────────────────────────────────────────────────


def card(title: str, subtitle: str, badges: Sequence[str], *blocks: str) -> str:
    """A self-contained HTML document for ``flyte.artifacts.Card.create_from(content=..., format="html")``."""
    chips = "".join(f"<span class='badge'>{esc(b)}</span>" for b in badges)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{esc(title)}</title>{STYLE}"
        "<style>body{margin:0;background:#f9f9f7}@media (prefers-color-scheme: dark){body{background:#0d0d0d}}"
        ".viz{border-radius:0;min-height:100vh;max-width:900px;margin:0 auto}</style></head><body>"
        f"<div class='viz'><h2>{esc(title)}</h2><p class='sub'>{subtitle}</p><div>{chips}</div>{''.join(blocks)}</div>"
        "</body></html>"
    )
