"""HTML report helpers for the GRPO loop.

Pure-python (no extra dependencies) so the CPU driver task stays light. Builds a single self-contained
HTML body for ``flyte.report`` that shows per-iteration GRPO progress (reward, accuracy, format
adherence, loss) plus a running summary. The driver rebuilds and re-publishes this every iteration via
``flyte.report.replace.aio(render_report(...), do_flush=True)`` so progress streams live in the UI.

The visual style intentionally mirrors the sibling ``llm_fine_tuning_lora_qlora`` example.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #0f3460; margin-top: 20px; }
  .report .card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0f3460; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report .stat .delta-up { color: #1e7e34; font-size: 0.8em; }
  .report .stat .delta-down { color: #b02a37; font-size: 0.8em; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #0f3460; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #dee2e6; }
  .report tr:nth-child(even) { background: #f8f9fa; }
  .report .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .chart-container { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report pre.sample { background: #1a1a2e; color: #e6e6e6; padding: 12px 14px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; font-size: 0.85em; line-height: 1.4; }
  .report .muted { color: #6c757d; font-size: 0.85em; }
</style>
"""


@dataclass
class IterationMetrics:
    """One row of GRPO progress, accumulated by the driver after each outer step."""

    iteration: int
    adapter_version: int
    num_rollouts: int
    mean_reward: float
    max_reward: float
    accuracy: float  # fraction of completions with the exactly-correct answer
    format_rate: float  # fraction of completions that emitted the '####' marker
    mean_loss: float  # mean GRPO loss reported by train_step
    contributing: int  # rollouts that produced a gradient (non-zero advantage)
    sample_question: str
    sample_completion: str
    sample_reward: float


def wrap_report(body: str) -> str:
    """Wrap an HTML fragment with the report stylesheet + container div."""
    return f'{REPORT_CSS}<div class="report">{body}</div>'


def make_line_chart(
    data: list[dict],
    x_key: str,
    y_keys: list[str],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    y_display_names: dict[str, str] | None = None,
    colors: list[str] | None = None,
    width: int = 720,
    height: int = 300,
) -> str:
    """Generate a self-contained SVG line chart from a list of dicts (no plotting deps)."""
    colors = colors or ["#0f3460", "#06d6a0", "#ffc107", "#5a7db5", "#6c757d"]

    ml, mr, mt, mb = 60, 20, 40, 50
    cw = width - ml - mr
    ch = height - mt - mb

    x_vals = [d[x_key] for d in data] if data else []
    x_min, x_max = (min(x_vals), max(x_vals)) if x_vals else (0, 1)
    x_range = (x_max - x_min) or 1

    all_y = [d[k] for k in y_keys for d in data if k in d]
    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 1.0
    y_pad = (y_max - y_min) * 0.1 or 0.1
    y_min_plot = y_min - y_pad
    y_max_plot = y_max + y_pad
    y_range = (y_max_plot - y_min_plot) or 1

    def sx(v: float) -> float:
        return ml + (v - x_min) / x_range * cw

    def sy(v: float) -> float:
        return mt + ch - (v - y_min_plot) / y_range * ch

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Horizontal grid + y ticks
    for i in range(6):
        y_tick = y_min_plot + y_range * i / 5
        py = sy(y_tick)
        lines.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" stroke="#e9ecef" stroke-width="1"/>')
        lines.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#6c757d">{y_tick:.2f}</text>'
        )

    # Axes
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#adb5bd" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#adb5bd" stroke-width="1.5"/>')

    # X ticks (integer iteration labels)
    for i, xv in enumerate(x_vals):
        px = sx(xv)
        lines.append(
            f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" font-size="11" fill="#6c757d">{int(xv)}</text>'
        )

    if not data:
        lines.append(
            f'<text x="{ml + cw / 2}" y="{mt + ch / 2}" text-anchor="middle" font-size="13" '
            f'fill="#adb5bd" font-style="italic">Waiting for the first iteration...</text>'
        )

    # Series
    for si, key in enumerate(y_keys):
        color = colors[si % len(colors)]
        points = [(sx(d[x_key]), sy(d[key])) for d in data if key in d]
        if not points:
            continue
        if len(points) >= 2:
            path_d = f"M {points[0][0]:.1f},{points[0][1]:.1f}" + "".join(
                f" L {px:.1f},{py:.1f}" for px, py in points[1:]
            )
            lines.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for px, py in points:
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')

    if title:
        lines.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" '
            f'fill="#1a1a2e">{html.escape(title)}</text>'
        )
    if x_label:
        lines.append(
            f'<text x="{ml + cw / 2}" y="{height - 6}" text-anchor="middle" font-size="12" '
            f'fill="#6c757d">{html.escape(x_label)}</text>'
        )
    if y_label:
        lines.append(
            f'<text x="14" y="{mt + ch / 2}" text-anchor="middle" font-size="12" fill="#6c757d" '
            f'transform="rotate(-90, 14, {mt + ch / 2})">{html.escape(y_label)}</text>'
        )

    # Legend
    names = y_display_names or {}
    if len(y_keys) > 1:
        for si, key in enumerate(y_keys):
            color = colors[si % len(colors)]
            ly = mt + 14 + si * 18
            lines.append(f'<rect x="{ml + 10}" y="{ly - 6}" width="12" height="12" rx="2" fill="{color}"/>')
            lines.append(
                f'<text x="{ml + 28}" y="{ly + 4}" font-size="11" fill="#1a1a2e">{html.escape(names.get(key, key))}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


def _delta_html(curr: float, prev: float | None, fmt: str = "{:+.3f}") -> str:
    if prev is None:
        return ""
    d = curr - prev
    cls = "delta-up" if d >= 0 else "delta-down"
    arrow = "▲" if d >= 0 else "▼"
    return f'<div class="{cls}">{arrow} {fmt.format(d)}</div>'


def render_report(
    history: list[IterationMetrics],
    *,
    base_model: str,
    num_iterations: int,
    group_size: int,
    prompts_per_iter: int,
    lora_rank: int,
    learning_rate: float,
    status: str = "running",
) -> str:
    """Render the full GRPO progress report (returns an HTML body for ``flyte.report.replace``)."""
    latest = history[-1] if history else None
    prev = history[-2] if len(history) >= 2 else None

    # --- Header + run config ---
    parts: list[str] = []
    badge = "✅ complete" if status == "complete" else "⏳ running"
    parts.append(f"<h2>GRPO + LoRA training progress &nbsp;<span class='muted'>({badge})</span></h2>")
    parts.append(
        "<div class='card'>"
        f"<b>Base model:</b> <code>{html.escape(base_model)}</code> &nbsp;|&nbsp; "
        f"<b>Iterations:</b> {len(history)}/{num_iterations} &nbsp;|&nbsp; "
        f"<b>Group size (G):</b> {group_size} &nbsp;|&nbsp; "
        f"<b>Prompts/iter:</b> {prompts_per_iter} &nbsp;|&nbsp; "
        f"<b>LoRA rank:</b> {lora_rank} &nbsp;|&nbsp; "
        f"<b>LR:</b> {learning_rate:g}"
        "</div>"
    )
    parts.append(
        "<div class='note'><b>Objective:</b> single-step, group-normalized GRPO policy gradient — "
        "maximize <code>mean( A&#770;<sub>i</sub> &middot; mean&#8203;<sub>t</sub> log &pi;<sub>&theta;</sub>"
        "(o<sub>i,t</sub>) )</code> with "
        "<code>A&#770;<sub>i</sub> = (r<sub>i</sub> &minus; mean&#8203;<sub>group</sub>) / "
        "(std&#8203;<sub>group</sub> + &epsilon;)</code> over completions, training the LoRA adapter only. "
        "(No PPO clip / no KL term in this MVP — see <code>README.md</code>.)</div>"
    )

    # --- Summary stat cards (latest values, with delta vs previous iter) ---
    if latest is not None:
        parts.append("<h3>Latest iteration</h3>")
        parts.append("<div class='stat-grid'>")
        cards = [
            ("Mean reward", f"{latest.mean_reward:.3f}", _delta_html(latest.mean_reward, prev.mean_reward if prev else None)),
            ("Accuracy", f"{latest.accuracy * 100:.1f}%", _delta_html(latest.accuracy, prev.accuracy if prev else None, "{:+.1%}")),
            ("Format rate", f"{latest.format_rate * 100:.1f}%", _delta_html(latest.format_rate, prev.format_rate if prev else None, "{:+.1%}")),
            ("Mean loss", f"{latest.mean_loss:.4f}", _delta_html(latest.mean_loss, prev.mean_loss if prev else None, "{:+.4f}")),
            ("Adapter version", f"v{latest.adapter_version}", ""),
        ]
        for label, value, delta in cards:
            parts.append(f"<div class='stat'><div class='value'>{value}</div>{delta}<div class='label'>{label}</div></div>")
        parts.append("</div>")

    # --- Charts ---
    chart_data = [
        {
            "iter": m.iteration,
            "mean_reward": m.mean_reward,
            "accuracy": m.accuracy,
            "format_rate": m.format_rate,
            "mean_loss": m.mean_loss,
        }
        for m in history
    ]
    parts.append("<div class='chart-container'>")
    parts.append(
        make_line_chart(
            chart_data,
            x_key="iter",
            y_keys=["mean_reward", "accuracy", "format_rate"],
            title="Reward & correctness vs. iteration",
            x_label="iteration",
            y_display_names={"mean_reward": "mean reward", "accuracy": "accuracy", "format_rate": "format rate"},
        )
    )
    parts.append("</div>")
    parts.append("<div class='chart-container'>")
    parts.append(
        make_line_chart(
            chart_data,
            x_key="iter",
            y_keys=["mean_loss"],
            title="GRPO loss vs. iteration",
            x_label="iteration",
            colors=["#b02a37"],
        )
    )
    parts.append("</div>")

    # --- Per-iteration table ---
    parts.append("<h3>Per-iteration metrics</h3>")
    parts.append(
        "<table><tr><th>Iter</th><th>Adapter</th><th>Rollouts</th><th>Mean reward</th>"
        "<th>Max reward</th><th>Accuracy</th><th>Format</th><th>Mean loss</th><th>Contributing</th></tr>"
    )
    for m in history:
        parts.append(
            f"<tr><td>{m.iteration}</td><td>v{m.adapter_version}</td><td>{m.num_rollouts}</td>"
            f"<td>{m.mean_reward:.3f}</td><td>{m.max_reward:.3f}</td>"
            f"<td>{m.accuracy * 100:.1f}%</td><td>{m.format_rate * 100:.1f}%</td>"
            f"<td>{m.mean_loss:.4f}</td><td>{m.contributing}/{m.num_rollouts}</td></tr>"
        )
    parts.append("</table>")

    # --- Best sample from the latest iteration (qualitative signal) ---
    if latest is not None:
        parts.append("<h3>Best completion (latest iteration)</h3>")
        parts.append(
            f"<p class='muted'>Question: <b>{html.escape(latest.sample_question)}</b> &nbsp;|&nbsp; "
            f"reward = {latest.sample_reward:.2f}</p>"
        )
        parts.append(f"<pre class='sample'>{html.escape(latest.sample_completion.strip()) or '(empty)'}</pre>")

    # --- Final summary ---
    if status == "complete" and history:
        first, last = history[0], history[-1]
        parts.append("<h2>Summary</h2>")
        parts.append(
            "<div class='card'>"
            f"Trained <b>{len(history)}</b> GRPO iterations on <code>{html.escape(base_model)}</code>. "
            f"Mean reward moved <b>{first.mean_reward:.3f} → {last.mean_reward:.3f}</b> "
            f"({last.mean_reward - first.mean_reward:+.3f}); "
            f"accuracy <b>{first.accuracy * 100:.1f}% → {last.accuracy * 100:.1f}%</b>; "
            f"format adherence <b>{first.format_rate * 100:.1f}% → {last.format_rate * 100:.1f}%</b>. "
            f"Final adapter: <b>v{last.adapter_version}</b>."
            "</div>"
        )

    return wrap_report("".join(parts))
