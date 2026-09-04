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
    # Part-2 (clustered/async) fields — all defaulted so Part-1 rows and old checkpoints still load.
    eval_accuracy: float | None = None  # held-out greedy-eval accuracy after this iteration's step
    mean_ratio: float | None = None  # mean importance ratio pi_new/pi_old (staleness signal)
    clip_fraction: float | None = None  # fraction of tokens outside the 1±eps clip window
    mean_kl: float | None = None  # mean k3 KL vs the adapter-disabled reference
    gen_seconds: float | None = None  # wall time to generate+score this iteration's rollouts
    train_seconds: float | None = None  # wall time of the clustered train step
    iter_seconds: float | None = None  # end-to-end iteration wall time (shows pipelining overlap)


def wrap_report(body: str) -> str:
    """Wrap an HTML fragment with the report stylesheet + container div."""
    return f'{REPORT_CSS}<div class="report">{body}</div>'


def _tick_label(value: float, step: float) -> str:
    """Format an axis tick with just enough precision that ticks `step` apart read as different.

    A fixed ``.2f`` collapses small-range series (KL ≈ 0.0007–0.0046, clip fraction ≈ 0.006) onto
    identical labels, so the axis looks frozen while the line moves. Show two significant figures
    of the tick spacing instead: step 0.00078 → 5 decimals, step 0.2 → 2, step 2500 → 0. Beyond six
    decimals fall back to scientific notation.
    """
    import math

    if step <= 0 or not math.isfinite(step):
        return f"{value:.2f}"
    decimals = max(0, 1 - math.floor(math.log10(step)))
    if decimals > 6:
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


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

    x_vals = [d[x_key] for d in data] if data else []
    x_min, x_max = (min(x_vals), max(x_vals)) if x_vals else (0, 1)
    x_range = (x_max - x_min) or 1

    all_y = [d[k] for k in y_keys for d in data if k in d]
    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 1.0
    y_pad = (y_max - y_min) * 0.1 or 0.1
    y_min_plot = y_min - y_pad
    if y_min >= 0 > y_min_plot:
        y_min_plot = 0.0  # non-negative data (fractions, seconds): never pad below zero
    y_max_plot = y_max + y_pad
    y_range = (y_max_plot - y_min_plot) or 1
    y_step = y_range / 5
    y_tick_labels = [_tick_label(y_min_plot + y_step * i, y_step) for i in range(6)]

    # Left margin grows with the widest tick label (~6.5px per character at font-size 11).
    ml = max(60, 12 + int(6.5 * max(len(s) for s in y_tick_labels)))
    mr, mt, mb = 20, 40, 50
    cw = width - ml - mr
    ch = height - mt - mb

    def sx(v: float) -> float:
        return ml + (v - x_min) / x_range * cw

    def sy(v: float) -> float:
        return mt + ch - (v - y_min_plot) / y_range * ch

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Horizontal grid + y ticks (label precision follows the tick spacing — see _tick_label)
    for i, label in enumerate(y_tick_labels):
        py = sy(y_min_plot + y_step * i)
        lines.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" stroke="#e9ecef" stroke-width="1"/>')
        lines.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#6c757d">{label}</text>'
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
    extra: dict | None = None,
) -> str:
    """Render the full GRPO progress report (returns an HTML body for ``flyte.report.replace``).

    ``extra`` is Part-2 run config (dataset, pipelined, clip_eps, kl_beta, replicas,
    baseline_accuracy). When present it switches the objective note to the clipped-ratio + KL form
    and adds the eval / staleness / timing charts (each rendered only if its data exists).
    """
    latest = history[-1] if history else None
    prev = history[-2] if len(history) >= 2 else None
    extra = extra or {}
    baseline_accuracy = extra.get("baseline_accuracy")

    # --- Header + run config ---
    parts: list[str] = []
    badge = "✅ complete" if status == "complete" else "⏳ running"
    parts.append(f"<h2>GRPO + LoRA training progress &nbsp;<span class='muted'>({badge})</span></h2>")
    config_bits = [
        f"<b>Base model:</b> <code>{html.escape(base_model)}</code>",
        f"<b>Iterations:</b> {len(history)}/{num_iterations}",
        f"<b>Group size (G):</b> {group_size}",
        f"<b>Prompts/iter:</b> {prompts_per_iter}",
        f"<b>LoRA rank:</b> {lora_rank}",
        f"<b>LR:</b> {learning_rate:g}",
    ]
    for key in ("profile", "dataset", "pipelined", "replicas", "nproc_per_node", "clip_eps", "kl_beta"):
        if key in extra:
            config_bits.append(f"<b>{html.escape(key)}:</b> {html.escape(str(extra[key]))}")
    parts.append("<div class='card'>" + " &nbsp;|&nbsp; ".join(config_bits) + "</div>")
    if "clip_eps" in extra:
        parts.append(
            "<div class='note'><b>Objective:</b> token-level clipped-ratio GRPO — maximize "
            "<code>mean&#8203;<sub>t</sub> min( r<sub>t</sub> A&#770;, clip(r<sub>t</sub>, 1&minus;&epsilon;, "
            "1+&epsilon;) A&#770; ) &minus; &beta;&middot;KL<sub>k3</sub>(&pi;<sub>&theta;</sub> &Vert; "
            "&pi;<sub>ref</sub>)</code> with <code>r<sub>t</sub> = &pi;<sub>&theta;</sub>(o<sub>t</sub>) / "
            "&pi;<sub>old</sub>(o<sub>t</sub>)</code> from vLLM sampling-time logprobs and the reference = "
            "the adapter-disabled base. The importance ratio is what makes the one-step-off-policy "
            "(pipelined) training sound.</div>"
        )
    else:
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
        if latest.eval_accuracy is not None:
            # Delta vs the pre-training baseline (not vs the previous iteration) — the headline number.
            cards.insert(
                2,
                (
                    "Eval accuracy (held-out)"
                    + (f" — base {baseline_accuracy * 100:.1f}%" if baseline_accuracy is not None else ""),
                    f"{latest.eval_accuracy * 100:.1f}%",
                    _delta_html(latest.eval_accuracy, baseline_accuracy, "{:+.1%}"),
                ),
            )
        for label, value, delta in cards:
            parts.append(f"<div class='stat'><div class='value'>{value}</div>{delta}<div class='label'>{label}</div></div>")
        parts.append("</div>")

    # --- Charts ---
    # Optional Part-2 keys are added only when set; make_line_chart skips rows missing a key, so
    # mixed-cadence series (e.g. eval every K iterations) degrade gracefully.
    chart_data: list[dict] = []
    for m in history:
        row: dict = {
            "iter": m.iteration,
            "mean_reward": m.mean_reward,
            "accuracy": m.accuracy,
            "format_rate": m.format_rate,
            "mean_loss": m.mean_loss,
        }
        if m.eval_accuracy is not None:
            row["eval_accuracy"] = m.eval_accuracy
        if m.mean_ratio is not None:
            row["ratio_dev"] = abs(m.mean_ratio - 1.0)  # |mean ratio − 1|: distance off-policy
        if m.clip_fraction is not None:
            row["clip_fraction"] = m.clip_fraction
        if m.mean_kl is not None:
            row["mean_kl"] = m.mean_kl
        if m.gen_seconds is not None:
            row["gen_seconds"] = m.gen_seconds
        if m.train_seconds is not None:
            row["train_seconds"] = m.train_seconds
        if m.iter_seconds is not None:
            row["iter_seconds"] = m.iter_seconds
        chart_data.append(row)
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

    # --- Part-2 charts (rendered only when their series exist in the data) ---
    if any("eval_accuracy" in d for d in chart_data):
        parts.append("<div class='chart-container'>")
        if baseline_accuracy is not None:
            parts.append(
                f"<p class='muted'>Held-out eval, greedy decoding. Pre-training baseline (base model, "
                f"no adapter): <b>{baseline_accuracy * 100:.1f}%</b>.</p>"
            )
        parts.append(
            make_line_chart(
                chart_data,
                x_key="iter",
                y_keys=["eval_accuracy"],
                title="Held-out eval accuracy vs. iteration",
                x_label="iteration",
                colors=["#06d6a0"],
            )
        )
        parts.append("</div>")
    if any(k in d for k in ("ratio_dev", "clip_fraction", "mean_kl") for d in chart_data):
        parts.append("<div class='chart-container'>")
        parts.append(
            "<p class='muted'>Staleness / trust-region signals: with pipelining the trainer consumes "
            "rollouts one adapter version old, so ratios drift off 1 and the clip engages — this chart "
            "is what makes the async claim measurable.</p>"
        )
        parts.append(
            make_line_chart(
                chart_data,
                x_key="iter",
                y_keys=["ratio_dev", "clip_fraction", "mean_kl"],
                title="Off-policy drift vs. iteration",
                x_label="iteration",
                y_display_names={
                    "ratio_dev": "|mean ratio − 1|",
                    "clip_fraction": "clip fraction",
                    "mean_kl": "KL(policy ‖ ref)",
                },
            )
        )
        parts.append("</div>")
    if any(k in d for k in ("gen_seconds", "train_seconds", "iter_seconds") for d in chart_data):
        timed = [d for d in chart_data if "iter_seconds" in d]
        parts.append("<div class='chart-container'>")
        if timed:
            saved = sum(d.get("gen_seconds", 0.0) + d.get("train_seconds", 0.0) - d["iter_seconds"] for d in timed)
            parts.append(
                f"<p class='muted'>Wall-clock per iteration. Overlap hidden by pipelining so far: "
                f"<b>{saved:.0f}s</b> (Σ gen + train − iter; ≈0 for a sequential run).</p>"
            )
        parts.append(
            make_line_chart(
                chart_data,
                x_key="iter",
                y_keys=["gen_seconds", "train_seconds", "iter_seconds"],
                title="Iteration timing (generation vs. training vs. end-to-end)",
                x_label="iteration",
                y_label="seconds",
                y_display_names={
                    "gen_seconds": "generate+score",
                    "train_seconds": "clustered train step",
                    "iter_seconds": "iteration end-to-end",
                },
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
        eval_bit = ""
        if last.eval_accuracy is not None and baseline_accuracy is not None:
            eval_bit = (
                f" Held-out eval accuracy: <b>{baseline_accuracy * 100:.1f}% (base) → "
                f"{last.eval_accuracy * 100:.1f}%</b> ({last.eval_accuracy - baseline_accuracy:+.1%})."
            )
        parts.append("<h2>Summary</h2>")
        parts.append(
            "<div class='card'>"
            f"Trained <b>{len(history)}</b> GRPO iterations on <code>{html.escape(base_model)}</code>. "
            f"Mean reward moved <b>{first.mean_reward:.3f} → {last.mean_reward:.3f}</b> "
            f"({last.mean_reward - first.mean_reward:+.3f}); "
            f"accuracy <b>{first.accuracy * 100:.1f}% → {last.accuracy * 100:.1f}%</b>; "
            f"format adherence <b>{first.format_rate * 100:.1f}% → {last.format_rate * 100:.1f}%</b>. "
            f"Final adapter: <b>v{last.adapter_version}</b>." + eval_bit + "</div>"
        )

    return wrap_report("".join(parts))
