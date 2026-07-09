"""Reporting, directives, and Flyte UI HTML for parallelized autoresearch."""

from __future__ import annotations

import difflib
import html
import json
import re
from typing import Any

from autoresearch_types import DEFAULT_MAX_STEPS, DatasetProfile, LeaderboardEntry
from tools import (
    baseline_train_py,
    build_train_py_with_config_overrides,
    format_research_history_for_directive,
    normalize_train_py,
)



def _entry_from_result(data: dict[str, Any], index: int, running_min: float) -> tuple[LeaderboardEntry, float, LeaderboardEntry | None]:
    """Build a :class:`LeaderboardEntry` from a ``run_experiment`` result dict."""
    best: LeaderboardEntry | None = None
    if data.get("success") is False or data.get("val_bpb") is None:
        err = data.get("error") or (data.get("stderr") or "")[:200]
        return (
            LeaderboardEntry(
                index=index,
                title=str(data.get("title", f"exp-{index}")),
                error=str(err)[:200] if err else "failed",
            ),
            running_min,
            None,
        )
    val = float(data["val_bpb"])
    kept = val < running_min
    entry = LeaderboardEntry(
        index=index,
        title=str(data.get("title", f"exp-{index}")),
        val_bpb=val,
        model_name=data.get("model_name"),
        n_params=data.get("n_params"),
        steps=int(data["steps"]) if data.get("steps") is not None else None,
        resources=data.get("resources"),
        oom_retries=int(data.get("oom_retries", 0)),
        kept=kept,
    )
    if kept:
        running_min = val
        best = entry
    return entry, running_min, best


def _extract_json_payload(text: str) -> Any:
    """Parse JSON from a tool message or code-mode ``Execution result:`` observation."""
    payload = text.strip()
    if payload.startswith("Execution result:"):
        payload = payload.split("Execution result:", 1)[1].strip()
    if not payload or payload == "(no value)":
        return None
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _collect_experiment_results(obj: Any, out: list[dict[str, Any]], seen: set[tuple[Any, ...]]) -> None:
    """Walk nested batch / evaluation payloads and collect run result dicts."""
    if isinstance(obj, dict):
        if obj.get("val_bpb") is not None and obj.get("title"):
            key = (str(obj.get("title")), float(obj.get("val_bpb")))
            if key not in seen:
                seen.add(key)
                out.append(obj)
        for nested_key in ("results", "ranked", "evaluation", "best"):
            nested = obj.get(nested_key)
            if nested is not None:
                _collect_experiment_results(nested, out, seen)
    elif isinstance(obj, list):
        for item in obj:
            _collect_experiment_results(item, out, seen)


def _leaderboard_from_result_dicts(raw: list[dict[str, Any]]) -> tuple[list[LeaderboardEntry], LeaderboardEntry | None]:
    entries: list[LeaderboardEntry] = []
    running_min = float("inf")
    best: LeaderboardEntry | None = None
    for idx, data in enumerate(raw, start=1):
        entry, running_min, entry_best = _entry_from_result(data, idx, running_min)
        if entry_best is not None:
            best = entry_best
        entries.append(entry)
    return entries, best


def parse_leaderboard(
    messages: list[dict[str, Any]],
    *,
    promising_fallback: list[dict[str, Any]] | None = None,
) -> tuple[list[LeaderboardEntry], LeaderboardEntry | None]:
    """Build a leaderboard from ``run_experiment`` results in the agent transcript.

    Handles both JSON tool-call mode (``role=tool``, ``name=run_experiment``) and
    **code mode**, where results appear inside ``Execution result:`` user messages
    from ``run_experiment_batch`` / ``flyte_map`` observations.
    """
    entries: list[LeaderboardEntry] = []
    idx = 0
    running_min = float("inf")
    best: LeaderboardEntry | None = None
    for msg in messages:
        if msg.get("role") != "tool" or msg.get("name") != "run_experiment":
            continue
        idx += 1
        content = msg.get("content", "")
        try:
            data = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            entries.append(LeaderboardEntry(index=idx, title="(failed)", error=str(content)[:200]))
            continue
        if not isinstance(data, dict):
            entries.append(LeaderboardEntry(index=idx, title="(failed)", error=str(content)[:200]))
            continue
        entry, running_min, entry_best = _entry_from_result(data, idx, running_min)
        if entry_best is not None:
            best = entry_best
        entries.append(entry)

    if entries:
        return entries, best

    seen: set[tuple[Any, ...]] = set()
    raw: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if role == "tool" and msg.get("name") in ("run_experiment_batch", "evaluate_batch_results"):
            parsed = _extract_json_payload(content)
            if parsed is not None:
                _collect_experiment_results(parsed, raw, seen)
        elif role == "user" and "Execution result:" in content:
            parsed = _extract_json_payload(content)
            if parsed is not None:
                _collect_experiment_results(parsed, raw, seen)

    if not raw and promising_fallback:
        for row in promising_fallback:
            val = row.get("val_bpb")
            if val is None:
                continue
            title = str(row.get("title", "experiment"))
            key = (title, float(val))
            if key in seen:
                continue
            seen.add(key)
            raw.append(
                {
                    "success": True,
                    "title": title,
                    "val_bpb": float(val),
                    "model_name": row.get("model_name"),
                    "steps": row.get("steps"),
                    "resources": row.get("resources"),
                    "oom_retries": row.get("oom_retries", 0),
                }
            )

    return _leaderboard_from_result_dicts(raw)


def directive_code_edit_fanout(
    n_experiments: int,
    profile: DatasetProfile,
    memory_key: str,
    *,
    batch_size: int = 3,
    max_batches: int | None = None,
    history: dict[str, Any] | None = None,
) -> str:
    """Build the user directive for the code-mode fan-out agent."""
    if max_batches is None:
        max_batches = max(1, (n_experiments + batch_size - 1) // batch_size)

    history_block = format_research_history_for_directive(history or {})

    return (
        f"Run {n_experiments} code-edit experiments on climbmix "
        f"({profile.n_parquet_files} shards, vocab_size={profile.vocab_size}) using "
        f"**batched parallel fan-out**. Work in up to {max_batches} batch(es) of "
        f"{batch_size} hypotheses at a time.\n\n"
        f"Use memory_key={memory_key!r} for all memory-backed tools.\n\n"
        "Workflow (CODE MODE — write Python plans each turn):\n"
        "1. ``get_code_edit_history()`` (if prior trials exist) + ``get_baseline_train_code`` "
        "+ ``inspect_dataset``; optionally ``search_arxiv``.\n"
        "2. Plan a batch: ``record_batch_plan(batch_id, experiments=[...])``.\n"
        "3. **Batch 1:** ``edit_train_code_batch(edits=[...])`` may use ``config_overrides`` "
        "for architecture/LR sweeps. **Batch 2+:** each edit must include substantive "
        "``train_py`` changes (LR schedule, optimizer, weight decay, grad clip) — "
        "``config_overrides`` alone is rejected.\n"
        "4. ``record_batch_hypotheses([...])`` then ``run_experiment_batch(titles, ...)`` "
        f"OR ``flyte_map('run_experiment', titles, budgets, keys, concurrency={batch_size})``.\n"
        "5. ``run_experiment_batch`` returns ``evaluation`` and ``seed`` — the platform "
        "advances ``seed`` to the batch best for the next round.\n"
        "6. Iterate: the next batch forks from ``seed`` automatically until "
        f"{n_experiments} experiments complete.\n"
        "7. Finish with a plain-text summary: best val_bpb, winning code changes, next batch idea.\n\n"
        f"**Batch diversity:** each parallel run must test a different hypothesis — spread "
        f"changes across training-loop code (batch 2+), depth/width, dropout, and batch size. "
        f"No duplicate configs (rejected at run time); no LR micro-sweeps within ±30% of best.\n\n"
        "**Plateau rule:** if 3 consecutive batches fail to beat the global best val_bpb by "
        ">0.01, stop hyperparameter sweeps and edit ``train.py`` (scheduler, optimizer, etc.).\n\n"
        "Do not repeat experiments already listed in prior research below. "
        "After each batch, the platform seeds the next round from the batch best; "
        "use ``get_seed_train_code()`` or ``read_train_code(seed_title)`` before designing edits.\n\n"
        f"time_budget_sec=45, max_steps={DEFAULT_MAX_STEPS} (default). "
        f"time_budget is a safety cap; max_steps ensures fair comparison across architectures. "
        f"Platform retries sandbox OOM with more memory per run."
        f"{history_block}"
    )


# Design tokens — keep in sync with the Slidev deck (#FDB51F gold)
GOLD = "#FDB51F"
GOLD_DARK = "#E5A31B"
GREEN = "#10b981"
GREEN_BG = "#ecfdf5"
GREEN_TEXT = "#065f46"
RED = "#ef4444"
RED_BG = "#fef2f2"
SLATE = "#64748b"
SLATE_LIGHT = "#94a3b8"
INK = "#0f172a"
BORDER = "#e2e8f0"
SURFACE = "#f8fafc"
SURFACE_ALT = "#f1f5f9"
FONT = "system-ui,-apple-system,'Segoe UI',Roboto,sans-serif"
MONO = "ui-monospace,'Cascadia Code','SF Mono',Menlo,monospace"
RADIUS = "10px"
SHADOW = "0 1px 3px rgba(15,23,42,.08),0 1px 2px rgba(15,23,42,.04)"


def _esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def _styles() -> str:
    """Shared inline CSS — Flyte reports accept raw HTML."""
    return f"""
<style>
  .ar {{ font-family:{FONT}; color:{INK}; line-height:1.55; }}
  .ar h2 {{ margin:0 0 12px; font-size:1.25rem; font-weight:700; color:{INK}; }}
  .ar h3 {{ margin:18px 0 8px; font-size:.95rem; font-weight:600; color:{SLATE}; text-transform:uppercase; letter-spacing:.04em; }}
  .ar code {{ font-family:{MONO}; font-size:.85em; background:{SURFACE_ALT}; padding:2px 6px; border-radius:4px; }}
  .ar-card {{ background:#fff; border:1px solid {BORDER}; border-radius:{RADIUS}; padding:16px 18px; box-shadow:{SHADOW}; margin-bottom:14px; }}
  .ar-stat {{ display:inline-block; min-width:120px; padding:10px 14px; background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px; margin-right:10px; margin-bottom:8px; }}
  .ar-stat b {{ display:block; font-size:1.35rem; color:{INK}; font-variant-numeric:tabular-nums; }}
  .ar-stat span {{ font-size:.75rem; color:{SLATE}; text-transform:uppercase; letter-spacing:.05em; }}
  .ar-badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:.72rem; font-weight:600; letter-spacing:.02em; }}
  .ar-badge--ok {{ background:{GREEN_BG}; color:{GREEN_TEXT}; }}
  .ar-badge--fail {{ background:{RED_BG}; color:#991b1b; }}
  .ar-badge--muted {{ background:{SURFACE_ALT}; color:{SLATE}; }}
  .ar-badge--gold {{ background:#fffbeb; color:#92400e; }}
  .ar-table {{ width:100%; border-collapse:separate; border-spacing:0; font-size:.875rem; border:1px solid {BORDER}; border-radius:{RADIUS}; overflow:hidden; }}
  .ar-table th {{ background:{SURFACE_ALT}; text-align:left; padding:10px 12px; font-weight:600; color:{SLATE}; border-bottom:1px solid {BORDER}; }}
  .ar-table td {{ padding:9px 12px; border-bottom:1px solid {BORDER}; vertical-align:top; }}
  .ar-table tr:last-child td {{ border-bottom:none; }}
  .ar-table tr.ar-best td {{ background:{GREEN_BG}; }}
  .ar-timeline {{ list-style:none; margin:0; padding:0; }}
  .ar-timeline li {{ position:relative; padding:8px 0 8px 28px; border-left:2px solid {BORDER}; margin-left:8px; }}
  .ar-timeline li:last-child {{ border-left-color:transparent; }}
  .ar-timeline .ar-dot {{ position:absolute; left:-7px; top:12px; width:12px; height:12px; border-radius:50%; background:#fff; border:2px solid {BORDER}; }}
  .ar-timeline .ar-dot--tool {{ border-color:{GOLD}; background:#fffbeb; }}
  .ar-timeline .ar-dot--err {{ border-color:{RED}; background:{RED_BG}; }}
  .ar-timeline .ar-dot--done {{ border-color:{GREEN}; background:{GREEN_BG}; }}
  .ar-kv {{ display:grid; grid-template-columns:140px 1fr; gap:6px 12px; font-size:.875rem; }}
  .ar-kv dt {{ color:{SLATE}; margin:0; }}
  .ar-kv dd {{ margin:0; }}
  .ar-hero {{ background:linear-gradient(135deg,#fffbeb 0%,#fff 60%); border:1px solid #fde68a; border-radius:{RADIUS}; padding:18px 20px; margin-bottom:16px; }}
  .ar-hero h1 {{ margin:0 0 6px; font-size:1.5rem; color:{GOLD_DARK}; }}
  .ar-pre {{ white-space:pre-wrap; word-break:break-word; background:{SURFACE}; padding:12px 14px; border-radius:8px; border:1px solid {BORDER}; font-family:{MONO}; font-size:.8rem; margin:0; }}
  .ar-code {{ max-height:360px; overflow:auto; }}
  .ar-code-card {{ border-left:4px solid {BORDER}; }}
  .ar-code-card--best {{ border-left-color:{GREEN}; background:{GREEN_BG}; }}
  .ar-diff .ar-diff-add {{ display:block; color:{GREEN_TEXT}; background:{GREEN_BG}; }}
  .ar-diff .ar-diff-del {{ display:block; color:#991b1b; background:{RED_BG}; }}
  .ar-diff .ar-diff-hunk {{ display:block; color:{SLATE}; background:{SURFACE_ALT}; }}
</style>
"""


def _wrap(body: str) -> str:
    return f'<div class="ar">{_styles()}{body}</div>'


def _badge(label: str, kind: str = "muted") -> str:
    return f'<span class="ar-badge ar-badge--{kind}">{_esc(label)}</span>'


def _stat(value: str, label: str) -> str:
    return f'<div class="ar-stat"><b>{_esc(value)}</b><span>{_esc(label)}</span></div>'


# ---------------------------------------------------------------------------
# Leaderboard chart (val_bpb vs experiment index; lower is better)
# ---------------------------------------------------------------------------


def _val_bpb_svg(points: list[tuple[int, float, str]]) -> str:
    if not points:
        return f'<p style="color:{SLATE};margin:0">No completed experiments yet.</p>'

    xs = [float(p[0]) for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin_raw, ymax_raw = min(ys), max(ys)
    span = ymax_raw - ymin_raw
    pad = max(span * 0.1, 1e-9) if span > 0 else max(abs(ymin_raw) * 0.05, 0.01)
    ymin, ymax = ymin_raw - pad, ymax_raw + pad
    if xmax <= xmin:
        xmax = xmin + 1.0

    pad_l, pad_r, pad_t, pad_b = 56, 28, 28, 44
    w, h = 640, 320
    gw = w - pad_l - pad_r
    gh = h - pad_t - pad_b

    def sx(x: float) -> float:
        return pad_l + (x - xmin) / (xmax - xmin) * gw

    def sy(y: float) -> float:
        return pad_t + gh - (y - ymin) / (ymax - ymin) * gh

    best_indices: list[int] = []
    running_min = float("inf")
    for i, (_, y, _) in enumerate(points):
        if y < running_min:
            running_min = y
            best_indices.append(i)
    best_idx_set = set(best_indices)
    best_points = [points[i] for i in best_indices]

    best_step_points: list[tuple[float, float]] = []
    if best_points:
        prev_x, prev_y, _ = best_points[0]
        best_step_points.append((float(prev_x), prev_y))
        for x, y, _ in best_points[1:]:
            best_step_points.append((float(x), prev_y))
            best_step_points.append((float(x), y))
            prev_x, prev_y = x, y
    line = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in best_step_points)

    grid_lines = "".join(
        f'<line x1="{pad_l}" y1="{sy(y):.2f}" x2="{pad_l + gw:.2f}" y2="{sy(y):.2f}" '
        f'stroke="#f1f5f9" stroke-width="1"/>'
        for y in [ymin_raw + span * i / 4 for i in range(5)] if span > 0
    )
    non_best_dots = "".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.5" fill="{SLATE_LIGHT}" opacity="0.85"/>'
        for i, (x, y, _) in enumerate(points)
        if i not in best_idx_set
    )
    best_dots = "".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5" fill="{GREEN}" stroke="#fff" stroke-width="1.5"/>'
        for x, y, _ in best_points
    )
    best_labels = "".join(
        (
            f'<text x="{sx(x) + 7:.2f}" y="{sy(y) - 8:.2f}" '
            f'font-size="9" font-weight="600" fill="{GREEN_TEXT}" '
            f'transform="rotate(-22 {sx(x) + 7:.2f} {sy(y) - 8:.2f})">'
            f"{html.escape((title or 'untitled')[:28])}</text>"
        )
        for x, y, title in best_points
    )
    y0 = sy(ymin)
    border = f"border:1px solid {BORDER};border-radius:{RADIUS}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="100%" style="max-width:{w}px;display:block;background:{SURFACE};{border}">\n'
        f"{grid_lines}\n"
        f'<line x1="{pad_l}" y1="{y0:.2f}" x2="{pad_l + gw:.2f}" y2="{y0:.2f}" stroke="{BORDER}" stroke-width="1"/>\n'
        f'<polyline fill="none" stroke="{GREEN}" stroke-width="2.5" stroke-linejoin="round" '
        f'stroke-linecap="round" points="{line}"/>\n'
        f"{non_best_dots}\n{best_dots}\n{best_labels}\n"
        f'<text x="{pad_l - 6}" y="{sy(ymin_raw):.1f}" text-anchor="end" dominant-baseline="middle" '
        f'font-size="11" fill="{SLATE}">{ymin_raw:.4g}</text>\n'
        f'<text x="{pad_l - 6}" y="{sy(ymax_raw):.1f}" text-anchor="end" dominant-baseline="middle" '
        f'font-size="11" fill="{SLATE}">{ymax_raw:.4g}</text>\n'
        f'<text transform="translate(14,{pad_t + gh / 2:.0f}) rotate(-90)" text-anchor="middle" '
        f'font-size="11" fill="{SLATE}">val_bpb ↓</text>\n'
        f'<text x="{pad_l + gw / 2:.0f}" y="{h - 6}" text-anchor="middle" font-size="11" '
        f'fill="{SLATE}">experiment #</text>\n'
        f'<circle cx="{pad_l + 10}" cy="{pad_t + 10}" r="3.5" fill="{SLATE_LIGHT}"/>'
        f'<text x="{pad_l + 18}" y="{pad_t + 13}" font-size="10" fill="{SLATE}">discarded</text>\n'
        f'<circle cx="{pad_l + 88}" cy="{pad_t + 10}" r="4" fill="{GREEN}"/>'
        f'<text x="{pad_l + 97}" y="{pad_t + 13}" font-size="10" fill="{GREEN_TEXT}">running best</text>\n'
        "</svg>"
    )


def _leaderboard_body(leaderboard: list[LeaderboardEntry], best: LeaderboardEntry | None) -> str:
    points = [(e.index, e.val_bpb, e.title) for e in leaderboard if e.val_bpb is not None]
    chart = _val_bpb_svg(points)

    rows: list[str] = []
    for e in leaderboard:
        val = f"{e.val_bpb:.6g}" if e.val_bpb is not None else "—"
        steps = str(e.steps) if e.steps is not None else "—"
        is_best = best is not None and e.index == best.index
        row_cls = ' class="ar-best"' if is_best else ""

        if e.error:
            status = _badge(e.error[:60] + ("…" if len(e.error) > 60 else ""), "fail")
        elif e.kept:
            status = _badge("kept", "ok")
        elif e.val_bpb is not None:
            status = _badge("ok", "muted")
        else:
            status = _badge("pending", "muted")

        oom = _badge(f"{e.oom_retries}× retry", "gold") if e.oom_retries else "—"
        title_cell = _esc(e.title)
        if is_best:
            title_cell += f' {_badge("best", "ok")}'

        rows.append(
            f"<tr{row_cls}>"
            f"<td>{e.index}</td>"
            f"<td><strong>{title_cell}</strong></td>"
            f"<td style='font-variant-numeric:tabular-nums;font-weight:600'>{val}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>{steps}</td>"
            f"<td>{_esc(e.model_name or '—')}</td>"
            f"<td><code>{_esc(e.resources or '—')}</code></td>"
            f"<td>{oom}</td>"
            f"<td>{status}</td>"
            f"</tr>"
        )

    body = "".join(rows) or f"<tr><td colspan='8' style='padding:14px;color:{SLATE}'>No experiments yet.</td></tr>"
    table = (
        '<table class="ar-table">'
        "<thead><tr>"
        "<th>#</th><th>Experiment</th><th>val_bpb</th><th>Steps</th><th>Model</th>"
        "<th>Resources</th><th>OOM</th><th>Status</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table>"
    )

    stats = ""
    if best is not None and best.val_bpb is not None:
        stats = (
            _stat(f"{best.val_bpb:.6g}", "best val_bpb")
            + _stat(_esc(best.title), "winning experiment")
            + _stat(str(len(leaderboard)), "total runs")
        )

    return (
        "<h2>Leaderboard</h2>\n"
        f'<div class="ar-card">{chart}</div>\n'
        f"{stats}\n"
        f"{table}\n"
    )


def render_leaderboard(leaderboard: list[LeaderboardEntry], best: LeaderboardEntry | None) -> str:
    return _wrap(_leaderboard_body(leaderboard, best))


# ---------------------------------------------------------------------------
# Live activity log (streamed from agent_progress_cb)
# ---------------------------------------------------------------------------

_EVENT_META: dict[str, tuple[str, str, str]] = {
    "agent_start": ("🚀", "muted", "Agent started"),
    "turn_start": ("🔄", "muted", "Turn"),
    "tool_start": ("🔧", "tool", "Tool call"),
    "tool_end": ("✅", "done", "Tool result"),
    "tool_error": ("💥", "err", "Tool error"),
    "message": ("💬", "muted", "Message"),
    "agent_end": ("🏁", "done", "Agent finished"),
}


def _parse_tool_result(result: str) -> Any | None:
    if not result or not (result.startswith("{") or result.startswith("[")):
        return None
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return None


def _truncate_text(text: str, *, max_lines: int = 20) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    omitted = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n… ({omitted} more lines)"


_BASELINE_TRAIN_PY: str | None = None


def _baseline_train_py() -> str:
    """Cached baseline ``train.py`` source for unified diffs."""
    global _BASELINE_TRAIN_PY
    if _BASELINE_TRAIN_PY is None:
        _BASELINE_TRAIN_PY = baseline_train_py()
    return _BASELINE_TRAIN_PY


def _effective_train_py_for_edit(edit: dict[str, Any]) -> str:
    """Return train.py source suitable for diff display (apply stored overrides if needed)."""
    train_py = edit.get("train_py") or ""
    overrides = edit.get("config_overrides") or {}
    if not overrides:
        return train_py
    baseline = _baseline_train_py()
    if normalize_train_py(train_py) != normalize_train_py(baseline):
        return train_py
    return build_train_py_with_config_overrides(baseline, overrides)


def _train_py_diff(baseline: str, edited: str, *, title: str = "edit") -> str:
    edited = normalize_train_py(edited)
    baseline = normalize_train_py(baseline)
    if edited == baseline:
        return "(no changes from baseline train.py — pass config_overrides when saving edits)"
    lines = difflib.unified_diff(
        baseline.splitlines(),
        edited.splitlines(),
        fromfile="baseline/train.py",
        tofile=f"{title}/train.py",
        lineterm="",
    )
    return "\n".join(lines) or "(no changes from baseline train.py — pass config_overrides when saving edits)"


def _diff_stats(baseline: str, edited: str) -> tuple[int, int]:
    diff_lines = difflib.unified_diff(
        baseline.splitlines(),
        edited.splitlines(),
        lineterm="",
    )
    adds = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    dels = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return adds, dels


def _render_diff_pre(diff_text: str, *, max_lines: int | None = None) -> str:
    if not diff_text or diff_text.startswith("(no changes"):
        return f'<pre class="ar-pre ar-code ar-diff">{_esc(diff_text)}</pre>'

    text = _truncate_text(diff_text, max_lines=max_lines) if max_lines else diff_text
    lines_html: list[str] = []
    for line in text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines_html.append(f'<span class="ar-diff-add">{_esc(line)}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            lines_html.append(f'<span class="ar-diff-del">{_esc(line)}</span>')
        elif line.startswith("@@"):
            lines_html.append(f'<span class="ar-diff-hunk">{_esc(line)}</span>')
        else:
            lines_html.append(_esc(line))
    body = "\n".join(lines_html)
    return f'<pre class="ar-pre ar-code ar-diff">{body}</pre>'


def _truncate_code(train_py: str, *, max_lines: int = 20) -> str:
    return _truncate_text(train_py, max_lines=max_lines)


def _format_edit_snippet(edit: dict[str, Any], *, compact: bool = False) -> str:
    title = _esc(edit.get("title", "experiment"))
    summary = _esc(edit.get("change_summary", ""))
    train_py = _effective_train_py_for_edit(edit)
    baseline = _baseline_train_py()
    diff = _train_py_diff(baseline, train_py, title=str(edit.get("title", "edit")))
    adds, dels = _diff_stats(baseline, train_py)
    max_lines = 16 if compact else 24
    stat = f"+{adds} −{dels}" if adds or dels else "unchanged"
    return (
        f"<div style='margin:6px 0 10px'>"
        f"<strong>{title}</strong> "
        f"{_badge(stat, 'muted')}"
        f"<p style='margin:4px 0 6px;color:{SLATE};font-size:.85rem'>{summary}</p>"
        f"{_render_diff_pre(diff, max_lines=max_lines)}"
        f"</div>"
    )


def _format_edit_tool_result(tool: str, parsed: dict[str, Any]) -> str:
    if tool == "edit_train_code" and parsed.get("saved"):
        header = (
            f"<strong>{_esc(tool)}</strong> → saved "
            f"<code>{_esc(parsed.get('title', ''))}</code>"
        )
        return header + _format_edit_snippet(parsed)

    if tool == "edit_train_code_batch" and parsed.get("count"):
        titles = ", ".join(str(t) for t in parsed.get("titles", [])[:6])
        suffix = "…" if len(parsed.get("titles", [])) > 6 else ""
        header = (
            f"<strong>{_esc(tool)}</strong> → saved {parsed.get('count')} edits: "
            f"{_esc(titles)}{suffix}"
        )
        snippets = "".join(
            _format_edit_snippet(edit, compact=True)
            for edit in parsed.get("edits", [])[:4]
        )
        extra = len(parsed.get("edits", [])) - 4
        if extra > 0:
            snippets += f'<p style="color:{SLATE};font-size:.85rem;margin:4px 0">+ {extra} more in Code edits tab</p>'
        return header + snippets

    return ""


def _format_event_detail(et: str, data: dict[str, Any]) -> str:
    if et == "tool_start":
        tool = str(data.get("tool", ""))
        args = dict(data.get("args", {}))
        if tool in ("edit_train_code", "edit_train_code_batch") and "train_py" in args:
            args = {**args, "train_py": f"({len(str(args['train_py']).splitlines())} lines)"}
        if tool == "edit_train_code_batch" and isinstance(args.get("edits"), list):
            args = {
                **args,
                "edits": [
                    {
                        **e,
                        "train_py": f"({len(str(e.get('train_py', '')).splitlines())} lines)",
                    }
                    if isinstance(e, dict) and e.get("train_py")
                    else e
                    for e in args["edits"]
                ],
            }
        args_json = json.dumps(args, default=str)
        if len(args_json) > 140:
            args_json = args_json[:137] + "…"
        return f'<strong>{_esc(tool)}</strong> <code>{_esc(args_json)}</code>'
    if et == "tool_end":
        tool = str(data.get("tool", ""))
        result = str(data.get("result", ""))
        if tool in ("edit_train_code", "edit_train_code_batch") and result:
            parsed = _parse_tool_result(result)
            if isinstance(parsed, dict):
                rich = _format_edit_tool_result(tool, parsed)
                if rich:
                    return rich
        if len(result) > 160:
            result = result[:157] + "…"
        return f'<strong>{_esc(tool)}</strong> → <span style="color:{SLATE}">{_esc(result)}</span>'
    if et == "tool_error":
        return (
            f'<strong>{_esc(data.get("tool", ""))}</strong> '
            f'<span style="color:#991b1b">{_esc(str(data.get("error", ""))[:200])}</span>'
        )
    if et == "turn_start":
        return f"Turn {data.get('turn')} of {data.get('max_turns')}"
    if et == "message":
        content = str(data.get("content", "")).strip()
        return _esc(content[:240]) if content else ""
    if et == "agent_end":
        return f"{data.get('turns')} turns · {data.get('elapsed_ms', 0):,} ms"
    return _esc(json.dumps(data, default=str)[:160])


def render_activity_log(events: list[dict[str, Any]]) -> str:
    items: list[str] = []
    for ev in events[-200:]:
        et = ev.get("type", "")
        data = ev.get("data", {})
        sym, dot_cls, label = _EVENT_META.get(et, ("•", "muted", et))
        detail = _format_event_detail(et, data)
        if et == "message" and not detail:
            continue
        items.append(
            f'<li><span class="ar-dot ar-dot--{dot_cls}"></span>'
            f'{_badge(label, "muted" if dot_cls == "muted" else ("fail" if dot_cls == "err" else "ok"))} '
            f"{sym} {detail}</li>"
        )

    feed = "".join(items) or f'<li style="color:{SLATE}">Waiting for agent activity…</li>'
    return _wrap(
        "<h2>Agent activity</h2>\n"
        f'<div class="ar-card" style="max-height:560px;overflow:auto">'
        f'<ul class="ar-timeline">{feed}</ul></div>\n'
    )


# ---------------------------------------------------------------------------
# Memory panel
# ---------------------------------------------------------------------------


def _render_leaderboard_memory(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return f'<p style="color:{SLATE};margin:0">No persisted experiments yet.</p>'
    rows = []
    for e in entries[-12:]:
        val = e.get("val_bpb")
        val_s = f"{float(val):.6g}" if val is not None else "—"
        rows.append(
            "<tr>"
            f"<td>{_esc(e.get('title', '—'))}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>{val_s}</td>"
            f"<td>{_esc(e.get('model_name', '—'))}</td>"
            "</tr>"
        )
    return (
        '<table class="ar-table">'
        "<thead><tr><th>Title</th><th>val_bpb</th><th>Model</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_hypotheses(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return f'<p style="color:{SLATE};margin:0">No hypotheses recorded yet.</p>'
    cards = []
    for h in entries[-8:]:
        cards.append(
            f'<div class="ar-card" style="padding:12px 14px;margin-bottom:8px">'
            f'<strong>{_esc(h.get("title", ""))}</strong> '
            f'{_badge(_esc(h.get("expected_effect", "")), "gold")}<br>'
            f'<span style="color:{SLATE};font-size:.875rem">{_esc(h.get("hypothesis", ""))}</span>'
            "</div>"
        )
    return "".join(cards)


def _render_promising(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return f'<p style="color:{SLATE};margin:0">No promising code edits yet.</p>'
    cards = []
    for p in entries[-6:]:
        val = p.get("val_bpb")
        val_s = f"{float(val):.6g}" if val is not None else "—"
        cards.append(
            f'<div class="ar-card" style="padding:12px 14px;margin-bottom:8px">'
            f'<strong>{_esc(p.get("title", ""))}</strong> '
            f'{_badge(f"val_bpb {val_s}", "ok")}<br>'
            f'<span style="color:{SLATE};font-size:.875rem">{_esc(p.get("change_summary", ""))}</span>'
            "</div>"
        )
    return "".join(cards)


def _render_code_edit_card(
    edit: dict[str, Any],
    *,
    highlight: bool = False,
    baseline: str | None = None,
) -> str:
    title = _esc(edit.get("title", "experiment"))
    summary = _esc(edit.get("change_summary", ""))
    train_py = _effective_train_py_for_edit(edit)
    slug = _esc(edit.get("slug", ""))
    path = _esc(edit.get("memory_path", f"memory/code/{slug}.py"))
    baseline_src = baseline or _baseline_train_py()
    diff = _train_py_diff(baseline_src, train_py, title=str(edit.get("title", "edit")))
    adds, dels = _diff_stats(baseline_src, train_py)
    val = edit.get("val_bpb")
    badges = [_badge(f"+{adds} −{dels}", "muted"), _badge(path, "muted")]
    if val is not None:
        badges.append(_badge(f"val_bpb {float(val):.6g}", "ok" if edit.get("kept") else "muted"))
    if edit.get("kept"):
        badges.append(_badge("promising", "ok"))
    card_cls = "ar-code-card--best" if highlight or edit.get("kept") else ""
    return (
        f'<div class="ar-card ar-code-card {card_cls}" style="padding:14px 16px;margin-bottom:12px">'
        f"<div style='margin-bottom:8px'><strong>{title}</strong> "
        f"{''.join(badges)}</div>"
        f'<p style="margin:0 0 10px;color:{SLATE};font-size:.875rem">{summary}</p>'
        f"{_render_diff_pre(diff)}"
        "</div>"
    )


def render_code_edits_panel(
    edits: list[dict[str, Any]],
    *,
    best_title: str | None = None,
) -> str:
    """Render saved ``train.py`` edits as unified diffs vs baseline."""
    if not edits:
        return _wrap(
            "<h2>Code edits</h2>\n"
            f'<p style="color:{SLATE}">No saved train.py edits yet.</p>\n'
        )

    baseline = _baseline_train_py()
    best_key = (best_title or "").strip().lower()
    best_val = float("inf")
    for edit in edits:
        val = edit.get("val_bpb")
        if val is not None and float(val) < best_val:
            best_val = float(val)
            best_key = str(edit.get("title", "")).strip().lower()

    cards = []
    for edit in edits[-12:]:
        title_key = str(edit.get("title", "")).strip().lower()
        cards.append(
            _render_code_edit_card(
                edit,
                highlight=title_key == best_key and best_key != "",
                baseline=baseline,
            )
        )

    return _wrap(
        "<h2>Code edits</h2>\n"
        f'<p style="color:{SLATE}">{len(edits)} saved train.py variant(s). '
        "Unified diff vs baseline <code>train.py</code> — green additions, red removals. "
        "Full edited source is still returned in each edit tool’s task output.</p>\n"
        f"{''.join(cards)}\n"
    )


def render_memory_panel(
    memory_key: str,
    n_messages: int,
    persisted_leaderboard: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    persisted_hypotheses: list[dict[str, Any]] | None = None,
    persisted_promising: list[dict[str, Any]] | None = None,
    code_edits: list[dict[str, Any]] | None = None,
) -> str:
    audit_rows = "".join(
        "<tr>"
        f"<td style='white-space:nowrap;font-size:.8rem'>{_esc(a.get('ts', ''))}</td>"
        f"<td>{_badge(_esc(a.get('op', '')), 'muted')}</td>"
        f"<td><code>{_esc(a.get('path', ''))}</code></td>"
        f"<td style='color:{SLATE}'>{_esc(a.get('reason', ''))}</td>"
        "</tr>"
        for a in audit[-20:]
    )
    audit_table = (
        '<table class="ar-table">'
        "<thead><tr><th>When</th><th>Op</th><th>Path</th><th>Reason</th></tr></thead>"
        f"<tbody>{audit_rows}</tbody></table>"
        if audit
        else f"<p style='color:{SLATE}'>No memory writes recorded.</p>"
    )

    promising_section = ""
    if persisted_promising:
        promising_section = (
            "<h3>Promising train.py edits</h3>\n" + _render_promising(persisted_promising) + "\n"
        )

    code_section = ""
    if code_edits:
        baseline = _baseline_train_py()
        code_section = (
            "<h3>Saved train.py variants</h3>\n"
            + "".join(
                _render_code_edit_card(e, highlight=bool(e.get("kept")), baseline=baseline)
                for e in code_edits[-6:]
            )
            + "\n"
        )

    return _wrap(
        "<h2>Agent memory</h2>\n"
        f'<p style="color:{SLATE}">Keyed <code>MemoryStore</code> at '
        f"<code>{_esc(memory_key)}</code> — survives across runs.</p>\n"
        f"{_stat(str(n_messages), 'transcript messages')}"
        f"{_stat(str(len(persisted_leaderboard)), 'saved experiments')}\n"
        "<h3>Persisted leaderboard</h3>\n"
        f"{_render_leaderboard_memory(persisted_leaderboard)}\n"
        "<h3>Recorded hypotheses</h3>\n"
        f"{_render_hypotheses(persisted_hypotheses or [])}\n"
        f"{promising_section}"
        f"{code_section}"
        "<h3>Audit trail</h3>\n"
        f"{audit_table}\n"
    )


def render_summary(
    directive: str,
    leaderboard: list[LeaderboardEntry],
    best: LeaderboardEntry | None,
    summary_text: str,
    code_edits: list[dict[str, Any]] | None = None,
) -> str:
    best_line = ""
    if best is not None and best.val_bpb is not None:
        best_line = (
            f'<p style="margin:8px 0 0">Best: <strong>{_esc(best.title)}</strong> — '
            f'<span style="color:{GREEN_TEXT};font-weight:700">{best.val_bpb:.6g}</span> val_bpb</p>'
        )

    best_edit_html = ""
    if code_edits and best is not None:
        best_key = best.title.strip().lower()
        baseline = _baseline_train_py()
        for edit in code_edits:
            if str(edit.get("title", "")).strip().lower() == best_key:
                best_edit_html = (
                    "<h2>Best experiment changes</h2>\n"
                    + _render_code_edit_card(edit, highlight=True, baseline=baseline)
                    + "\n"
                )
                break

    return _wrap(
        f'<div class="ar-hero">'
        f"<h1>Autoresearch run</h1>"
        f'<p style="margin:0;color:{SLATE}">{_esc(directive[:320])}{"…" if len(directive) > 320 else ""}</p>'
        f"{best_line}</div>\n"
        f"{_leaderboard_body(leaderboard, best)}"
        f"{best_edit_html}"
        "<h2>Agent summary</h2>\n"
        f'<pre class="ar-pre">{_esc(summary_text or "(none)")}</pre>\n'
    )
