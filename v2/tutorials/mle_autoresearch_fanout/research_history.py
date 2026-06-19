"""Cross-session research context for code-edit autoresearch agents."""

from __future__ import annotations

from typing import Any

from flyte.ai.agents import MemoryStore


def _title_key(title: str) -> str:
    return str(title or "").strip().lower()


def _best_from_entries(entries: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    best_val: float | None = None
    best_title: str | None = None
    for row in entries:
        val = row.get("val_bpb")
        if val is None:
            continue
        fval = float(val)
        if best_val is None or fval < best_val:
            best_val = fval
            best_title = str(row.get("title", ""))
    return best_val, best_title


def _latest_by_title(rows: list[dict[str, Any]], *, title_field: str = "title") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _title_key(str(row.get(title_field, "")))
        if key:
            out[key] = row
    return out


def _outcome_label(
    *,
    val_bpb: float | None,
    success: bool | None,
    best_val: float | None,
) -> str:
    if success is False or (val_bpb is None and success is not True):
        return "failed"
    if val_bpb is None:
        return "not_run"
    if best_val is None:
        return "ran"
    delta = float(val_bpb) - best_val
    if delta <= 0:
        return "new_best"
    return "regression"


def _vs_best_text(val_bpb: float | None, best_val: float | None) -> str:
    if val_bpb is None or best_val is None:
        return "—"
    delta = float(val_bpb) - best_val
    if abs(delta) < 1e-12:
        return "0 (ties best)"
    sign = "+" if delta > 0 else ""
    quality = "worse" if delta > 0 else "better"
    return f"{sign}{delta:.6g} ({quality})"


async def load_research_history(memory_key: str) -> dict[str, Any]:
    """Merge saved edits, run results, and outcomes for cross-session agent context."""
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    code_index: list[dict[str, Any]] = await memory.read_json.aio("memory/code_index.json", default=[])
    leaderboard: list[dict[str, Any]] = await memory.read_json.aio("memory/leaderboard.json", default=[])
    promising: list[dict[str, Any]] = await memory.read_json.aio("memory/promising_code.json", default=[])
    hypotheses: list[dict[str, Any]] = await memory.read_json.aio("memory/hypotheses.json", default=[])

    lb_by_title = _latest_by_title(leaderboard)
    prom_by_title = _latest_by_title(promising)
    hyp_by_title = _latest_by_title(hypotheses)

    best_val, best_title = _best_from_entries(leaderboard)
    if best_val is None:
        best_val, best_title = _best_from_entries(promising)

    trials: list[dict[str, Any]] = []
    seen: set[str] = set()

    for edit in code_index:
        title = str(edit.get("title", ""))
        key = _title_key(title)
        if not key:
            continue
        seen.add(key)
        lb = lb_by_title.get(key, {})
        prom = prom_by_title.get(key, {})
        hyp = hyp_by_title.get(key, {})

        val = lb.get("val_bpb")
        if val is None:
            val = prom.get("val_bpb")
        val_f = float(val) if val is not None else None

        success = lb.get("success")
        if success is None and val_f is not None:
            success = True
        if success is None and lb.get("error"):
            success = False

        beat_best = val_f is not None and best_val is not None and val_f <= best_val
        trials.append(
            {
                "title": title,
                "change_summary": edit.get("change_summary") or prom.get("change_summary") or "",
                "edited_at": edit.get("edited_at"),
                "val_bpb": val_f,
                "model_name": lb.get("model_name"),
                "success": success,
                "error": lb.get("error"),
                "hypothesis": hyp.get("hypothesis"),
                "expected_effect": hyp.get("expected_effect"),
                "beat_best": beat_best,
                "delta_vs_best": (val_f - best_val) if val_f is not None and best_val is not None else None,
                "vs_best": _vs_best_text(val_f, best_val),
                "outcome": _outcome_label(val_bpb=val_f, success=success, best_val=best_val),
                "kept": bool(prom.get("kept")),
            }
        )

    for key, lb in lb_by_title.items():
        if key in seen:
            continue
        val = lb.get("val_bpb")
        val_f = float(val) if val is not None else None
        success = lb.get("success")
        if success is None and val_f is not None:
            success = True
        trials.append(
            {
                "title": lb.get("title", key),
                "change_summary": "",
                "edited_at": None,
                "val_bpb": val_f,
                "model_name": lb.get("model_name"),
                "success": success,
                "error": lb.get("error"),
                "hypothesis": hyp_by_title.get(key, {}).get("hypothesis"),
                "expected_effect": hyp_by_title.get(key, {}).get("expected_effect"),
                "beat_best": val_f is not None and best_val is not None and val_f <= best_val,
                "delta_vs_best": (val_f - best_val) if val_f is not None and best_val is not None else None,
                "vs_best": _vs_best_text(val_f, best_val),
                "outcome": _outcome_label(val_bpb=val_f, success=success, best_val=best_val),
                "kept": bool(prom_by_title.get(key, {}).get("kept")),
            }
        )

    trials.sort(key=lambda t: (t.get("edited_at") or "", t.get("title", "")))

    return {
        "memory_key": memory_key,
        "best_val_bpb": best_val,
        "best_title": best_title,
        "trials": trials,
        "count_edits": len(code_index),
        "count_runs": sum(1 for t in trials if t.get("val_bpb") is not None or t.get("success") is False),
        "count_regressions": sum(1 for t in trials if t.get("outcome") == "regression"),
        "count_new_best": sum(1 for t in trials if t.get("outcome") == "new_best"),
    }


def format_research_history_for_directive(history: dict[str, Any], *, max_rows: int = 20) -> str:
    """Render prior edits/results as a compact block for the run directive."""
    trials: list[dict[str, Any]] = history.get("trials") or []
    if not trials:
        return ""

    best_val = history.get("best_val_bpb")
    best_title = history.get("best_title")
    header = "\n\n## Prior research (from memory — continue, do not repeat)\n"
    if best_val is not None:
        header += f"Current best: **val_bpb={best_val:.6g}** ({best_title}). Lower is better.\n"
    else:
        header += "No successful runs recorded yet.\n"

    header += (
        "Call ``get_code_edit_history()`` at the start to refresh this table. "
        "Use ``read_train_code`` on the best title to fork winners.\n\n"
    )

    lines = [
        "| Title | Change | val_bpb | vs best | Outcome |",
        "| --- | --- | --- | --- | --- |",
    ]
    for trial in trials[-max_rows:]:
        title = str(trial.get("title", ""))
        change = str(trial.get("change_summary", ""))[:72]
        val = trial.get("val_bpb")
        val_s = f"{float(val):.6g}" if val is not None else ("failed" if trial.get("success") is False else "—")
        lines.append(
            f"| {title} | {change} | {val_s} | {trial.get('vs_best', '—')} | {trial.get('outcome', '—')} |"
        )

    omitted = len(trials) - max_rows
    footer = ""
    if omitted > 0:
        footer = f"\n({omitted} older trial(s) omitted — use get_code_edit_history for the full list.)\n"

    return header + "\n".join(lines) + footer
