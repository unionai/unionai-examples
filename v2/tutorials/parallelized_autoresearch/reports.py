"""Agent run helpers for building directives and parsing experiment results."""

from __future__ import annotations

import json
import re
from typing import Any

from autoresearch_types import LeaderboardEntry


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
