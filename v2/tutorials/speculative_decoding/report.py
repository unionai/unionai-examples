"""Turn benchmark results into speedup, acceptance, and losslessness numbers, and HTML."""

from html import escape

import report_ui as ui
from workloads import gsm8k_correct


# {{docs-fragment summarize}}
def summarize(results: list[dict]) -> dict:
    """Compare every configuration against the no-speculation baseline."""
    by_name = {r["config"]: r for r in results}
    base = by_name["baseline"]
    summary: dict = {"configs": [r["config"] for r in results], "workloads": {}}

    for workload, base_runs in base["workloads"].items():
        rows = {}
        base_c1 = base_runs["1"]
        concurrencies = sorted((c for c in base_runs if c.isdigit()), key=int)
        for name, result in by_name.items():
            runs = result["workloads"][workload]
            row = {
                "acceptance_length": runs["1"]["acceptance_length"],
                "speedup": {
                    c: runs[c]["per_request_tok_s"] / base_runs[c]["per_request_tok_s"]
                    for c in concurrencies
                },
                "throughput_tok_s": {c: runs[c]["throughput_tok_s"] for c in concurrencies},
                "tpot_p50_ms": runs["1"]["tpot_p50_ms"],
                "ttft_p50_s": runs["1"]["ttft_p50_s"],
                # Losslessness: greedy outputs should match the baseline's. Over a long
                # output, one floating-point difference flips a token and the texts diverge,
                # so also measure how much of the output matches before that happens.
                "exact_match": _match_rate(runs["1"]["outputs"], base_c1["outputs"]),
                "prefix_agreement": _prefix_agreement(runs["1"]["outputs"], base_c1["outputs"]),
            }
            if workload == "math":
                answers = result["math_answers"]
                row["gsm8k_accuracy"] = sum(
                    gsm8k_correct(o, a) for o, a in zip(runs["1"]["outputs"], answers)
                ) / len(answers)
            rows[name] = row
        # The noise floor: the baseline against itself, same prompts, at a different batch
        # size. Batched kernels are not bit-identical across batch sizes, speculation or not.
        summary["workloads"][workload] = {
            "rows": rows,
            "noise_floor_exact_match": _match_rate(base_runs["noise_floor_outputs"], base_c1["outputs"]),
            "noise_floor_prefix_agreement": _prefix_agreement(base_runs["noise_floor_outputs"], base_c1["outputs"]),
            "concurrencies": concurrencies,
        }
    return summary
# {{/docs-fragment summarize}}


def _match_rate(a: list[str], b: list[str]) -> float:
    return sum(x == y for x, y in zip(a, b)) / max(len(b), 1)


def _prefix_agreement(a: list[str], b: list[str]) -> float:
    """Mean fraction of each reference output that matches before the first divergence."""
    fractions = []
    for x, y in zip(a, b):
        n = 0
        for cx, cy in zip(x, y):
            if cx != cy:
                break
            n += 1
        fractions.append(n / max(len(y), 1))
    return sum(fractions) / max(len(fractions), 1)


# {{docs-fragment render}}
def render_html(summary: dict, title: str = "Speculative decoding benchmark") -> str:
    """The benchmark report: headline numbers, then one section per workload."""
    workloads = summary["workloads"]
    best = {}
    for name, data in workloads.items():
        rows = {n: r for n, r in data["rows"].items() if n != "baseline"}
        if rows:
            winner = max(rows, key=lambda n: rows[n]["speedup"]["1"])
            best[name] = (winner, rows[winner]["speedup"]["1"])

    headline = {
        f"{w}: best at concurrency 1": f"{s:.2f}x ({n})" for w, (n, s) in best.items()
    }
    math = workloads.get("math", {}).get("rows", {})
    if math:
        accs = [r["gsm8k_accuracy"] for r in math.values()]
        headline["GSM8K accuracy range"] = f"{min(accs):.1%} – {max(accs):.1%}"

    parts = [
        ui.hero(
            title,
            "Every configuration serves Qwen3-8B on an identical GPU and replays the same held-out prompts.",
            summary["configs"],
        ),
        ui.tiles(headline),
        ui.section(
            "How to read this",
            lede=(
                "Speedup is per-request decode speed relative to no speculation. Acceptance length "
                "is the mean number of tokens each target forward pass produces. Exact match and "
                "matches-before-divergence compare greedy outputs with the baseline's; the noise "
                "floor is the baseline compared with itself at a different batch size."
            ),
        ),
    ]
    for workload, data in workloads.items():
        rows, cs = data["rows"], data["concurrencies"]
        top = cs[-1]
        spec = {n: r for n, r in rows.items() if n != "baseline"}
        headers = ["config", "acceptance length"] + [f"speedup c={c}" for c in cs]
        headers += ["TPOT p50 (ms)", "TTFT p50 (s)", "exact match", "matches before divergence"]
        if workload == "math":
            headers.append("GSM8K accuracy")
        table_rows = []
        for name, r in rows.items():
            row = [name, f"{r['acceptance_length']:.2f}"]
            row += [f"{r['speedup'][c]:.2f}x" for c in cs]
            row += [f"{r['tpot_p50_ms']:.1f}", f"{r['ttft_p50_s']:.3f}", f"{r['exact_match']:.0%}", f"{r['prefix_agreement']:.0%}"]
            if workload == "math":
                row.append(f"{r['gsm8k_accuracy']:.1%}")
            table_rows.append(row)
        parts.append(
            ui.section(
                workload,
                ui.charts(
                    ui.bar_chart("Speedup at concurrency 1", {n: r["speedup"]["1"] for n, r in rows.items()}, "{:.2f}x", 1.0),
                    ui.bar_chart(f"Speedup at concurrency {top}", {n: r["speedup"][top] for n, r in rows.items()}, "{:.2f}x", 1.0),
                    ui.bar_chart("Acceptance length", {n: r["acceptance_length"] for n, r in spec.items()}),
                    ui.line_chart(
                        "Speedup across concurrency",
                        {n: [(int(c), r["speedup"][c]) for c in cs] for n, r in spec.items()},
                        y_label="concurrent requests",
                    ),
                ),
                ui.table(headers, table_rows),
                f"<p class='lede'>Noise floor: replayed at concurrency {top}, the baseline matches its own "
                f"concurrency-1 outputs exactly on {data['noise_floor_exact_match']:.0%} of prompts, and for "
                f"{data['noise_floor_prefix_agreement']:.0%} of their length before diverging. A lossless "
                f"configuration should look like that, not like 100%.</p>",
            )
        )
    return ui.page(*parts)
# {{/docs-fragment render}}


def server_report(report: dict, speculative_config: dict | None) -> str:
    """One configuration's results: speed across concurrency and acceptance by position."""
    workloads = report["workloads"]
    tiles = {}
    for name, runs in workloads.items():
        tiles[f"{name}: tok/s per request (c=1)"] = f"{runs['1']['per_request_tok_s']:.1f}"
        if speculative_config:
            tiles[f"{name}: acceptance length"] = f"{runs['1']['acceptance_length']:.2f}"
    charts = [
        ui.line_chart(
            "Decode speed per request",
            {n: [(int(c), r["per_request_tok_s"]) for c, r in runs.items() if c.isdigit()] for n, runs in workloads.items()},
            y_label="concurrent requests",
        ),
        ui.line_chart(
            "Aggregate throughput (tok/s)",
            {n: [(int(c), r["throughput_tok_s"]) for c, r in runs.items() if c.isdigit()] for n, runs in workloads.items()},
            y_label="concurrent requests",
        ),
    ]
    if speculative_config:
        for name, runs in workloads.items():
            by_pos = runs["1"].get("acceptance_by_position") or {}
            if by_pos:
                charts.append(
                    ui.bar_chart(f"{name}: acceptance by draft position", {f"token {int(k) + 1}": v for k, v in by_pos.items()}, "{:.0%}")
                )
    samples = [
        f"<div class='chat'><div class='msg assistant'><div class='role'>{escape(n)}</div>{escape(runs['1']['outputs'][0][:800])}</div></div>"
        for n, runs in workloads.items() if runs["1"]["outputs"]
    ]
    return ui.page(
        ui.hero(
            f"Benchmark: {report['config']}",
            "Qwen3-8B on one GPU, replaying held-out chat, agent, and math prompts with greedy decoding.",
            [speculative_config.get("method", "") if speculative_config else "no speculation"],
        ),
        ui.tiles(tiles),
        ui.charts(*charts),
        ui.section("Server configuration", ui.pretty_json(speculative_config or {"speculative_config": None})),
        ui.section("First greedy output per workload", *samples),
    )


def gate_report(
    window: str,
    n_traces: int,
    n_train: int,
    n_eval_prompts: int,
    acceptance: dict[str, float] | None,
    promoted: bool | None,
    reason: str,
) -> str:
    """The nightly retraining decision: what it trained on and whether the candidate won."""
    tiles = {
        "traces in window": f"{n_traces:,}",
        "training traces": f"{n_train:,}",
        "held-out prompts": f"{n_eval_prompts:,}",
    }
    charts = []
    if acceptance:
        gain = acceptance["candidate"] / acceptance["current"] - 1
        tiles["current acceptance length"] = f"{acceptance['current']:.3f}"
        tiles["candidate acceptance length"] = f"{acceptance['candidate']:.3f}"
        tiles["change"] = f"{gain:+.1%}"
        charts.append(ui.bar_chart("Acceptance length on held-out recent traffic", acceptance, "{:.3f}"))
    tone = "good" if promoted else "neutral"
    verdict = "Promoted the candidate" if promoted else "Kept the current draft head"
    return ui.page(
        ui.hero("Nightly draft head retraining", window, ["traces → training data", "acceptance-length gate"]),
        f"<div class='decision {tone}'>{escape(verdict)}: {escape(reason)}</div>",
        ui.tiles(tiles, tone={"change": "good"} if promoted else {}),
        ui.charts(*charts) if charts else "",
    )
