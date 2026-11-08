"""Reports: the per-checkpoint scorecard, the training report, the leaderboard and the release card."""

import time

import viz
from config import BASE_MODEL, BENCHMARKS, GUARD_BENCHMARK, GUARD_MAX_DROP


def _benchmarks(rows: dict[int, dict]) -> list[str]:
    seen = {b for r in rows.values() for b in r["scores"]}
    return [b for b in BENCHMARKS if b in seen]


def _acc(row: dict, b: str) -> float | None:
    s = row["scores"].get(b)
    return s["accuracy"] if s else None


# {{docs-fragment ship_rule}}
def pick_best(rows: dict[int, dict]) -> dict | None:
    """The ship rule: highest GSM8K among checkpoints that keep MMLU within GUARD_MAX_DROP of step 0."""
    if 0 not in rows:
        return None
    floor = (_acc(rows[0], GUARD_BENCHMARK) or 0.0) - GUARD_MAX_DROP
    ok = [r for s, r in rows.items() if s > 0 and (_acc(r, GUARD_BENCHMARK) or 0.0) >= floor]
    return max(ok, key=lambda r: (_acc(r, "gsm8k") or 0.0, -r["step"])) if ok else None
# {{/docs-fragment ship_rule}}


def _example(e: dict) -> str:
    verdict = viz.status("good", f"predicted {e['pred']}") if e["ok"] else viz.status("critical", f"predicted {e['pred']}")
    return (f"<p class='note'><b>Gold {viz.esc(e['gold'])}</b> · {verdict}</p>"
            f"<pre style='white-space:pre-wrap;max-height:220px;overflow:auto'>{viz.esc(e['output'])}</pre>")


def scorecard_html(result: dict) -> str:
    """One eval run: the suite's scores for one checkpoint, and a few GSM8K generations."""
    scores = result["scores"]
    tiles = [(BENCHMARKS[b]["label"], f"{s['accuracy']:.1f}%",
              f"±{s['stderr']:.1f} · n={s['n']} · v{s['version']} · {s['seconds']:.0f} s") for b, s in scores.items()]
    blocks = [viz.tiles(tiles, hero=True)]
    gsm = scores.get("gsm8k")
    if gsm and gsm["examples"]:
        blocks.append(viz.panel("GSM8K generations", "".join(
            f"<p><b>Q:</b> {viz.esc(e['question'])}</p>{_example(e)}" for e in gsm["examples"][:3])))
    return viz.page(f"Checkpoint step {result['step']}",
                    f"Training run {viz.esc(result['train_run'])} · profile {result['profile']} · "
                    f"suite took {result['finished_at'] - result['started_at']:.0f} s end to end", *blocks)


def training_html(history: list[dict], published: list[dict], total: int, n_examples: int, model: str) -> str:
    last = history[-1] if history else {"step": 0, "loss": None, "elapsed_s": 0}
    xs = [h["step"] for h in history]
    loss = viz.line_chart([viz.Series("train loss", [h["loss"] for h in history], 1, xs=xs)], height=200,
                          markers={p["step"]: f"ckpt {p['step']}" for p in published if p["step"] > 0}, x_label="step")
    rows = [(p["step"], f"{p['saved_s'] / 60:.1f} min", f"{p['loss']:.3f}" if p["loss"] is not None else "base model")
            for p in published]
    return viz.page(
        f"Fine-tuning {model} on MetaMathQA",
        f"{n_examples:,} GSM8K-style problems · {total} optimizer steps · checkpoints publish as artifact versions "
        "and the evaluator picks them up on its own",
        viz.tiles([("Step", f"{last['step']}/{total}"),
                   ("Train loss", f"{last['loss']:.3f}" if last["loss"] is not None else "–"),
                   ("Checkpoints published", str(len(published))),
                   ("Elapsed", f"{last['elapsed_s'] / 60:.1f} min")]),
        viz.panel("Training loss", loss, "Vertical lines mark saved checkpoints."),
        viz.panel("Published checkpoints", viz.table(["step", "saved at", "train loss"], rows)),
    )


# {{docs-fragment leaderboard}}
def leaderboard_html(train_run: str, rows: dict[int, dict], times: dict[int, float], started: float,
                     info: dict | None, expected: int, final: bool, release: str | None = None) -> str:
    """Learning curves per benchmark, the checkpoints x benchmarks table, a timeline, and generations."""
    benches = _benchmarks(rows)
    steps = sorted(rows)
    best = pick_best(rows) if final else None
    base = rows.get(0)
    state = "final" if final else ("training finished, waiting for evals" if info else "training")
    blocks = []

    if best and base:
        g0, g1 = _acc(base, "gsm8k"), _acc(best, "gsm8k")
        m0, m1 = _acc(base, GUARD_BENCHMARK), _acc(best, GUARD_BENCHMARK)
        blocks.append(viz.tiles([
            ("Release candidate: step", str(best["step"]), f"published as {release}" if release else "ship rule winner"),
            ("GSM8K", f"{g1:.1f}%", f"{g1 - g0:+.1f} pts vs base ({g0:.1f}%)"),
            ("MMLU guard", f"{m1:.1f}%", f"{m1 - m0:+.1f} pts vs base, limit -{GUARD_MAX_DROP:.0f}"),
            ("Checkpoints evaluated", f"{len(rows)}/{expected}", f"{sum(len(r['scores']) for r in rows.values())} benchmark runs"),
        ], hero=True))
    elif final and base:
        top = max((r for s, r in rows.items() if s > 0), key=lambda r: _acc(r, "gsm8k") or 0.0, default=None)
        blocks.append(viz.tiles([
            ("Release candidate", "none", f"no checkpoint keeps MMLU within {GUARD_MAX_DROP:.0f} pts of the base model"),
            ("Best GSM8K", f"{_acc(top, 'gsm8k'):.1f}%" if top else "–", f"step {top['step']}" if top else ""),
            ("Checkpoints evaluated", f"{len(rows)}/{expected}"),
        ], hero=True))
    else:
        blocks.append(viz.tiles([("Status", state), ("Checkpoints evaluated", f"{len(rows)}/{expected}"),
                                 ("Elapsed", f"{(time.time() - started) / 60:.0f} min")]))

    # One small chart per benchmark: the curve, the base model as a reference line, the winner marked.
    charts = []
    for i, b in enumerate(benches):
        pts = [(s, _acc(rows[s], b)) for s in steps if _acc(rows[s], b) is not None]
        refs = [(_acc(base, b), "base")] if base and _acc(base, b) is not None else []
        marks = {best["step"]: "release"} if best else None
        chart = viz.line_chart([viz.Series(BENCHMARKS[b]["label"], [v for _, v in pts], i + 1, xs=[s for s, _ in pts])],
                               height=200, width=360, refs=refs, markers=marks, x_label="step",
                               y_fmt=lambda v: f"{v:.0f}%")
        note = {"target": "What the SFT is for.", "guard": "Regression guard: the ship rule rejects drops over "
                f"{GUARD_MAX_DROP:.0f} pts.", "reasoning": "General reasoning, multiple choice."}[BENCHMARKS[b]["role"]]
        charts.append(viz.panel(BENCHMARKS[b]["label"], chart, note))
    if charts:
        blocks.append("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;"
                      "margin:0 0 14px'>" + "".join(c.replace("class='panel'", "class='panel' style='margin:0'", 1)
                                                    for c in charts) + "</div>")

    # The checkpoints x benchmarks table.
    floor = (_acc(base, GUARD_BENCHMARK) or 0) - GUARD_MAX_DROP if base else None
    table_rows = []
    for s in steps:
        r = rows[s]
        cells = [f"{_acc(r, b):.1f} <span class='note'>±{r['scores'][b]['stderr']:.1f}</span>"
                 if b in r["scores"] else "–" for b in benches]
        if s == 0:
            verdict = viz.status("neutral", "base model")
        elif best and s == best["step"]:
            verdict = viz.status("good", "release candidate")
        elif floor is not None and (_acc(r, GUARD_BENCHMARK) or 0) < floor:
            verdict = viz.status("critical", "fails guard")
        else:
            verdict = viz.status("neutral", "passes guard")
        table_rows.append([f"<b>{s}</b>", *cells, verdict, f"<code>{viz.esc(r['eval_run'])}</code>"])
    blocks.append(viz.panel("Checkpoints × benchmarks (accuracy %, ± one standard error)",
                            viz.table(["step", *[BENCHMARKS[b]["label"] for b in benches], "ship rule", "eval run"],
                                      table_rows)))

    # Timeline: evals overlap training because the trigger, not the trainer, launches them.
    spans = []
    if times:
        # Training runs from the step-0 checkpoint to the last one (or to now, while it runs).
        t_end = max(times.values()) if info or final else time.time()
        spans.append(viz.Span("training", "step 0 to last checkpoint", times.get(0, min(times.values())), t_end, 1))
    for s in steps:
        spans.append(viz.Span("evals", f"step {s}", rows[s]["started_at"], rows[s]["finished_at"], 3))
    blocks.append(viz.panel("Timeline", viz.gantt(spans, unit="s", packed=False),
                            "Each green bar is the newest eval run for one checkpoint, from start to finish."))

    # Generations: the same GSM8K problems, base model vs the release candidate. Problems the
    # fine-tuning fixed come first.
    if base and best and "gsm8k" in base["scores"] and "gsm8k" in best["scores"]:
        pairs = list(zip(base["scores"]["gsm8k"]["examples"], best["scores"]["gsm8k"]["examples"]))
        pairs.sort(key=lambda p: (p[0]["ok"] or not p[1]["ok"], p[0]["ok"] == p[1]["ok"]))
        cells = "".join(
            f"<div class='panel'><p><b>Q:</b> {viz.esc(a['question'])}</p>"
            f"<div class='row'><div><h3>Base model</h3>{_example(a)}</div>"
            f"<div><h3>Step {best['step']}</h3>{_example(c)}</div></div></div>"
            for a, c in pairs[:3])
        blocks.append(viz.panel("GSM8K generations, base vs release candidate", cells))

    return viz.page(f"Evaluate every checkpoint · {train_run}",
                    f"{state} · scores arrive from independent, trigger-launched eval runs", *blocks)
# {{/docs-fragment leaderboard}}


def release_card(train_run: str, rows: dict[int, dict], best: int) -> str:
    benches = _benchmarks(rows)
    base, win = rows[0], rows[best]
    table = viz.table(["benchmark", "base", f"step {best}", "change"],
                      [(BENCHMARKS[b]["label"], f"{_acc(base, b):.1f}", f"{_acc(win, b):.1f}",
                        f"{_acc(win, b) - _acc(base, b):+.1f}") for b in benches
                       if _acc(base, b) is not None and _acc(win, b) is not None])
    return viz.card(f"Math SFT release candidate · step {best}",
                    f"Picked by the ship rule from {len(rows)} evaluated checkpoints of {viz.esc(train_run)}: highest "
                    f"GSM8K with MMLU no more than {GUARD_MAX_DROP:.0f} points below the base model.",
                    [BASE_MODEL.split("/")[-1], "MetaMathQA SFT", f"step {best}"], viz.panel("Scores", table))
