"""The refinery's report, its tabs, and the dataset card."""

from __future__ import annotations

import viz


def _rules(shards) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in shards:
        for k, v in s.rejected.items():
            out[k] = out.get(k, 0) + v
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def funnel(shards, dedup, curated=None, threshold: float | None = None) -> list[tuple[str, int]]:
    rows = [("raw web documents", sum(s.raw for s in shards)), ("pass heuristic filters", sum(s.kept for s in shards))]
    if dedup:
        rows += [("no exact duplicate", dedup.after_exact), ("no near duplicate", dedup.after_near),
                 ("no eval overlap", dedup.after_decontam)]
    if curated:
        rows.append((f"educational score ≥ {threshold:g}", curated.kept))
    return rows


def _funnel_chart(rows: list[tuple[str, int]]) -> str:
    raw = rows[0][1] or 1
    bars = [viz.Bar(name, n, 1 if i < len(rows) - 1 or len(rows) < 6 else 3, f"{n / raw:.1%}") for i, (name, n) in enumerate(rows)]
    return viz.bar_chart(bars, label_width=190, max_value=raw, fmt=lambda v: f"{v:,.0f}")


def _rules_chart(shards) -> str:
    rules = _rules(shards)
    if not rules:
        return "<p class='note'>no rejections</p>"
    return viz.bar_chart([viz.Bar(k, v, 2) for k, v in list(rules.items())[:12]], label_width=260, fmt=lambda v: f"{v:,.0f}")


def _histogram(curated, threshold: float) -> str:
    cats = [f"{i * 0.25:g}" for i in range(20)]
    below = [v if (i + 1) * 0.25 <= threshold else 0 for i, v in enumerate(curated.histogram)]
    above = [v if (i + 1) * 0.25 > threshold else 0 for i, v in enumerate(curated.histogram)]
    return viz.stacked_columns(cats, [viz.Series("dropped", below, "neutral"), viz.Series(f"kept (≥ {threshold:g})", above, 1)],
                               height=230, fmt=viz.compact)


def _confusion(m: dict, ref: str, pred: str) -> str:
    return viz.heatmap(m["confusion"], [f"{ref} {i}" for i in range(6)], [f"{i}" for i in range(6)],
                       width=420, label_width=80, cell_h=30, fmt=lambda v: f"{v:,.0f}") + \
        f"<p class='note'>Rows: {viz.esc(ref)} score. Columns: {viz.esc(pred)} score, rounded. Hover a cell for its count.</p>"


def _agreement_table(clf) -> str:
    rows = [("distilled classifier vs LLM labels", clf.held_out), ("public FineWeb-Edu classifier vs LLM labels", clf.public_vs_llm),
            ("distilled vs public classifier", clf.ours_vs_public)]
    return viz.table(
        ["comparison (held-out documents)", "n", "Spearman", "MAE", "exact", "within 1", "F1 at ≥ 3"],
        [[viz.esc(name), m["n"], f"{m['spearman']:.2f}", f"{m['mae']:.2f}", f"{m['exact']:.0%}", f"{m['within_one']:.0%}",
          f"{m['f1']:.2f}"] for name, m in rows],
    )


def live(config, shards, dedup, labels, clf, curated, threshold, eval_counts) -> str:
    rows = funnel(shards, dedup)
    blocks = [
        viz.tiles([("Raw documents", f"{rows[0][1]:,}", f"{len(shards)} shards"),
                   ("Survivors so far", f"{rows[-1][1]:,}", rows[-1][0]),
                   ("LLM labels", f"{sum(l.parsed for l in labels):,}" if labels else "—"),
                   ("Classifier vs LLM, Spearman", f"{clf.held_out['spearman']:.2f}" if clf else "—")], hero=True),
        viz.row(viz.panel("Documents kept at each stage", _funnel_chart(rows)),
                viz.panel("Documents rejected, by heuristic rule", _rules_chart(shards))),
    ]
    if clf:
        blocks.append(viz.panel("Agreement on held-out documents", _agreement_table(clf)))
    return viz.page("Pretraining data refinery", "Running: the funnel fills in as each stage finishes.", *blocks)


def summarize(config, shards, dedup, labels, clf, scores, curated, threshold, timing, label_model: str) -> dict:
    llm_docs = sum(l.n for l in labels)
    llm_s = sum(l.seconds for l in labels)
    clf_docs = sum(s.n for s in scores)
    clf_s = sum(s.seconds for s in scores)
    llm_per_doc, clf_per_doc = llm_s / max(llm_docs, 1), clf_s / max(clf_docs, 1)
    return {
        "threshold": threshold,
        "label_model": label_model,
        "funnel": funnel(shards, dedup, curated, threshold),
        "rules": _rules(shards),
        "config": vars(config),
        "filters_version": config.version,
        "tokens_kept": curated.tokens_kept,
        "tokens_scored": curated.tokens_scored,
        "contaminated": sum(s.contaminated for s in shards),
        "agreement": {"held_out": clf.held_out, "public_vs_llm": clf.public_vs_llm, "ours_vs_public": clf.ours_vs_public},
        "cost": {
            "llm_docs": llm_docs, "llm_gpu_s": llm_s, "llm_per_doc": llm_per_doc,
            "llm_startup_s": sum(l.startup_s for l in labels),
            "llm_tokens": sum(l.prompt_tokens + l.completion_tokens for l in labels),
            "clf_docs": clf_docs, "clf_gpu_s": clf_s, "clf_per_doc": clf_per_doc,
            "clf_load_s": sum(s.load_s for s in scores),
            "ratio": llm_per_doc / clf_per_doc if clf_per_doc else 0.0,
            "llm_all_gpu_h": llm_per_doc * clf_docs / 3600,
        },
        "timing": timing,
        "label_counts": clf.label_counts,
        "n_train": clf.n_train,
    }


def _cost_panel(c: dict) -> str:
    bars = [viz.Bar("LLM labeler (L40S)", c["llm_per_doc"] * 1000, 2),
            viz.Bar("distilled classifier (L4)", c["clf_per_doc"] * 1000, 1)]
    return viz.panel(
        "GPU time per document",
        viz.bar_chart(bars, unit=" ms", label_width=190, fmt=lambda v: f"{v:,.1f}"),
        f"The classifier is {c['ratio']:,.0f}x cheaper per document, on a smaller GPU. Labeling all {c['clf_docs']:,} "
        f"surviving documents with the LLM would take about {c['llm_all_gpu_h']:.1f} L40S-hours; the classifier took "
        f"{c['clf_gpu_s'] / 60:.1f} L4-minutes. Model loads are excluded: {c['llm_startup_s']:.0f} s for vLLM, "
        f"{c['clf_load_s']:.0f} s for the encoder.",
    )


def final(summary: dict, shards, dedup, clf, curated, eval_counts) -> str:
    rows = summary["funnel"]
    raw = rows[0][1]
    held = summary["agreement"]["held_out"]
    timing = summary["timing"]
    return viz.page(
        "Pretraining data refinery",
        f"{raw:,} FineWeb documents through heuristic filters, global deduplication, decontamination against "
        f"{sum(eval_counts.values()):,} eval questions, and an educational-value classifier distilled from LLM labels.",
        viz.tiles([("Curated documents", f"{curated.kept:,}", f"{curated.kept / raw:.1%} of raw · {curated.tokens_kept / 1e6:,.1f}M tokens"),
                   ("Classifier vs LLM", f"{held['spearman']:.2f}", f"Spearman on {held['n']} held-out labels"),
                   ("Cheaper per document", f"{summary['cost']['ratio']:,.0f}x", "classifier vs LLM, GPU-seconds"),
                   ("Removed as eval overlap", f"{summary['contaminated']:,}", "13-gram match with GSM8K, MMLU or ARC")], hero=True),
        viz.row(viz.panel("Documents kept at each stage", _funnel_chart(rows)),
                viz.panel("Distilled classifier scores, every surviving document", _histogram(curated, summary["threshold"]),
                          f"{curated.scored:,} documents scored; the threshold keeps {curated.kept:,}.")),
        viz.row(viz.panel("Distilled classifier against held-out LLM labels", _confusion(held, "LLM", "classifier")),
                viz.panel("Score by score", viz.scatter([p[0] for p in clf.points], [p[1] for p in clf.points], jitter=0.3,
                                                        x_label="LLM label (jittered)", y_label="distilled classifier",
                                                        x_range=(-0.5, 5.5), y_range=(-0.5, 5.5)))),
        viz.row(_cost_panel(summary["cost"]),
                viz.panel("Wall-clock time of each stage in this run",
                          viz.bar_chart([viz.Bar(k, v, 7) for k, v in timing.items()], unit=" s", label_width=170,
                                        fmt=lambda v: f"{v:,.0f}"),
                          "Measured by the driver. On a rerun, a cached stage takes seconds.")),
        viz.panel("Agreement on held-out documents", _agreement_table(clf),
                  f"The public classifier was distilled from Llama-3-70B-Instruct labels; ours from {viz.esc(summary['label_model'])} labels."),
        viz.row(viz.panel("Documents rejected, by heuristic rule", _rules_chart(shards)),
                viz.panel("Largest near-duplicate clusters", viz.table(
                    ["documents", "first URL"], [[c["size"], f"<code>{viz.esc(c['url'][:44])}</code>"] for c in dedup.clusters]))
                if dedup.clusters else viz.panel("Largest near-duplicate clusters", "<p class='note'>none</p>")),
    )


def _doc(title: str, url: str, text: str, extra: str = "") -> str:
    return (f"<div style='margin:0 0 12px'><p style='margin:0 0 4px'><b>{viz.esc(title)}</b> · "
            f"<code>{viz.esc(url[:90])}</code></p>{extra}<pre style='white-space:pre-wrap'>{viz.esc(text)}</pre></div>")


def rejected_tab(shards, dedup, curated) -> str:
    heur = [e for s in shards for e in s.examples]
    seen, picked = set(), []
    for e in heur:  # one example per rule
        if e["rule"] not in seen:
            seen.add(e["rule"])
            picked.append(e)
    contam = [e for s in shards for e in s.contam_examples][:8]
    blocks = [
        viz.panel("Heuristic filters", "".join(_doc(e["rule"], e["url"], e["text"]) for e in picked[:10])),
        viz.panel("Duplicates", "".join(
            _doc(f"{e['stage']} (removed)", e["url"], e["text"]) + _doc("kept copy", e["kept_url"], e["kept_text"])
            + "<hr style='border:0;border-top:1px solid var(--grid)'/>" for e in dedup.examples) or "<p class='note'>none</p>"),
        viz.panel("Eval overlap", "".join(
            _doc(f"{e['benchmark']} · {e['matched_ngrams']} matching 13-gram(s)", e["url"], e["context"],
                 f"<p class='note' style='margin:0 0 4px'>Matched: <code>{viz.esc(e['match'])}</code><br/>"
                 f"Eval question: {viz.esc(e['question'])}</p>") for e in contam) or "<p class='note'>none</p>"),
        viz.panel("Below the educational threshold, and the highest scored",
                  "".join(_doc(f"{e['stage']} · {e['rule']}", e["url"], e["text"]) for e in curated.examples)),
    ]
    return viz.page("Rejected documents", "Examples of what each stage removed, and why.", *blocks)


def labels_tab(clf, labels) -> str:
    counts = clf.label_counts
    startup = sum(l.startup_s for l in labels)
    rows = [[i, f"{l.n:,}", f"{l.parsed:,}", f"{l.seconds:,.0f} s", f"{l.startup_s:,.0f} s" if l.startup_s else "warm",
             f"{(l.prompt_tokens + l.completion_tokens) / max(l.seconds, 1e-9):,.0f}"] for i, l in enumerate(labels)]
    ex = "".join(
        _doc(f"LLM {e['llm']} · distilled {e['ours']:.2f} · public {e['public']:.2f}", "", e["text"],
             f"<p class='note' style='margin:0 0 4px'>{viz.esc(e['justification'][-600:])}</p>") for e in clf.examples)
    return viz.page(
        "LLM labels and the classifier",
        f"{sum(counts):,} documents labeled by the LLM; {clf.n_train:,} trained the classifier and the rest are held out.",
        viz.row(viz.panel("LLM label distribution", viz.bar_chart([viz.Bar(f"score {i}", c, 1) for i, c in enumerate(counts)],
                                                                  label_width=80, fmt=lambda v: f"{v:,.0f}")),
                viz.panel("Labeling calls on the warm GPU worker",
                          viz.table(["call", "documents", "scored", "labeling", "model load", "tokens/s"], rows),
                          f"Only the first call paid the {startup:.0f} s vLLM start; the rest found the server warm.")),
        viz.row(viz.panel("Public FineWeb-Edu classifier against the same LLM labels", _confusion(clf.public_vs_llm, "LLM", "public")),
                viz.panel("Distilled vs public classifier", viz.scatter([p[2] for p in clf.points], [p[1] for p in clf.points],
                                                                        x_label="public FineWeb-Edu classifier", y_label="distilled classifier",
                                                                        x_range=(-0.5, 5.5), y_range=(-0.5, 5.5), color=3))),
        viz.panel("Head training: class-balanced validation MSE by epoch",
                  viz.line_chart([viz.Series("validation loss", clf.curve, 1, xs=list(range(1, len(clf.curve) + 1)))], x_label="epoch")),
        viz.panel("One held-out document per LLM score, with the LLM's justification", ex),
    )


def card(summary: dict) -> str:
    rows = summary["funnel"]
    held = summary["agreement"]["held_out"]
    c = summary["cost"]
    cfg = summary["config"]
    thresholds = [[viz.esc(k), viz.esc(v)] for k, v in cfg.items() if k != "version"]
    return viz.card(
        "edu-web-corpus",
        f"FineWeb documents filtered, deduplicated, decontaminated, and kept at an educational score of at least "
        f"{summary['threshold']:g}. Each row has the classifier's <code>edu_score</code>.",
        ["pretraining data", "FineWeb sample-10BT", "FineWeb-Edu rubric", summary["filters_version"]],
        viz.tiles([("Documents", f"{rows[-1][1]:,}", f"{rows[-1][1] / rows[0][1]:.1%} of {rows[0][1]:,} raw"),
                   ("Tokens", f"{summary['tokens_kept'] / 1e6:,.1f}M", f"of {summary['tokens_scored'] / 1e6:,.1f}M scored"),
                   ("Classifier vs LLM", f"{held['spearman']:.2f}", "Spearman, held out")], hero=True),
        viz.panel("Funnel", viz.table(["stage", "documents", "share of raw"],
                                      [[viz.esc(n), f"{v:,}", f"{v / rows[0][1]:.1%}"] for n, v in rows])),
        viz.panel("Heuristic rejections", viz.table(["rule", "documents"], [[viz.esc(k), f"{v:,}"] for k, v in summary["rules"].items()])),
        viz.panel("Labeler and classifier", viz.table(["", ""], [
            ["LLM labeler", f"{viz.esc(summary['label_model'])}, FineWeb-Edu prompt, greedy"],
            ["Labels", f"{sum(summary['label_counts']):,} ({summary['n_train']:,} train)"],
            ["Classifier", "Snowflake/snowflake-arctic-embed-m (frozen) + regression head"],
            ["Agreement with LLM", f"Spearman {held['spearman']:.2f}, F1 at ≥ 3 {held['f1']:.2f}"],
            ["GPU time per document", f"LLM {c['llm_per_doc'] * 1000:,.1f} ms · classifier {c['clf_per_doc'] * 1000:,.2f} ms"],
        ])),
        viz.panel("Filter thresholds", viz.table(["setting", "value"], thresholds)),
        viz.panel("Load it", viz.code(
            'import flyte\nfrom flyte.remote import Artifact\n\n'
            'flyte.init_from_config()\nartifact = Artifact.get("edu-web-corpus")')),
    )
