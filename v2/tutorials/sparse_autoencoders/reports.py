"""Reports for the SAE run: training curves, the metrics overview, the feature browser, and the card."""

import html

import viz

SCORE_GOOD, SCORE_OK = 0.85, 0.7


def _pct(v: float, digits: int = 0) -> str:
    return f"{100 * v:.{digits}f}%"


def _score(s: float) -> str:
    s = round(s, 2)
    kind = "good" if s >= SCORE_GOOD else "warning" if s >= SCORE_OK else "critical"
    return viz.status(kind, f"{s:.2f}")


def page_start(layers: list[int], widths: list[int], ks: list[int], n_tokens: int) -> str:
    return viz.page(
        "Sparse autoencoders on Pythia-70m",
        f"Harvesting {viz.compact(n_tokens)} tokens of residual stream at layers {', '.join(map(str, layers))}, "
        f"then training {len(layers) * len(widths)} GPU tasks of TopK SAEs (widths {', '.join(map(str, widths))}, "
        f"k = {', '.join(map(str, ks))}). Each trainer's live curves are in its own report.",
    )


# {{docs-fragment training}}
def training(layer: int, width: int, ks: list[int], steps: int, curve: list[dict], elapsed: float) -> str:
    """The live report of one trainer: normalized MSE and dead fraction for every k."""
    xs = [p["step"] for p in curve]
    last = curve[-1]
    return viz.page(
        f"Training TopK SAEs: layer {layer}, {width:,} features",
        f"One SAE per k on the same batches. Step {last['step']:,} of {steps:,}, {elapsed / 60:.1f} min.",
        viz.tiles([(f"Normalized MSE, k={k}", f"{last[f'k{k}']:.3f}", f"dead {_pct(last[f'dead_k{k}'], 1)}") for k in ks]),
        viz.row(
            viz.panel("Normalized MSE (lower is better)", viz.line_chart(
                [viz.Series(f"k={k}", [p[f"k{k}"] for p in curve], i + 1, xs=xs) for i, k in enumerate(ks)],
                width=420, height=200, x_label="step", y_fmt=lambda v: f"{v:.2f}")),
            viz.panel("Dead features (no firing in the last 1M tokens)", viz.line_chart(
                [viz.Series(f"k={k}", [p[f"dead_k{k}"] for p in curve], i + 1, xs=xs) for i, k in enumerate(ks)],
                width=420, height=200, x_label="step", y_fmt=lambda v: _pct(v, 1))),
        ),
    )
# {{/docs-fragment training}}


def _metrics_table(trained: list) -> str:
    rows = []
    for t in sorted(trained, key=lambda t: (t.layer, t.width)):
        for s in t.stats:
            rows.append([
                f"<b>{s.layer}</b>", f"{s.width:,}", s.k, f"{s.fve:.3f}", f"{s.l0:.1f}", _pct(s.dead_frac, 1),
                f"{s.ce_clean:.3f}", f"{s.ce_sae:.3f}", f"{s.ce_zero:.2f}",
                f"<b>{_pct(s.loss_recovered, 1)}</b>",
            ])
    return viz.table(["layer", "width", "k", "variance explained", "L0", "dead", "loss clean", "loss with SAE",
                      "loss zero-ablated", "loss recovered"], rows)


def _frontier(trained: list, key: str, label: str, fmt) -> str:
    series = []
    for i, t in enumerate(sorted(trained, key=lambda t: (t.layer, t.width))):
        pts = sorted(t.stats, key=lambda s: s.l0)
        series.append(viz.Series(f"L{t.layer} · {t.width:,}", [getattr(s, key) for s in pts], i + 1,
                                 xs=[s.l0 for s in pts]))
    return viz.panel(label, viz.line_chart(series, width=440, height=230, x_label="L0", y_fmt=fmt,
                                           x_fmt=lambda v: f"{v:.0f}"))


def _density(trained: list, k: int) -> str:
    """The classic feature-density histogram: how often each feature fires, on a log scale."""
    panels = []
    for t in sorted(trained, key=lambda t: (t.layer, t.width)):
        s = next((s for s in t.stats if s.k == k), t.stats[0])
        h = s.density_hist
        cats = ["dead"] + [f"{e:g}" for e in h["edges"][:-1]]
        vals = [h["dead"]] + h["counts"]
        panels.append(viz.panel(f"Layer {t.layer}, {t.width:,} features, k={s.k}", viz.stacked_columns(
            cats, [viz.Series("features", vals, 1)], width=420, height=170),
            "x: log10 of the share of tokens a feature fires on; the first column is dead features."))
    return viz.row(*panels)


def _autointerp_table(labeled: list[dict]) -> str:
    rows = []
    for lab in sorted(labeled, key=lambda x: x["layer"]):
        sc = [f["score"] for f in lab["features"]]
        ct = [f["control"] for f in lab["features"]]
        rows.append([f"<b>{lab['layer']}</b>", f"{lab['width']:,}", lab["k"], len(sc), _score(sum(sc) / len(sc)),
                     f"{sum(ct) / len(ct):.2f}", f"{sum(s >= SCORE_GOOD for s in sc)} of {len(sc)}"])
    return viz.table(["layer", "width", "k", "features labeled", "detection score", "mismatched explanation",
                      f"score ≥ {SCORE_GOOD}"], rows)


# {{docs-fragment overview}}
def overview(trained: list, labeled: list[dict], harvest, elapsed: float) -> str:
    best = max((s for t in trained for s in t.stats), key=lambda s: s.loss_recovered)
    scores = [f["score"] for lab in labeled for f in lab["features"]]
    controls = [f["control"] for lab in labeled for f in lab["features"]]
    label_k = next((lab["k"] for lab in labeled), trained[0].ks[len(trained[0].ks) // 2])
    tiles = viz.tiles([
        ("Best loss recovered", _pct(best.loss_recovered, 1), f"layer {best.layer}, {best.width:,} features, k={best.k}"),
        ("SAEs trained", str(sum(len(t.stats) for t in trained)), f"{len(trained)} GPU tasks"),
        ("Features labeled by Claude", str(len(scores)) if scores else "pending"),
        ("Mean detection score", f"{sum(scores) / len(scores):.2f}" if scores else "pending",
         f"mismatched explanations: {sum(controls) / len(controls):.2f}; chance: 0.50" if controls else ""),
        ("Wall clock", f"{elapsed / 60:.0f} min"),
    ], hero=True)
    blocks = [
        tiles,
        viz.panel("Every SAE on held-out text", _metrics_table(trained),
                  "Variance explained and L0 on held-out tokens. Loss recovered = (zero-ablated − with SAE) / "
                  "(zero-ablated − clean), from next-token loss with the layer's output replaced."),
        viz.row(
            _frontier(trained, "fve", "Sparsity vs reconstruction: variance explained", lambda v: f"{v:.2f}"),
            _frontier(trained, "loss_recovered", "Sparsity vs downstream loss: loss recovered", lambda v: _pct(v)),
        ),
    ]
    if labeled:
        blocks.append(viz.panel("Autointerp: can Claude predict where a feature fires from its explanation?",
                                _autointerp_table(labeled),
                                "Balanced accuracy on held-out contexts, half activating and half not. "
                                "The control scores each feature with another feature's explanation. "
                                "Open the Feature browser tab for every feature."))
    blocks.append(_density(trained, label_k))
    blocks.append(viz.panel("Data", viz.table(["model", "dataset", "contexts", "tokens", "layers"], [[
        f"<code>{viz.esc(harvest.model)}</code>", f"<code>{viz.esc(harvest.dataset)}</code>", f"{harvest.n_seqs:,}",
        viz.compact(harvest.n_seqs * harvest.ctx_len), ", ".join(map(str, harvest.layers))]])))
    return viz.page("Sparse autoencoders on Pythia-70m",
                    "TopK SAEs on the residual stream, one GPU task per (layer, width), features labeled and "
                    "scored by Claude.", *blocks)
# {{/docs-fragment overview}}


# ── feature browser ─────────────────────────────────────────────────────────

BROWSER_STYLE = """
<style>
.viz .feat { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px; margin: 0 0 12px; }
.viz .feat .head { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline; margin-bottom: 6px; }
.viz .feat .label { font-size: 15px; font-weight: 600; }
.viz .feat .meta { font-size: 12px; color: var(--ink-2); }
.viz .feat .notes { font-size: 12px; color: var(--ink-2); margin: 0 0 8px; }
.viz .ctx { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; line-height: 1.9;
  white-space: pre-wrap; word-break: break-word; border-bottom: 1px solid var(--grid); padding: 2px 0; }
.viz .ctx span.t { border-radius: 3px; }
.viz .fgrid { display: grid; grid-template-columns: minmax(0, 1fr) 190px; gap: 14px; }
@media (max-width: 700px) { .viz .fgrid { grid-template-columns: 1fr; } }
</style>
"""


def _token(t: str, a: float, peak: float) -> str:
    text = html.escape(t.replace("\n", "↵"))
    if a <= 0:
        return text
    pct = max(12, round(85 * a / peak))
    return (f"<span class='t' style='background:color-mix(in srgb, var(--s2) {pct}%, transparent)' "
            f"title='{a:.2f}'>{text}</span>")


def context_html(ex: dict, peak: float) -> str:
    return "<div class='ctx'>" + "".join(_token(t, a, peak) for t, a in zip(ex["tokens"], ex["acts"])) + "</div>"


def _hist(h: dict, width: int = 190, height: int = 60) -> str:
    counts = h["counts"]
    top = max(counts) or 1
    bw = width / len(counts)
    bars = "".join(
        f"<rect x='{i * bw + 0.5:.1f}' y='{height - height * c / top:.1f}' width='{bw - 1:.1f}' "
        f"height='{height * c / top:.1f}' style='fill:var(--s1)'><title>{h['edges'][i]:.2f}–{h['edges'][i + 1]:.2f}: "
        f"{c:,} tokens</title></rect>" for i, c in enumerate(counts))
    axis = (f"<text x='0' y='{height + 12}'>0</text>"
            f"<text x='{width}' y='{height + 12}' text-anchor='end'>{h['edges'][-1]:.1f}</text>")
    return f"<svg viewBox='0 0 {width} {height + 14}' style='max-width:{width}px' role='img'>{bars}{axis}</svg>"


def feature_html(f: dict, layer: int, width: int, k: int, n_contexts: int = 10) -> str:
    peak = f["max_act"]
    ctxs = "".join(context_html(ex, peak) for ex in f["top"][:n_contexts])
    more = "".join(context_html(ex, peak) for ex in f["top"][n_contexts:] + f.get("sampled", []))
    held = "".join(context_html(ex, peak) for ex in f["test"] if ex["active"])
    return (
        "<div class='feat'><div class='head'>"
        f"<span class='label'>{viz.esc(f['explanation'])}</span>"
        f"<span class='meta'>detection {_score(f['score'])} · mismatched {f['control']:.2f}</span>"
        f"<span class='meta'>layer {layer} · {width:,} features · k={k} · feature #{f['feature']}</span>"
        f"<span class='meta'>fires on {_pct(f['density'], 2)} of tokens, {_pct(f['context_rate'], 1)} of contexts</span>"
        "</div>"
        f"<p class='notes'>{viz.esc(f['notes'])}</p>"
        f"<div class='fgrid'><div>{ctxs}"
        + (f"<details><summary>{len(f['top']) - n_contexts} more top contexts, then a random sample of the rest</summary>{more}</details>" if more else "")
        + (f"<details><summary>held-out contexts where it fires (from the detection test)</summary>{held}</details>" if held else "")
        + f"</div><div><div class='meta'>activation histogram (nonzero)</div>{_hist(f['hist'])}</div></div></div>"
    )


# {{docs-fragment browser}}
def browser(labeled: list[dict]) -> str:
    """Every labeled feature, most convincing first: highest detection score, then the widest margin over the control."""
    feats = [(f, lab) for lab in labeled for f in lab["features"]]
    feats.sort(key=lambda p: (-p[0]["score"], -(p[0]["score"] - p[0]["control"]), -p[0]["context_rate"]))
    good = sum(f["score"] >= SCORE_GOOD for f, _ in feats)
    return viz.page(
        "Feature browser",
        f"{len(feats)} features sampled at random from the live features of each SAE, sorted by how well Claude's "
        f"explanation predicts held-out contexts. {good} score at least {SCORE_GOOD}. Highlight strength is the "
        "feature's activation; hover a token for its value.",
        BROWSER_STYLE,
        "".join(feature_html(f, lab["layer"], lab["width"], lab["k"]) for f, lab in feats),
    )
# {{/docs-fragment browser}}


def card(best, stats, labeled: dict, run_name: str) -> str:
    feats = sorted(labeled["features"], key=lambda f: -f["score"])
    scores = [f["score"] for f in feats]
    return viz.card(
        f"TopK SAE: Pythia-70m-deduped, layer {stats.layer}",
        f"Trained in run <code>{viz.esc(run_name)}</code> on the residual stream after layer {stats.layer}. "
        f"{stats.width:,} features, k={stats.k}.",
        ["PyTorch", "TopK SAE", "Pythia-70m-deduped", f"layer {stats.layer}", f"{stats.width:,} features", ".pt"],
        BROWSER_STYLE,
        viz.tiles([
            ("Loss recovered", _pct(stats.loss_recovered, 1), f"loss {stats.ce_clean:.3f} → {stats.ce_sae:.3f}"),
            ("Variance explained", f"{stats.fve:.3f}", f"L0 {stats.l0:.0f}"),
            ("Dead features", _pct(stats.dead_frac, 1), "on held-out tokens"),
            ("Detection score", f"{sum(scores) / len(scores):.2f}", f"{len(scores)} features labeled by Claude"),
        ], hero=True),
        viz.panel("Its five most interpretable sampled features",
                  "".join(feature_html(f, stats.layer, stats.width, stats.k, n_contexts=4) for f in feats[:5])),
        viz.panel("Load it", viz.code(
            "import torch\n"
            "from flyte.io import File\n"
            "from flyte.remote import Artifact\n"
            'sae_file = Artifact.get("pythia-70m-topk-sae").to_python(File)\n'
            "ckpt = torch.load(sae_file.download_sync())\n"
            '# {"state_dict", "scale", "layer", "width", "k", "hook"}: TopKSAE in sae.py; multiply\n'
            "# activations by scale before encoding and divide reconstructions by it after.")),
    )
