"""The discovery report and its tabs, the approval prompt, and the dataset card.

Everything here reads the loop state (a list of rounds, each with its proposal and its
measurements) and turns it into HTML with viz.py, or Markdown for the approval prompt.
RDKit is imported only where molecules are drawn, so launching a run needs flyte alone.
"""

from __future__ import annotations

import csv
import math
from statistics import NormalDist, median

import viz

ACTIVE, CONTROL = 1, 2  # categorical color slots
ARMS = (("active", "active learning", ACTIVE), ("control", "random search", CONTROL))


# ── numbers ─────────────────────────────────────────────────────────────────


def curve(state, arm: str, hit_gap: float) -> list[dict]:
    """After each round: oracle calls spent by the arm, best gap so far, and hits below ``hit_gap``."""
    out, calls, best, hits = [], 0, None, set()
    for r in state.rounds:
        ms = getattr(r, arm)
        calls += len(ms)
        for m in ms:
            if m.ok:
                best = m.gap if best is None else min(best, m.gap)
                if m.gap < hit_gap:
                    hits.add(m.smiles)
        out.append({"round": r.round, "calls": calls, "best": best, "hits": len(hits)})
    return out


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    for rank, i in enumerate(order):
        ranks[i] = float(rank)
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra))
    sb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (sa * sb) if sa and sb else None


def pairs(picks, measurements) -> list[tuple[float, float, float]]:
    """(predicted mean, predicted std, measured gap) for each successfully measured pick."""
    by = {m.smiles: m for m in measurements if m.ok}
    return [(p.mean, p.std, by[p.smiles].gap) for p in picks if p.smiles in by]


def errors(pts: list[tuple[float, float, float]]) -> dict:
    if not pts:
        return {}
    res = [abs(y - mu) for mu, _, y in pts]
    return {
        "n": len(pts),
        "mae": sum(res) / len(res),
        "rho": spearman([mu for mu, _, _ in pts], [y for _, _, y in pts]),
        "in1": sum(e <= sd for e, (_, sd, _) in zip(res, pts)) / len(pts),
        "in2": sum(e <= 2 * sd for e, (_, sd, _) in zip(res, pts)) / len(pts),
        "sd": sum(sd for _, sd, _ in pts) / len(pts),
    }


def round_metrics(state) -> list[dict]:
    """The surrogate's prospective error each round: on the batch it chose, and on the random control."""
    rows = []
    for r in state.rounds:
        if r.proposal is None or not r.active:
            continue
        rows.append({"round": r.round, "train": r.proposal.train_size,
                     "active": errors(pairs(r.proposal.batch, r.active)),
                     "control": errors(pairs(r.proposal.control, r.control))})
    return rows


def calibration(state) -> list[tuple[float, float]]:
    """Expected against observed coverage of central intervals, pooled over every control molecule."""
    pts = [p for r in state.rounds if r.proposal and r.control for p in pairs(r.proposal.control, r.control)]
    if not pts:
        return []
    out = []
    for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        z = NormalDist().inv_cdf(0.5 + q / 2)
        out.append((q, sum(abs(y - mu) <= z * max(sd, 1e-9) for mu, sd, y in pts) / len(pts)))
    return out


def all_measurements(state) -> list:
    seen, out = set(), []
    for r in state.rounds:
        for m in r.active + r.control:
            if m.smiles not in seen:
                seen.add(m.smiles)
                out.append(m)
    return out


def top_molecules(state, n: int = 50) -> list[dict]:
    """The smallest-gap molecules from both arms, with who found them and what the model predicted."""
    found: dict[str, dict] = {}
    for r in state.rounds:
        preds = {p.smiles: p for p in (r.proposal.batch + r.proposal.control if r.proposal else [])}
        for arm, label, _ in ARMS:
            for m in getattr(r, arm):
                if not m.ok:
                    continue
                row = found.setdefault(m.smiles, {
                    "smiles": m.smiles, "gap_ev": round(m.gap, 4), "homo_ev": round(m.homo, 4),
                    "lumo_ev": round(m.lumo, 4), "atoms": m.atoms, "round": r.round, "found_by": set(),
                    "predicted_gap_ev": round(preds[m.smiles].mean, 4) if m.smiles in preds else None,
                    "predicted_std_ev": round(preds[m.smiles].std, 4) if m.smiles in preds else None,
                })
                row["found_by"].add("seed library" if r.round == 0 else label)
    rows = sorted(found.values(), key=lambda d: d["gap_ev"])[:n]
    for i, row in enumerate(rows, 1):
        row["rank"] = i
        row["found_by"] = " + ".join(sorted(row["found_by"]))
    return rows


def summary(state, hit_gap: float) -> dict:
    a, c = curve(state, "active", hit_gap)[-1], curve(state, "control", hit_gap)[-1]
    ms = all_measurements(state)
    return {"rounds": len(state.rounds) - 1, "calls": len(ms), "failed": sum(not m.ok for m in ms),
            "best_active_ev": f"{a['best']:.3f}", "best_random_ev": f"{c['best']:.3f}",
            "hits_active": a["hits"], "hits_random": c["hits"], "hit_gap_ev": hit_gap}


def write_csv(rows: list[dict], path: str) -> None:
    cols = ["rank", "smiles", "gap_ev", "homo_ev", "lumo_ev", "atoms", "found_by", "round",
            "predicted_gap_ev", "predicted_std_ev"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── the approval prompt (Markdown) ─────────────────────────────────────────


def approval_prompt(proposal, state, timeout_s: int) -> str:
    batch = proposal.batch
    below = sum(p.mean < 1.0 for p in batch)
    lines = [
        f"### Round {proposal.round}: {len(batch)} molecules proposed for the oracle",
        "",
        f"The surrogate (an 8-model MLP ensemble trained on {proposal.train_size} measured molecules) predicts "
        f"a mean gap of **{sum(p.mean for p in batch) / len(batch):.2f} eV** for this batch, against "
        f"{proposal.pool_median:.2f} eV for the median molecule in the pool. {below} are predicted below 1 eV.",
        "",
    ]
    metrics = round_metrics(state)
    if metrics and metrics[-1]["control"]:
        m = metrics[-1]["control"]
        lines += [f"Last round, on {m['n']} random control molecules, its error was {m['mae']:.2f} eV (MAE), "
                  f"with {m['in1']:.0%} inside one predicted standard deviation.", ""]
    a, c = curve(state, "active", 1.0)[-1], curve(state, "control", 1.0)[-1]
    lines += [f"Best gap measured so far: **{a['best']:.2f} eV** by active learning, {c['best']:.2f} eV by random search.", "",
              "| # | SMILES | predicted gap (eV) |", "| --- | --- | --- |"]
    lines += [f"| {i} | `{p.smiles}` | {p.mean:.2f} ± {p.std:.2f} |" for i, p in enumerate(batch[:10], 1)]
    if len(batch) > 10:
        lines += [f"| | and {len(batch) - 10} more | |"]
    lines += ["", f"**Approve** (true) sends this batch and {len(proposal.control)} random control molecules to the "
              "oracle. **Reject** (false) skips the round, and the rejected molecules are never proposed again. "
              f"With no answer in {timeout_s} s, the batch is approved automatically."]
    return "\n".join(lines)


# ── drawing ─────────────────────────────────────────────────────────────────


def mol_svg(smiles: str, width: int = 220, height: int = 150) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        d.drawOptions().clearBackground = False
        d.drawOptions().padding = 0.08
        d.DrawMolecule(Chem.MolFromSmiles(smiles))
        d.FinishDrawing()
        svg = d.GetDrawingText()
        return svg[svg.find("<svg"):]
    except Exception as e:
        print(f"could not draw {smiles}: {e}")
        return f"<code>{viz.esc(smiles)}</code>"


def molecule_grid(rows: list[dict], n: int = 12) -> str:
    cells = []
    for row in rows[:n]:
        pred = (f" · predicted {row['predicted_gap_ev']:.2f} ± {row['predicted_std_ev']:.2f}"
                if row.get("predicted_gap_ev") is not None else "")
        cells.append(
            f"<div class='mol'><div class='draw'>{mol_svg(row['smiles'])}</div>"
            f"<div class='v'>#{row['rank']} · {row['gap_ev']:.2f} eV</div>"
            f"<div class='meta'>{viz.esc(row['found_by'])}, round {row['round']}{pred}</div>"
            f"<code>{viz.esc(row['smiles'])}</code></div>")
    return f"<div class='mols'>{''.join(cells)}</div>"


def scatter(points: list[tuple[float, float, int, str]], *, width: int = 420, height: int = 300,
            x_label: str = "", y_label: str = "") -> str:
    """Points (x, y, color slot, tooltip) on equal axes, with the y = x diagonal."""
    if not points:
        return "<p class='note'>no data yet</p>"
    vals = [p[0] for p in points] + [p[1] for p in points]
    ticks = viz.nice_ticks(min(vals), max(vals))
    lo, hi = ticks[0], ticks[-1]
    L, R, T, B = 44, 12, 10, 34
    W, H = width - L - R, height - T - B

    def X(v):
        return L + W * (v - lo) / (hi - lo)

    def Y(v):
        return T + H * (1 - (v - lo) / (hi - lo))

    g = "".join(f"<line class='gl' x1='{L}' x2='{L + W}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>"
                f"<text x='{L - 6}' y='{Y(t) + 4:.1f}' text-anchor='end'>{viz.compact(t)}</text>"
                f"<text x='{X(t):.1f}' y='{T + H + 16}' text-anchor='middle'>{viz.compact(t)}</text>" for t in ticks)
    g += f"<line class='ref' x1='{X(lo):.1f}' y1='{Y(lo):.1f}' x2='{X(hi):.1f}' y2='{Y(hi):.1f}' stroke-dasharray='4 3'/>"
    g += f"<text x='{L + W}' y='{T + H + 30}' text-anchor='end'>{viz.esc(x_label)}</text>"
    g += f"<text x='{L}' y='{T - 2}'>{viz.esc(y_label)}</text>" if y_label else ""
    marks = "".join(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='3.5' style='fill:{viz.color(c)};opacity:.75'>"
                    f"<title>{viz.esc(t)}</title></circle>" for x, y, c, t in points)
    return f"<svg viewBox='0 0 {width} {height}' style='max-width:{width}px' role='img'>{g}{marks}</svg>"


# ── report pieces ───────────────────────────────────────────────────────────


def _curves(state, hit_gap: float) -> tuple[str, str]:
    series_best, series_hits = [], []
    for arm, label, c in ARMS:
        pts = curve(state, arm, hit_gap)
        series_best.append(viz.Series(label, [p["best"] for p in pts], c, xs=[p["calls"] for p in pts], step=True))
        series_hits.append(viz.Series(label, [p["hits"] for p in pts], c, xs=[p["calls"] for p in pts], step=True))
    best = viz.line_chart(series_best, x_label="oracle calls", y_fmt=lambda v: f"{v:.2f}", height=230)
    hits = viz.line_chart(series_hits, x_label="oracle calls", height=230)
    return best, hits


def _batch_means(state) -> str:
    series = []
    for arm, label, c in ARMS:
        xs, ys = [], []
        for r in state.rounds:
            ok = [m.gap for m in getattr(r, arm) if m.ok]
            if r.round > 0 and ok:
                xs.append(r.round)
                ys.append(sum(ok) / len(ok))
        series.append(viz.Series(label, ys, c, xs=xs))
    return viz.line_chart(series, x_label="round", y_fmt=lambda v: f"{v:.1f}", height=200)


def _tiles(state, hit_gap: float, total_rounds: int | None = None) -> str:
    a, c = curve(state, "active", hit_gap)[-1], curve(state, "control", hit_gap)[-1]
    ms = all_measurements(state)
    done = len(state.rounds) - 1
    return viz.tiles([
        (f"Molecules below {hit_gap:g} eV", f"{a['hits']} vs {c['hits']}", "active learning vs random, same budget"),
        ("Best gap, active learning", f"{a['best']:.2f} eV", f"random search: {c['best']:.2f} eV"),
        ("Round", f"{done} / {total_rounds}" if total_rounds else str(done)),
        ("Oracle calls", f"{a['calls']} + {c['calls']}", f"{len(ms)} distinct molecules"),
        ("Failed calculations", str(sum(not m.ok for m in ms)), "after retries"),
    ], hero=True)


def _batch_table(r, n: int | None = None) -> str:
    if r.proposal is None:
        rows = [[i, f"<code>{viz.esc(m.smiles)}</code>", "—", f"{m.gap:.2f}" if m.ok else viz.status("critical", "failed")]
                for i, m in enumerate(r.active[:n] if n else r.active, 1)]
        return viz.table(["#", "SMILES", "predicted", "measured gap (eV)"], rows)
    by = {m.smiles: m for m in r.active}
    rows = []
    for i, p in enumerate(r.proposal.batch[:n] if n else r.proposal.batch, 1):
        m = by.get(p.smiles)
        if m is None:
            measured = viz.status("neutral", "not sent")
        elif not m.ok:
            measured = viz.status("critical", "failed")
        else:
            measured = f"{m.gap:.2f}"
        rows.append([i, f"<code>{viz.esc(p.smiles)}</code>", f"{p.mean:.2f} ± {p.std:.2f}", f"{p.score:.2f}", measured])
    return viz.table(["#", "SMILES", "predicted gap (eV)", "LCB score", "measured gap (eV)"], rows)


def _decision(r) -> str:
    kind = {"approved": "good", "auto-approved": "warning", "rejected": "critical"}.get(r.decision, "neutral")
    wait = f" after {r.gate_seconds:.0f} s" if r.gate_seconds else ""
    return viz.status(kind, r.decision + wait)


def live(state, rounds: int, hit_gap: float, pool_size: int) -> str:
    best, hits = _curves(state, hit_gap)
    last = state.rounds[-1]
    return viz.page(
        "Closed-loop discovery: smallest HOMO-LUMO gap",
        f"Active learning against random search over {pool_size:,} ZINC molecules, with the same GFN2-xTB oracle "
        "and the same budget per round.",
        _tiles(state, hit_gap, rounds),
        viz.row(viz.panel("Best gap found (eV), lower is better", best),
                viz.panel(f"Molecules found below {hit_gap:g} eV", hits)),
        viz.panel(f"Round {last.round}: {_decision(last) if last.round else 'seed library'}", _batch_table(last, 10)),
    )


def final(state, hit_gap: float, pool_size: int, top: list[dict], elapsed: float) -> str:
    best, hits = _curves(state, hit_gap)
    return viz.page(
        "Closed-loop discovery: smallest HOMO-LUMO gap",
        f"{len(state.rounds) - 1} rounds over {pool_size:,} ZINC molecules; the rounds took {elapsed / 60:.0f} minutes. Each round, "
        "an MLP ensemble picked a batch by a lower confidence bound, a scientist (or a timeout) approved it, and "
        "GFN2-xTB measured it, next to a random batch of the same size.",
        _tiles(state, hit_gap),
        viz.row(viz.panel("Best gap found (eV), lower is better", best),
                viz.panel(f"Molecules found below {hit_gap:g} eV", hits)),
        viz.panel("Mean measured gap of each round's batch (eV)", _batch_means(state),
                  "Random batches sample the pool's average; the model's batches show where it thinks small gaps are."),
        viz.panel("Top molecules found", molecule_grid(top, 12),
                  "GFN2-xTB underestimates the gap of donor-acceptor molecules, so these are candidates for a "
                  "higher-level calculation, not final answers."),
    )


def surrogate_tab(state) -> str:
    metrics = round_metrics(state)
    rows = []
    for m in metrics:
        a, c = m["active"], m["control"]
        rows.append([m["round"], m["train"],
                     f"{a['mae']:.2f}" if a else "—", f"{c['mae']:.2f}" if c else "—",
                     f"{c['rho']:.2f}" if c and c["rho"] is not None else "—",
                     f"{c['in1']:.0%}" if c else "—", f"{c['in2']:.0%}" if c else "—", f"{c['sd']:.2f}" if c else "—"])
    mae = viz.line_chart(
        [viz.Series("on its own batch", [m["active"]["mae"] for m in metrics if m["active"]], ACTIVE,
                    xs=[m["round"] for m in metrics if m["active"]]),
         viz.Series("on the random control", [m["control"]["mae"] for m in metrics if m["control"]], CONTROL,
                    xs=[m["round"] for m in metrics if m["control"]])],
        x_label="round", y_fmt=lambda v: f"{v:.2f}", height=200)
    cal = calibration(state)
    cal_chart = viz.line_chart(
        [viz.Series("ideal", [q for q, _ in cal], "neutral", xs=[q for q, _ in cal]),
         viz.Series("observed", [o for _, o in cal], ACTIVE, xs=[q for q, _ in cal])],
        y_range=(0, 1), x_label="expected coverage", y_fmt=lambda v: f"{v:.0%}", x_fmt=lambda v: f"{v:.0%}", height=220)
    last = next((r for r in reversed(state.rounds) if r.proposal and r.active), None)
    pts = []
    if last:
        pts = [(mu, y, CONTROL, f"control · predicted {mu:.2f}, measured {y:.2f}") for mu, _, y in pairs(last.proposal.control, last.control)]
        pts += [(mu, y, ACTIVE, f"batch · predicted {mu:.2f}, measured {y:.2f}") for mu, _, y in pairs(last.proposal.batch, last.active)]
    return viz.page(
        "Surrogate",
        "Every prediction here is prospective: the model made it before the molecule was measured. The random "
        "control batch is an unbiased test set; the model's own batch is where it is most optimistic.",
        viz.row(viz.panel("Mean absolute error per round (eV)", mae),
                viz.panel("Calibration, pooled over the random control", cal_chart,
                          "Share of molecules whose measured gap falls inside the central interval of the "
                          "predicted normal distribution. Below the diagonal: overconfident.")),
        viz.panel(f"Predicted against measured, round {last.round if last else '—'}",
                  viz.legend([("model's batch", ACTIVE), ("random control", CONTROL)])
                  + scatter(pts, x_label="predicted gap (eV)", y_label="measured gap (eV)")),
        viz.panel("Per round", viz.table(
            ["round", "trained on", "MAE, own batch", "MAE, control", "Spearman, control", "within 1σ", "within 2σ", "mean σ"], rows),
            "Within 1σ and 2σ: the share of control molecules inside one and two predicted standard deviations "
            "(68% and 95% if calibrated)."),
    )


def batches_tab(state) -> str:
    blocks = []
    for r in state.rounds:
        if r.round == 0:
            blocks.append(viz.panel("Round 0: seed library", _batch_table(r),
                                    "Random molecules measured before any model exists, shared by both arms."))
            continue
        ok = [m.gap for m in r.active if m.ok]
        note = f"Gate: {_decision(r)}." + (f" Measured mean gap {sum(ok) / len(ok):.2f} eV." if ok else "")
        body = (f"<p class='note'>{note}</p><details><summary>all {len(r.proposal.batch)} molecules</summary>"
                f"{_batch_table(r)}</details>")
        blocks.append(viz.panel(f"Round {r.round}", body))
    return viz.page("Batches", "Every batch the model proposed, what it predicted, and what the oracle measured.", *blocks)


def oracle_tab(state, run_name: str) -> str:
    ms = all_measurements(state)
    failed = [m for m in ms if not m.ok]
    retried = [m for m in ms if m.ok and m.attempts > 1]
    reused = [m for m in ms if m.measured_in and m.measured_in != run_name]
    secs = sorted(m.seconds for m in ms if m.ok)
    gates = [[r.round, _decision(r)] for r in state.rounds if r.round > 0]
    fail_rows = [[f"<code>{viz.esc(m.smiles)}</code>", m.attempts, viz.esc(m.error)] for m in failed]
    retry_rows = [[f"<code>{viz.esc(m.smiles)}</code>", m.attempts, f"{m.gap:.2f}", viz.esc(m.error)] for m in retried]
    return viz.page(
        "Oracle",
        "GFN2-xTB on every molecule: 8 ETKDG conformers, MMFF relaxation, xTB optimization of the best 3, and a "
        "single point. A failed attempt is retried with random-coordinate embedding and a higher electronic temperature.",
        viz.tiles([
            ("Distinct molecules measured", str(len(ms))),
            ("Failed after every retry", str(len(failed)), f"{len(failed) / max(len(ms), 1):.1%}"),
            ("Recovered by a retry", str(len(retried))),
            ("Median time per molecule", f"{median(secs):.1f} s" if secs else "—", f"max {secs[-1]:.0f} s" if secs else ""),
            ("Reused from earlier runs", str(len(reused)), "cache hits, never recomputed"),
        ]),
        viz.row(viz.panel("Human approval per round", viz.table(["round", "decision"], gates)),
                viz.panel("Recovered by a retry", viz.table(["SMILES", "attempts", "gap (eV)", "earlier errors"], retry_rows)
                          if retry_rows else "<p class='note'>none</p>")),
        viz.panel("Failed measurements", viz.table(["SMILES", "attempts", "errors"], fail_rows) if fail_rows
                  else "<p class='note'>none</p>"),
    )


def card(state, top: list[dict], hit_gap: float) -> str:
    s = summary(state, hit_gap)
    best, hits = _curves(state, hit_gap)
    preview = viz.table(["rank", "SMILES", "gap (eV)", "found by"],
                        [[r["rank"], f"<code>{viz.esc(r['smiles'])}</code>", f"{r['gap_ev']:.3f}", viz.esc(r["found_by"])]
                         for r in top[:20]])
    return viz.card(
        "Closed-loop discovery: top candidates",
        f"The {len(top)} smallest GFN2-xTB HOMO-LUMO gaps found by active learning and a random control over "
        f"{s['rounds']} rounds, {s['calls']} calculations in all.",
        ["dataset", "CSV", "GFN2-xTB", "ZINC250k", "active learning"],
        viz.tiles([("Best gap", f"{top[0]['gap_ev']:.2f} eV", viz.esc(top[0]["found_by"])),
                   (f"Below {hit_gap:g} eV", f"{s['hits_active']} vs {s['hits_random']}", "active learning vs random"),
                   ("Failed calculations", str(s["failed"]))], hero=True),
        viz.row(viz.panel("Best gap found (eV)", best), viz.panel(f"Molecules below {hit_gap:g} eV", hits)),
        viz.panel("Top molecules", molecule_grid(top, 8)),
        viz.panel("First 20 rows", preview),
        viz.panel("Columns", viz.table(["column", "meaning"], [
            ["gap_ev, homo_ev, lumo_ev", "GFN2-xTB orbital energies of the lowest-energy optimized conformer"],
            ["found_by", "seed library, active learning, random search, or both"],
            ["predicted_gap_ev, predicted_std_ev", "the surrogate's prediction before the molecule was measured"],
        ])),
        viz.panel("Load it", viz.code(
            'from flyte.remote import Artifact\n'
            'top = Artifact.get("closed-loop-top-candidates")   # a CSV file')),
    )
