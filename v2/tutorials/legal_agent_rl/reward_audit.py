"""What does a rubric reward pay for? Read a sweep's verdicts the way a GRPO trainer would.

Harvey's reward is ``0.35 c + 0.35 h + 0.30 a``: the fraction of criteria passed, the fraction of issue groups
passed in full, and an all-pass bonus, with zero for an episode cut off by its turn or context limit. The trainer
never sees those terms on their own. It sees each rollout's advantage, its reward minus its group's mean. So the
questions here are about groups:

1. **Which term carries the advantage?** Within each (task, policy) group, the share of the reward's variance that
   comes from each term (``Cov(w_k x_k, R) / Var(R)``, which sums to 1), and how often a term is constant across
   the group, so contributes nothing.
2. **How loud is the judge?** The same rollouts re-graded by the same judge: how much a rollout's reward moves between
   gradings, as a fraction of how much rewards differ within its group. That fraction is the share of the advantage
   that is judge noise.
3. **Do two judges agree, and does a judge favour its own family?** Criterion-level agreement between judges, and the
   gap between them by policy family.

Pure functions over the sweep's JSONL, so re-weighting the reward or adding a judge recomputes in seconds on a CPU.
The rollouts and grades it reads are the expensive part, and they're already paid for.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics as st
from collections import defaultdict
from typing import Iterable

from lab_harness import TRUNCATED, RewardWeights, reward, reward_terms


def load(path: str) -> tuple[list[dict], dict[str, dict[str, list[str]]]]:
    rollouts, groups = [], {}
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["kind"] == "groups":
                groups[row["task_id"]] = row["groups"]
            else:
                rollouts.append(row)
    return rollouts, groups


def grades_of(r: dict, judge: str, sample: int = 0) -> dict[str, str] | None:
    return next((g["verdicts"] for g in r["grades"] if g["judge"] == judge and g["judge_sample"] == sample), None)


def scored(r: dict, groups: dict, judge: str, sample: int = 0, w: RewardWeights = RewardWeights()) -> dict | None:
    v = grades_of(r, judge, sample)
    if v is None:
        return None
    t = reward_terms(v, groups.get(r["task_id"], {}))
    trunc = r["finish_reason"] in TRUNCATED and w.zero_on_truncation
    return {**t, "trunc": trunc, "R": reward(t, r["finish_reason"], w), "errors": sum(x == "error" for x in v.values())}


def _var(xs: list[float]) -> float:
    return st.pvariance(xs) if len(xs) > 1 else 0.0


def _cov(xs: list[float], ys: list[float]) -> float:
    mx, my = st.fmean(xs), st.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)


# {{docs-fragment term_shares}}
def term_shares(rows: list[dict], w: RewardWeights = RewardWeights()) -> dict:
    """Where a group's reward variance comes from. ``rows`` is one group: G scored rollouts of one task.

    R = w_c c + w_h h + w_a a - trunc * (w_c c + w_h h + w_a a): three rubric parts and one truncation part. Each
    part's share of the group's reward variance is Cov(part, R) / Var(R), and the four shares sum to 1. A term that's
    constant across the group has share 0: it moves no rollout's advantage.
    """
    R = [x["R"] for x in rows]
    vr = _var(R)
    raw = [w.criteria * x["c"] + w.groups * x["h"] + w.all_pass * x["a"] for x in rows]
    parts = {"c": [w.criteria * x["c"] for x in rows], "h": [w.groups * x["h"] for x in rows],
             "a": [w.all_pass * x["a"] for x in rows], "trunc": [-r if x["trunc"] else 0.0 for x, r in zip(rows, raw)]}
    out = {"var_R": vr, "constant": {k: _var([x[k] for x in rows]) == 0 for k in ("c", "h", "a")},
           "all_zero_R": vr == 0}
    if vr > 0:
        out["share"] = {k: _cov(p, R) / vr for k, p in parts.items()}
    return out
# {{/docs-fragment term_shares}}


def by_group(rollouts: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    g: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rollouts:
        g[(r["policy"], r["task_id"])].append(r)
    return g


def policy_table(rollouts: list[dict], groups: dict, judge: str) -> dict[str, dict]:
    out = {}
    for pol, rs in sorted(_by(rollouts, "policy").items()):
        sc = [s for r in rs if (s := scored(r, groups, judge))]
        out[pol] = {
            "rollouts": len(rs), "graded": len(sc),
            "criteria_pass": st.fmean(s["c"] for s in sc) if sc else None,
            "groups_pass": st.fmean(s["h"] for s in sc) if sc else None,
            "all_pass": st.fmean(s["a"] for s in sc) if sc else None,
            "mean_reward": st.fmean(s["R"] for s in sc) if sc else None,
            "truncated": st.fmean(r["finish_reason"] in TRUNCATED for r in rs),
            "no_deliverable": st.fmean(r["deliverable_chars"] == 0 for r in rs),
            "turns_median": st.median(r["turns"] for r in rs),
            "output_tokens_median": st.median(r["output_tokens"] for r in rs),
            "finish_reasons": _counts(r["finish_reason"] for r in rs),
        }
    return out


def advantage_audit(rollouts: list[dict], groups: dict, judge: str, w: RewardWeights = RewardWeights()) -> dict:
    out = {}
    for pol, rs in sorted(_by(rollouts, "policy").items()):
        gs = [term_shares(sc, w) for rows in by_group(rs).values()
              if len(sc := [s for r in rows if (s := scored(r, groups, judge, 0, w))]) > 1]
        live = [g for g in gs if not g["all_zero_R"]]
        out[pol] = {
            "groups": len(gs),
            "groups_with_no_signal": sum(g["all_zero_R"] for g in gs),
            "term_constant_in_group": {k: sum(g["constant"][k] for g in gs) for k in ("c", "h", "a")},
            "mean_share": {k: st.fmean(g["share"][k] for g in live) for k in ("c", "h", "a", "trunc")} if live else None,
        }
    return out


# {{docs-fragment judge_noise}}
def judge_noise(rollouts: list[dict], groups: dict, judge: str) -> dict:
    """Re-grade noise against group signal. For each group: the mean over rollouts of the variance of R across the
    judge's samples (noise), against the variance of R across the group's rollouts on sample 0 (signal)."""
    out = {}
    for pol, rs in sorted(_by(rollouts, "policy").items()):
        flips, n_crit, ratios, d_c, a_flip = 0, 0, [], [], 0
        n_roll = 0
        for rows in by_group(rs).values():
            per = []
            for r in rows:
                samples = sorted({g["judge_sample"] for g in r["grades"] if g["judge"] == judge})
                if len(samples) < 2:
                    continue
                vs = [grades_of(r, judge, k) for k in samples]
                for cid in vs[0]:
                    vals = {v.get(cid) for v in vs}
                    n_crit += 1
                    flips += len(vals) > 1
                sc = [scored(r, groups, judge, k) for k in samples]
                d_c.append(max(s["c"] for s in sc) - min(s["c"] for s in sc))
                a_flip += len({s["a"] for s in sc}) > 1
                n_roll += 1
                per.append((_var([s["R"] for s in sc]), sc[0]["R"]))
            if len(per) > 1:
                signal = _var([p[1] for p in per])
                if signal > 0:
                    ratios.append(st.fmean(p[0] for p in per) / signal)
        out[pol] = {"criterion_flip_rate": flips / n_crit if n_crit else None,
                    "rollouts_regraded": n_roll,
                    "median_c_range": st.median(d_c) if d_c else None,
                    "all_pass_changed": a_flip / n_roll if n_roll else None,
                    "noise_to_signal_median": st.median(ratios) if ratios else None,
                    "noise_to_signal_p90": sorted(ratios)[int(0.9 * (len(ratios) - 1))] if ratios else None}
    return out
# {{/docs-fragment judge_noise}}


def judge_agreement(rollouts: list[dict], groups: dict, a: str, b: str) -> dict:
    """Criterion-level agreement between two judges (sample 0 each), and their criteria-pass gap by policy."""
    both = n = 0
    pa = pb = 0
    gap: dict[str, list[float]] = defaultdict(list)
    for r in rollouts:
        va, vb = grades_of(r, a), grades_of(r, b)
        if not va or not vb:
            continue
        for cid, x in va.items():
            y = vb.get(cid)
            if y is None or "error" in (x, y):
                continue
            n += 1
            both += x == y
            pa += x == "pass"
            pb += y == "pass"
        sa, sb = scored(r, groups, a), scored(r, groups, b)
        gap[r["policy"]].append(sa["c"] - sb["c"])
    if not n:
        return {}
    po, p1, p2 = both / n, pa / n, pb / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return {"criteria": n, "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None,
            "pass_rate": {a: p1, b: p2},
            "c_gap_by_policy": {p: {"mean": st.fmean(g), "se": st.stdev(g) / math.sqrt(len(g)) if len(g) > 1 else None,
                                    "n": len(g)} for p, g in sorted(gap.items())}}


def all_pass_ceiling(rollouts: list[dict], groups: dict, judge: str) -> dict:
    """If criteria passed independently at each policy's observed per-task rates, how often would anything all-pass?"""
    out = {}
    for pol, rs in sorted(_by(rollouts, "policy").items()):
        probs = []
        for rows in by_group(rs).values():
            vs = [v for r in rows if (v := grades_of(r, judge))]
            if not vs:
                continue
            ids = list(vs[0])
            p = [st.fmean(v.get(i) == "pass" for v in vs) for i in ids]
            probs.append(math.prod(p))
        out[pol] = {"expected_all_pass_if_independent": st.fmean(probs) if probs else None, "tasks": len(probs)}
    return out


def report(path: str, judge_a: str, judge_b: str | None = None, w: RewardWeights = RewardWeights()) -> dict:
    rollouts, groups = load(path)
    out = {"weights": w.model_dump(), "judge": judge_a,
           "policies": policy_table(rollouts, groups, judge_a),
           "advantage": advantage_audit(rollouts, groups, judge_a, w),
           "judge_noise": judge_noise(rollouts, groups, judge_a),
           "all_pass_ceiling": all_pass_ceiling(rollouts, groups, judge_a),
           "issue_groups": {"tasks": len(groups), "median_groups_per_task": st.median(len(g) for g in groups.values())
                            if groups else None}}
    if judge_b:
        out["agreement"] = judge_agreement(rollouts, groups, judge_a, judge_b)
    return out


def _by(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r[key]].append(r)
    return d


def _counts(xs):
    d: dict = defaultdict(int)
    for x in xs:
        d[x] += 1
    return dict(d)


if __name__ == "__main__":
    import sys

    print(json.dumps(report(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None), indent=1))
