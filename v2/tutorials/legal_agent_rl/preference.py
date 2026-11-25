"""Pairwise preference between two deliverables for the same task, and the statistics to read it with.

Harvey evaluates with a pairwise generative reward model: work products from two models, shown in both orders with
identities hidden, converted to Elo. Their prompt isn't published; ``PREFERENCE_PROMPT`` is ours, written to be
neutral about length (it doesn't tell the judge to prefer shorter or longer work), and every result names it.
"""

from __future__ import annotations

import math
import random
import statistics as st
from typing import Any

PREFERENCE_PROMPT = """You are a senior partner at a law firm reviewing two draft work products prepared for the same
client request. Decide which draft you would rather send to the client after light edits.

## The request
{instructions}

## Draft 1
{draft_1}

## Draft 2
{draft_2}

Judge the drafts as finished legal work for this request. Respond with JSON only:
{{"reasoning": "<two or three sentences>", "preferred": 1 or 2}}"""

_SCHEMA = {"type": "object", "properties": {"reasoning": {"type": "string"}, "preferred": {"type": "integer",
                                                                                            "enum": [1, 2]}},
           "required": ["reasoning", "preferred"], "additionalProperties": False}


def judge_pair(client, model: str, instructions: str, first: str, second: str, max_chars: int = 60_000,
               reasoning_effort: str | None = "low", temperature: float = 0.0,
               thinking: bool | None = None) -> dict[str, Any]:
    """One judgment, in one order. Returns {"preferred": 1|2, "reasoning": ...}."""
    from lab_core.evaluation.judge import Judge

    clip = lambda s: s if len(s) <= max_chars else s[:max_chars] + "\n[...truncated for length...]"  # noqa: E731
    extra: dict[str, Any] = {"structured_outputs": {"json": _SCHEMA}}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
    if thinking is not None:
        extra["chat_template_kwargs"] = {"enable_thinking": thinking}
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=4096, extra_body=extra,
        messages=[{"role": "user", "content": PREFERENCE_PROMPT.format(instructions=instructions, draft_1=clip(first),
                                                                       draft_2=clip(second))}])
    out = Judge._parse_json(r.choices[0].message.content or "{}")
    return {"preferred": int(out.get("preferred", 0)), "reasoning": str(out.get("reasoning", ""))[:400]}


# {{docs-fragment both_orders}}
def both_orders(client, model: str, instructions: str, a: str, b: str, **kw) -> dict[str, Any]:
    """A vs B in both positions. ``score`` is A's share of the two judgments (1, 0.5 or 0), so a judge that always
    picks the first draft scores every pair 0.5: position bias shows up as ties, not as a fake preference."""
    ab = judge_pair(client, model, instructions, a, b, **kw)
    ba = judge_pair(client, model, instructions, b, a, **kw)
    wins_a = (ab["preferred"] == 1) + (ba["preferred"] == 2)
    return {"score": wins_a / 2, "first_wins": (ab["preferred"] == 1) + (ba["preferred"] == 1),
            "reasoning_ab": ab["reasoning"], "reasoning_ba": ba["reasoning"]}
# {{/docs-fragment both_orders}}


def win_rate(scores: list[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean of per-pair scores with a bootstrap 95% CI, and a two-sided sign test against 0.5 over decided pairs."""
    if not scores:
        return {}
    rng = random.Random(seed)
    boots = sorted(st.fmean(rng.choices(scores, k=len(scores))) for _ in range(n_boot))
    wins, losses = sum(s > 0.5 for s in scores), sum(s < 0.5 for s in scores)
    n = wins + losses
    p = (sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) * 2 / 2 ** n) if n else 1.0
    return {"mean": st.fmean(scores), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1],
            "pairs": len(scores), "wins": wins, "losses": losses, "ties": len(scores) - n, "sign_test_p": min(p, 1.0)}


def pairs_needed(p: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Decided pairs a sign test needs to tell a true win rate ``p`` from 50% (normal approximation)."""
    za, zb = 1.959964, {0.8: 0.841621, 0.9: 1.281552}[power]
    return math.ceil(((za * 0.5 + zb * math.sqrt(p * (1 - p))) / (p - 0.5)) ** 2)


# {{docs-fragment bradley_terry}}
def bradley_terry(outcomes: list[tuple[str, str, float]], iters: int = 500) -> dict[str, float]:
    """Fit strengths from (a, b, score_a) outcomes, score_a in [0, 1]. Returns Elo-scaled ratings, mean 1000."""
    players = sorted({x for a, b, _ in outcomes for x in (a, b)})
    s = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for p in players:
            w = sum(sc if a == p else 1 - sc for a, b, sc in outcomes if p in (a, b))
            d = sum(1 / (s[a] + s[b]) for a, b, _ in outcomes if p in (a, b))
            new[p] = max(w, 1e-3) / d if d else s[p]
        g = math.exp(st.fmean(math.log(v) for v in new.values()))
        s = {p: v / g for p, v in new.items()}
    return {p: 1000 + 400 * math.log10(v) for p, v in s.items()}
# {{/docs-fragment bradley_terry}}


def elo_with_ci(outcomes: list[tuple[str, str, float]], n_boot: int = 300, seed: int = 0) -> dict[str, dict]:
    rng = random.Random(seed)
    point = bradley_terry(outcomes)
    boots = [bradley_terry(rng.choices(outcomes, k=len(outcomes)), iters=200) for _ in range(n_boot)]
    out = {}
    for p, v in point.items():
        xs = sorted(b.get(p, v) for b in boots)
        out[p] = {"elo": v, "lo": xs[int(0.025 * n_boot)], "hi": xs[int(0.975 * n_boot) - 1]}
    return out
