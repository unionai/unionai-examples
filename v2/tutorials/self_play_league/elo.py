"""Elo ratings from head-to-head results, fitted jointly with a Bradley-Terry model."""

import math


# {{docs-fragment fit}}
def fit_elo(results: list[dict], anchor: str = "random", anchor_rating: float = 0.0, iters: int = 2000) -> dict[str, float]:
    """Ratings that best explain every match result at once.

    Each result is ``{"a", "b", "win", "draw", "loss"}`` from A's side, and a draw counts as half
    a win for each player. Every pairing also gets one virtual draw, so a player who wins every
    game gets a large rating instead of an infinite one. Unlike updating Elo one game at a time,
    a joint fit does not depend on the order the matches finished in, so a re-run of the same
    matches gives the same ladder.
    """
    players = sorted({r["a"] for r in results} | {r["b"] for r in results})
    rating = {p: 0.0 for p in players}
    lr = 20.0
    for _ in range(iters):
        grad = {p: 0.0 for p in players}
        for r in results:
            n = r["win"] + r["draw"] + r["loss"] + 1
            expected = 1 / (1 + 10 ** ((rating[r["b"]] - rating[r["a"]]) / 400))
            score = (r["win"] + 0.5 * r["draw"] + 0.5) / n
            grad[r["a"]] += n * (score - expected)
            grad[r["b"]] -= n * (score - expected)
        total = sum(r["win"] + r["draw"] + r["loss"] + 1 for r in results) / max(len(players), 1)
        for p in players:
            rating[p] += lr * grad[p] / max(total, 1)
    shift = anchor_rating - rating.get(anchor, 0.0)
    return {p: round(v + shift, 1) for p, v in rating.items()}
# {{/docs-fragment fit}}


def expected_score(ra: float, rb: float) -> float:
    return 1 / (1 + math.pow(10, (rb - ra) / 400))
