"""Reports for the league (Elo ladder, win-rate matrix, sample game) and the champion's card."""

import viz

ROWS, COLS = 6, 7

BOT_COLOR = 8


def _players(ladder: list[dict]) -> list[str]:
    return list(ladder[-1]["ratings"]) if ladder else []


def _ordered(names: list[str]) -> list[str]:
    bots = [n for n in names if not n.startswith("gen-")]
    gens = sorted((n for n in names if n.startswith("gen-")), key=lambda n: int(n.split("-")[1]))
    return bots + gens


# {{docs-fragment ladder}}
def ladder_chart(ladder: list[dict]) -> str:
    """Each snapshot's rating as of every later generation, with the bots as reference lines."""
    xs = [row["generation"] for row in ladder]
    last = ladder[-1]["ratings"]
    gens = [n for n in _ordered(list(last)) if n.startswith("gen-")]
    series = [viz.Series("learner (newest snapshot)", [row["ratings"][f"gen-{row['generation']}"] for row in ladder], 1, xs=xs)]
    refs = [(last[b], f"{b} bot") for b in ("random", "tactical") if b in last]
    best = max(gens, key=lambda n: last[n]) if gens else ""
    return viz.line_chart(series, refs=refs, x_label="generation", x_fmt=lambda v: f"{v:.0f}",
                          y_fmt=lambda v: f"{v:.0f}", height=220) + (
        f"<p class='note'>Highest rated: <code>{viz.esc(best)}</code> at {last[best]:.0f}.</p>" if best else "")
# {{/docs-fragment ladder}}


def win_matrix(results: list[dict]) -> str:
    """Row player's score against the column player: wins plus half the draws."""
    names = _ordered(sorted({r["a"] for r in results} | {r["b"] for r in results}))
    score = {}
    for r in results:
        n = r["win"] + r["draw"] + r["loss"]
        score[(r["a"], r["b"])] = (r["win"] + 0.5 * r["draw"]) / n
        score[(r["b"], r["a"])] = (r["loss"] + 0.5 * r["draw"]) / n
    matrix = [[score.get((a, b), 0.5) for b in names] for a in names]
    # Short column labels, so 20 generations fit: R and T for the bots, the generation number otherwise.
    short = [n.split("-")[1] if n.startswith("gen-") else n[0].upper() for n in names]
    return viz.heatmap(matrix, names, short, lo=0, hi=1, fmt=lambda v: f"{v:.0%}", label_width=70, cell_h=22)


def board_html(moves: list[int], first: str, second: str, size: int = 34) -> str:
    """The final position of a game, each stone numbered by the move that placed it."""
    grid = [[None] * COLS for _ in range(ROWS)]
    heights = [0] * COLS
    for i, c in enumerate(moves):
        grid[ROWS - 1 - heights[c]][c] = i
        heights[c] += 1
    colors = ("var(--s2)", "var(--s4)")
    cells = ""
    for r in range(ROWS):
        for c in range(COLS):
            i = grid[r][c]
            fill = "var(--plane)" if i is None else colors[i % 2]
            label = "" if i is None else str(i + 1)
            cells += (f"<div style='width:{size}px;height:{size}px;border-radius:50%;background:{fill};"
                      f"display:flex;align-items:center;justify-content:center;font-size:11px;color:#fff'>{label}</div>")
    legend = viz.legend([(f"{first} (moves first)", 2), (second, 4)])
    return (legend + f"<div style='display:inline-grid;grid-template-columns:repeat({COLS},{size}px);gap:4px;"
            f"padding:8px;border-radius:10px;background:var(--s1)'>{cells}</div>")


def _training_chart(training: list[dict]) -> str:
    """Score against each opponent type over every training iteration, smoothed over 5 iterations."""
    series, xs, offset = {}, {}, 0
    for gen in training:
        for row in gen["history"]:
            for name, res in zip(gen["opponents"], row["results"]):
                key = name if not name.startswith("gen-") else "past snapshots"
                series.setdefault(key, []).append(res["win"] + 0.5 * res["draw"])
                xs.setdefault(key, []).append(offset + row["iteration"])
        offset += len(gen["history"])
    colors = {"random": 8, "tactical": 4, "self": 3, "past snapshots": 1}

    def smooth(v: list[float], k: int = 5) -> list[float]:
        return [sum(v[max(0, i - k + 1):i + 1]) / len(v[max(0, i - k + 1):i + 1]) for i in range(len(v))]

    return viz.line_chart([viz.Series(k, smooth(v), colors.get(k, 5), xs=xs[k]) for k, v in series.items()],
                          y_range=(0, 1), x_label="training iteration", y_fmt=lambda v: f"{v:.0%}", height=200)


def _tiles(ladder: list[dict], results: list[dict], generations: int) -> str:
    last = ladder[-1]["ratings"] if ladder else {}
    gens = [n for n in last if n.startswith("gen-")]
    best = max(gens, key=lambda n: last[n]) if gens else None
    vs = next((r for r in results if best and r["a"] == best and r["b"] == "tactical"), None)
    n = sum(r["win"] + r["draw"] + r["loss"] for r in results)
    return viz.tiles([
        ("Champion Elo", f"{last[best]:.0f}" if best else "—", f"<code>{viz.esc(best)}</code> · tactical bot {last.get('tactical', 0):.0f}" if best else ""),
        ("Against the tactical bot", f"{vs['win'] / (vs['win'] + vs['draw'] + vs['loss']):.0%}" if vs else "—", "wins, greedy play"),
        ("Generation", f"{len(ladder)} / {generations}"),
        ("Rated games", viz.compact(n), f"{len(results)} matches"),
    ], hero=True)


def league_live(ladder: list[dict], results: list[dict], training: list[dict], generations: int) -> str:
    return viz.page(
        "Connect Four self-play league",
        "Each generation the learner trains against the bots, its recent snapshots, and itself; then the new snapshot "
        "plays every league member, and Elo is refitted over every result.",
        _tiles(ladder, results, generations),
        viz.row(viz.panel("Elo of the newest snapshot", ladder_chart(ladder),
                          "Random is pinned at 0. 400 points means a 10-to-1 expected score."),
                viz.panel("Training: score against each opponent type", _training_chart(training))),
        viz.panel("Score matrix (row against column)", win_matrix(results), "Wins plus half the draws, greedy play after a random opening."),
    )


def league_final(ladder: list[dict], results: list[dict], training: list[dict], champion: str, game: dict) -> str:
    winner = game["winner"] or "nobody (draw)"
    return viz.page(
        "Connect Four self-play league",
        f"{len(ladder)} generations. The champion is the highest-rated snapshot, <code>{viz.esc(champion)}</code>.",
        _tiles(ladder, results, len(ladder)),
        viz.row(viz.panel("Elo of the newest snapshot", ladder_chart(ladder)),
                viz.panel(f"Sample game: {champion} against the tactical bot",
                          board_html(game["moves"], game["first"], game["second"]),
                          f"{len(game['moves'])} moves; won by {viz.esc(winner)}.")),
        viz.panel("Training: score against each opponent type", _training_chart(training)),
        viz.panel("Score matrix (row against column)", win_matrix(results)),
        viz.panel("Final ratings", viz.table(
            ["player", "Elo"], [[f"<code>{viz.esc(n)}</code>", f"{ladder[-1]['ratings'][n]:.0f}"]
                                for n in sorted(ladder[-1]["ratings"], key=lambda n: -ladder[-1]["ratings"][n])])),
    )


def training_live(generation: int, history: list[dict], names: list[str], iterations: int, games: int) -> str:
    last = history[-1]
    rows = [[f"<code>{viz.esc(n)}</code>", f"{r['win']:.0%}", f"{r['draw']:.0%}", f"{r['loss']:.0%}"]
            for n, r in zip(names, last["results"])]
    return viz.page(
        f"Generation {generation}: self-play training",
        f"{games:,} games per iteration, split evenly across {len(names)} opponents.",
        viz.tiles([("Iteration", f"{len(history)} / {iterations}"), ("Moves trained on", viz.compact(last["moves"])),
                   ("Seconds per iteration", f"{last['seconds']:.1f}"), ("Entropy", f"{last['entropy']:.2f}")], hero=True),
        viz.row(viz.panel("Latest iteration, by opponent", viz.table(["opponent", "win", "draw", "loss"], rows)),
                viz.panel("Score against each opponent", _training_chart([{"opponents": names, "history": history}]))),
    )


def card(ladder: list[dict], results: list[dict], champion: str, game: dict) -> str:
    ratings = ladder[-1]["ratings"]
    bars = [viz.Bar(n, ratings[n], 1 if n == champion else (BOT_COLOR if not n.startswith("gen-") else 7))
            for n in sorted(ratings, key=lambda n: -ratings[n])]
    return viz.card(
        f"Connect Four champion: {champion}",
        f"The highest-rated snapshot of a {len(ladder)}-generation self-play league, rated by a joint Elo fit over "
        f"{len(results)} matches.",
        ["PyTorch", "PPO", "self-play", "league", "Elo", "Connect Four"],
        _tiles(ladder, results, len(ladder)),
        viz.row(viz.panel("Ratings", viz.bar_chart(bars, label_width=90, fmt=lambda v: f"{v:.0f}")),
                viz.panel("Sample game against the tactical bot", board_html(game["moves"], game["first"], game["second"], 28))),
        viz.panel("Load it", viz.code(
            'from flyte.remote import Artifact\n'
            'champion = Artifact.get("connect4-champion")   # torch checkpoint: {"model", "optimizer", "generation"}\n'
            '# PolicyValueNet().load_state_dict(torch.load(path)["model"]), with PolicyValueNet from connect4.py')),
    )
