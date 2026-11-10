"""The Eureka run's report, its Candidates tab, and the card for the published reward function."""

import viz

STOPPED = "critical"


def _status(r) -> str:
    if r.outcome == "trained":
        return viz.status("good" if r.success > 0 else "neutral", f"trained · {r.success:.0%}")
    return viz.status(STOPPED if r.outcome.startswith("stopped") else "serious", r.outcome)


def _progress_chart(history: list[dict], base: list | None) -> str:
    xs = [h["generation"] for h in history]
    refs = [(b.success, b.name) for b in base or []]
    return viz.line_chart(
        [viz.Series("best of generation", [h["success"] for h in history], 1, xs=xs)],
        refs=refs, y_range=(0, 1), x_label="generation", y_fmt=lambda v: f"{v:.0%}", x_fmt=lambda v: f"{v:.0f}",
        height=200, width=420,
    )


def _generation_table(results: list) -> str:
    rows = sorted(results, key=lambda r: (r.generation, -r.success, r.steps))
    return viz.table(
        ["gen", "candidate", "result", "per seed", "steps to flag"],
        [[r.generation, f"<code>{viz.esc(r.name)}</code>", _status(r),
          " / ".join(f"{s:.0%}" for s in r.per_seed) or "—",
          f"{r.steps:.0f}" if r.success else "—"] for r in rows],
    )


def _curves(results: list, top: int = 5) -> str:
    trained = sorted((r for r in results if r.curve), key=lambda r: -r.success)[:top]
    if not trained:
        return "<p class='note'>no candidate trained yet</p>"
    return viz.line_chart(
        [viz.Series(f"g{r.generation} {r.name}", [p["train_success"] for p in r.curve], i + 1,
                    xs=[p["step"] for p in r.curve]) for i, r in enumerate(trained)],
        y_range=(0, 1), x_label="environment steps", y_fmt=lambda v: f"{v:.0%}", height=220,
    )


def _tiles(results: list, history: list[dict], generations: int) -> str:
    best = max(results, key=lambda r: (r.success, -r.steps)) if results else None
    stopped = [r for r in results if r.outcome.startswith("stopped")]
    return viz.tiles([
        ("Best task metric", f"{best.success:.0%}" if best else "—", f"<code>{viz.esc(best.name)}</code>" if best else ""),
        ("Generation", f"{len(history)} / {generations}"),
        ("Rewards trained", str(sum(r.outcome == "trained" for r in results)), f"of {len(results)} written"),
        ("Stopped by the sandbox", str(len(stopped)), ", ".join(r.name for r in stopped)),
    ], hero=True)


def live(results: list, history: list[dict], base: list | None, generations: int) -> str:
    return viz.page(
        "Eureka: Claude writes MountainCar rewards",
        "Each generation, Claude writes reward functions; each trains PPO on several seeds, each in its own sandbox, "
        "and the best one's training statistics go back to Claude for the next generation.",
        _tiles(results, history, generations),
        viz.row(viz.panel("Task metric, best of each generation", _progress_chart(history, base)),
                viz.panel("Training curves, top candidates", _curves(results))),
        viz.panel("Every candidate", _generation_table(results), "The Candidates tab has each one's idea and code."),
    )


def _gif(gif: str, label: str) -> str:
    if not gif:
        return "<p class='note'>no candidate reached the flag, so there is nothing to film</p>"
    return f"<img alt='{viz.esc(label)}' src='data:image/gif;base64,{gif}' style='width:100%;max-width:420px;border-radius:8px'/>"


def _fresh_table(fresh: list, search: list) -> str:
    return viz.table(
        ["reward", "in the search", "on fresh seeds", "per fresh seed", "steps to flag"],
        [[f"<code>{viz.esc(f.name)}</code>", f"{s.success:.0%}", f"{f.success:.0%}",
          " / ".join(f"{x:.0%}" for x in f.per_seed) or "—", f"{f.steps:.0f}" if f.success else "—"]
         for f, s in zip(fresh, search)],
    )


def final(results: list, history: list[dict], base: list, winner, gif: str, fresh: list) -> str:
    bars = [viz.Bar(b.name, b.success, 8) for b in base] + [
        viz.Bar(f"gen {h['generation']}: {h['name']}", h["success"], 1) for h in history]
    return viz.page(
        "Eureka: Claude writes MountainCar rewards",
        f"{len(history)} generations, {len(results)} reward functions. Every policy trained for the same budget "
        "and was graded only on reaching the flag.",
        _tiles(results, history, len(history)),
        viz.row(
            viz.panel("Baselines against the best of each generation",
                      viz.bar_chart(bars, max_value=1.0, label_width=230, fmt=lambda v: f"{v:.0%}",
                                    legend_items=[("baseline", 8), ("Claude", 1)]),
                      "Share of greedy evaluation episodes that reach the flag, mean over seeds."),
            viz.panel(f"The winner: {winner.name}", _gif(gif, winner.name),
                      f"Generation {winner.generation} · {winner.success:.0%} of episodes reach the flag · "
                      f"{winner.steps:.0f} steps on average"),
        ),
        viz.panel("The winner and the baselines on fresh seeds", _fresh_table(fresh, [winner, *base]),
                  "The search score selected the winner, so it is biased upward. These policies were trained again "
                  "on seeds none of them had seen."),
        viz.panel("Winning reward function", viz.code(winner.code), viz.esc(winner.idea)),
        viz.panel("Training curves, top candidates", _curves(results)),
        viz.panel("Every candidate", _generation_table(results)),
    )


def candidates_tab(results: list) -> str:
    blocks = [
        viz.panel(f"Generation {r.generation}: {r.name}",
                  f"<p>{_status(r)} · {viz.esc(r.idea)}</p>" + viz.code(r.code)
                  + (f"<p class='note'>{viz.esc(r.error)}</p>" if r.error else ""))
        for r in sorted(results, key=lambda r: (r.generation, -r.success))
    ]
    return viz.page("Candidates", "Every reward function Claude wrote, and what happened to it.", *blocks)


def card(winner, base: list, history: list[dict], gif: str) -> str:
    bars = [viz.Bar(b.name, b.success, 8) for b in base] + [viz.Bar(winner.name, winner.success, 1)]
    return viz.card(
        f"MountainCar reward function: {winner.name}",
        f"Written by Claude in generation {winner.generation} of a Eureka-style search, trained in a sandbox, "
        "and selected by how often the trained policy reaches the flag.",
        ["reward function", "MountainCar-v0", "PPO", "Eureka", "unionai-sandbox"],
        viz.tiles([("Flag reached", f"{winner.success:.0%}", f"mean of {len(winner.per_seed)} seeds"),
                   ("Steps to flag", f"{winner.steps:.0f}", "of 200"),
                   ("Generations", str(len(history)))], hero=True),
        viz.row(viz.panel("Against the baselines", viz.bar_chart(bars, max_value=1.0, label_width=200, fmt=lambda v: f"{v:.0%}")),
                viz.panel("A greedy episode", _gif(gif, winner.name))),
        viz.panel("Source", viz.code(winner.code)),
        viz.panel("Load it", viz.code(
            'from flyte.remote import Artifact\n'
            'reward = Artifact.get("mountaincar-reward-function")   # a .py file with compute_reward')),
    )
