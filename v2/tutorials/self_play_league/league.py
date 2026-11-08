# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
# ]
# main = "league"
# params = "generations=2 iterations=4"
# ///
"""Self-play league for Connect Four: train against past selves, rate everyone with Elo.

    for each generation:
        train_generation (warm GPU)   PPO against the league: bots, snapshots, and itself
        N x play_match (CPU pool)     the new snapshot against every member of the league
        fit Elo over every result so far
    publish the highest-rated snapshot as an artifact; app.py serves it so you can play it

A learner that only plays its current self can chase its own tail: it learns to beat today's
habits and forgets how to beat yesterday's. A league keeps every past snapshot around as an
opponent, which is how AlphaStar and OpenAI Five kept their agents from cycling. The Elo ladder
over generations shows whether each snapshot is really stronger than the ones before it.

    python league.py              # 20 generations, about 25 minutes
    python league.py --smoke      # 2 short generations
"""

import asyncio
import json
import sys
import tempfile
import time
from dataclasses import dataclass

import elo
import flyte
import flyte.artifacts as artifacts
import flyte.report
import reports
from flyte.io import File

# {{docs-fragment envs}}
# unionai-reuse provides the bridge that reusable environments need.
image = flyte.Image.from_debian_base(name="connect4-league").with_pip_packages("torch==2.9.1", "numpy<3", "unionai-reuse==0.2.1")
# The game and the trainer are imported inside the tasks, so the code bundle lists them.
CODE = ["connect4.py", "selfplay.py"]

# The learner keeps a warm CUDA context and its optimizer between generations: one reusable
# GPU replica serves every train_generation call of a run.
train_env = flyte.TaskEnvironment(
    name="c4-train",
    image=image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="L4:1"),
    reusable=flyte.ReusePolicy(replicas=1, concurrency=1, idle_ttl=900),
    cache="auto",
    include=CODE,
)

# Matches are CPU work, and a new snapshot plays every member of the league at once. A warm
# pool of CPU replicas takes the fan-out without starting a pod per match.
match_env = flyte.TaskEnvironment(
    name="c4-match",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    reusable=flyte.ReusePolicy(replicas=(1, 4), concurrency=4, idle_ttl=600, scaledown_ttl=120),
    cache="auto",
    include=CODE,
)

driver_env = flyte.TaskEnvironment(
    name="c4-league-driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[train_env, match_env],
)
# {{/docs-fragment envs}}


@dataclass
class Player:
    """A league member: a scripted bot (``policy`` is None) or a snapshot of the learner."""

    name: str
    policy: File | None = None


# Torch is imported inside the tasks, so launching a run only needs flyte installed locally.
def load_net(path: str, device: str = "cpu"):
    import torch
    from connect4 import PolicyValueNet

    net = PolicyValueNet()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
    return net.to(device).eval()


async def as_bot(player: Player, device: str = "cpu", greedy: bool = False):
    from connect4 import BOTS, net_bot

    if player.policy is None:
        return BOTS[player.name]
    return net_bot(load_net(await player.policy.download(), device), greedy=greedy)


# {{docs-fragment train}}
@train_env.task(report=True)
async def train_generation(
    generation: int, learner: File | None, opponents: list[Player], iterations: int = 25, games: int = 2048,
) -> tuple[File, str]:
    """Train the learner for one generation against the given league opponents and itself, and snapshot it."""
    import torch
    from connect4 import PolicyValueNet, net_bot
    from selfplay import train

    device = "cuda"
    torch.manual_seed(generation)
    net = PolicyValueNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    if learner is not None:
        state = torch.load(await learner.download(), map_location=device, weights_only=True)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])

    # Games are split evenly across the opponents and the current learner itself.
    names = [p.name for p in opponents] + ["self"]
    players = [await as_bot(p, device) for p in opponents] + [net_bot(net)]

    def live(row: dict) -> None:
        history.append(row)
        flyte.report.replace(reports.training_live(generation, history, names, iterations, games))
        flyte.report.flush()

    history: list[dict] = []
    train(net, opt, players, iterations, games, device, seed=generation, on_iteration=live)

    path = f"{tempfile.mkdtemp()}/gen-{generation}.pt"
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "generation": generation}, path)
    return await File.from_local(path), json.dumps({"opponents": names, "history": history})
# {{/docs-fragment train}}


# {{docs-fragment match}}
@match_env.task
async def play_match(a: Player, b: Player, games: int = 400, seed: int = 0) -> str:
    """Play ``games`` games between two league members, each side greedy after a random opening."""
    import torch
    from selfplay import match

    torch.set_num_threads(2)
    t0 = time.time()
    result = match(await as_bot(a, greedy=True), await as_bot(b, greedy=True), games, seed=seed)
    print(f"{a.name} vs {b.name}: {result} in {time.time() - t0:.1f}s")
    return json.dumps({"a": a.name, "b": b.name, **result})
# {{/docs-fragment match}}


@match_env.task
async def sample_game(a: Player, b: Player, seed: int = 7) -> str:
    """One game between two players, move by move, for the report and the card."""
    import torch
    from connect4 import Boards

    gen = torch.Generator().manual_seed(seed)
    boards, players = Boards(1), [await as_bot(a, greedy=True), await as_bot(b, greedy=True)]
    moves, winner = [], None
    while True:
        side = len(moves) % 2
        col = int(players[side](boards, gen)[0])
        moves.append(col)
        won, over = boards.step(torch.tensor([col]))
        if bool(over[0]):
            winner = [a.name, b.name][side] if bool(won[0]) else None
            break
    return json.dumps({"first": a.name, "second": b.name, "moves": moves, "winner": winner})


# {{docs-fragment opponents}}
def pick_opponents(members: list[Player], results: list[dict], top: int = 3) -> list[Player]:
    """The bots, plus the strongest past snapshots by Elo.

    Training only against recent snapshots lets the learner forget how to beat older styles of
    play. Training against the strongest members keeps the pressure where the learner is
    weakest, a simple form of the prioritized league play behind AlphaStar.
    """
    ratings = elo.fit_elo(results)
    bots = [m for m in members if m.policy is None]
    snapshots = sorted((m for m in members if m.policy is not None), key=lambda m: -ratings.get(m.name, 0))
    return bots + snapshots[:top]
# {{/docs-fragment opponents}}


# {{docs-fragment driver}}
@driver_env.task(report=True, produces_artifacts=True, retries=2)
async def league(generations: int = 20, iterations: int = 25, games: int = 2048, match_games: int = 400) -> File:
    """Run the self-play league, keep the Elo ladder up to date, and publish the champion."""
    members = [Player("random"), Player("tactical")]
    results = [json.loads(await play_match(members[1], members[0], match_games))]
    ladder: list[dict] = []  # Elo of every member after each generation
    training: list[dict] = []
    learner: File | None = None

    for g in range(1, generations + 1):
        with flyte.group(f"generation-{g}"):
            learner, log = await train_generation(g, learner, pick_opponents(members, results), iterations, games)
            snapshot = Player(f"gen-{g}", learner)
            # The new snapshot plays every existing member at once, on the CPU pool.
            new = await asyncio.gather(*[play_match(snapshot, m, match_games, seed=g) for m in members])
        members.append(snapshot)
        results += [json.loads(r) for r in new]
        training.append({"generation": g, **json.loads(log)})
        ratings = elo.fit_elo(results)
        ladder.append({"generation": g, "ratings": ratings})
        await flyte.report.replace.aio(reports.league_live(ladder, results, training, generations), do_flush=True)

    ratings = ladder[-1]["ratings"]
    champion = max((m for m in members if m.policy is not None), key=lambda m: ratings[m.name])
    game = json.loads(await sample_game(champion, Player("tactical")))
    await flyte.report.replace.aio(reports.league_final(ladder, results, training, champion.name, game), do_flush=True)

    card = await artifacts.Card.create_from.aio(
        content=reports.card(ladder, results, champion.name, game), format="html", card_type="model")
    vs_tactical = next(r for r in results if r["a"] == champion.name and r["b"] == "tactical")
    metadata = artifacts.Metadata.create_model_metadata(
        name="connect4-champion",
        description=f"Connect Four self-play champion {champion.name}: Elo {ratings[champion.name]:.0f}, "
                    f"{vs_tactical['win'] / match_games:.0%} wins against the tactical bot",
        framework="PyTorch",
        model_type="policy-value network",
        architecture="residual CNN, 4 blocks x 64 channels",
        task="Connect Four",
        serial_format="pt",
        card=card,
        attrs={"generation": champion.name, "elo": f"{ratings[champion.name]:.0f}",
               "win_vs_tactical": f"{vs_tactical['win'] / match_games:.3f}"},
    )
    return artifacts.new(champion.policy, metadata)
# {{/docs-fragment driver}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    if "--smoke" in sys.argv:
        run = flyte.run(league, generations=2, iterations=4, games=512, match_games=100)
    else:
        run = flyte.run(league)
    print(run.url)
# {{/docs-fragment run}}
