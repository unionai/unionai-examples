"""Self-play PPO against a league of opponents, and head-to-head matches between any two players.

A training generation plays thousands of games at once against opponents sampled from the
league: earlier snapshots of the learner, the scripted bots, and the learner itself. Only the
learner's moves are trained on. Every move in a game gets the game's final outcome as its
return (+1 win, 0 draw, -1 loss), and PPO pushes the policy toward moves whose outcome beat the
value head's prediction.
"""

import time

import torch
import torch.nn.functional as F
from connect4 import Boards, PolicyValueNet


# {{docs-fragment collect}}
@torch.no_grad()
def collect(net: PolicyValueNet, opponents: list, games: int, device: str, gen: torch.Generator) -> dict:
    """Play ``games`` games in parallel, split evenly across ``opponents``, and keep the learner's moves."""
    boards = Boards(games, device)
    group = torch.arange(games, device=device) % len(opponents)  # which opponent each game faces
    learner_turn = torch.arange(games, device=device) % 2 == 0  # the learner moves first in half the games
    buf = {k: [] for k in ("board", "legal", "action", "logp", "value", "game")}
    outcome = torch.full((games,), float("nan"), device=device)
    alive = torch.ones(games, dtype=torch.bool, device=device)
    while alive.any():
        legal = boards.legal()
        logits, value = net(boards.board, legal)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.multinomial(dist.probs, 1, generator=gen).squeeze(1)
        for i, opponent in enumerate(opponents):
            mask = (group == i) & ~learner_turn & alive
            if mask.any():
                action[mask] = opponent(boards, gen)[mask]
        mine = learner_turn & alive
        for k, v in (("board", boards.board), ("legal", legal), ("action", action), ("logp", dist.log_prob(action)),
                     ("value", value), ("game", torch.arange(games, device=device))):
            buf[k].append(v[mine].clone())
        # Finished games keep playing a legal dummy move; their results are already recorded.
        action = torch.where(alive, action, legal.float().argmax(1))
        won, over = boards.step(action)
        newly = over & alive
        outcome[newly & won & learner_turn] = 1.0
        outcome[newly & won & ~learner_turn] = -1.0
        outcome[newly & ~won] = 0.0
        alive &= ~over
        learner_turn = ~learner_turn
    data = {k: torch.cat(v) for k, v in buf.items()}
    data["return"] = outcome[data.pop("game")]
    per_opponent = [outcome[group == i] for i in range(len(opponents))]
    data["results"] = [{"win": float((o == 1).float().mean()), "draw": float((o == 0).float().mean()),
                        "loss": float((o == -1).float().mean())} for o in per_opponent]
    return data
# {{/docs-fragment collect}}


# {{docs-fragment update}}
def ppo_update(net: PolicyValueNet, opt: torch.optim.Optimizer, data: dict, epochs: int = 3,
               batch: int = 4096, clip: float = 0.2) -> dict:
    """PPO-clip on Monte Carlo returns, with a value baseline and an entropy bonus."""
    n = data["action"].shape[0]
    adv = data["return"] - data["value"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
    for _ in range(epochs):
        for idx in torch.randperm(n, device=adv.device).split(batch):
            logits, value = net(data["board"][idx], data["legal"][idx])
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(data["action"][idx]) - data["logp"][idx]).exp()
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
            vl = F.mse_loss(value, data["return"][idx])
            ent = dist.entropy().mean()
            loss = pg + 0.5 * vl - 0.01 * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["policy_loss"] += float(pg.detach())
            stats["value_loss"] += float(vl.detach())
            stats["entropy"] += float(ent.detach())
            stats["updates"] += 1
    return {k: v / max(stats["updates"], 1) for k, v in stats.items() if k != "updates"}
# {{/docs-fragment update}}


def train(net: PolicyValueNet, opt: torch.optim.Optimizer, opponents: list, iterations: int, games: int,
          device: str, seed: int, on_iteration=None) -> list[dict]:
    """One generation: ``iterations`` rounds of collect-then-update against a fixed set of opponents."""
    gen = torch.Generator(device=device).manual_seed(seed)
    history = []
    for it in range(iterations):
        t0 = time.time()
        net.eval()
        data = collect(net, opponents, games, device, gen)
        net.train()
        stats = ppo_update(net, opt, data)
        row = {"iteration": it, "moves": int(data["action"].shape[0]), "results": data["results"],
               "seconds": round(time.time() - t0, 2), **stats}
        history.append(row)
        if on_iteration:
            on_iteration(row)
    return history


# {{docs-fragment match}}
@torch.no_grad()
def match(player_a, player_b, games: int, device: str = "cpu", seed: int = 0, opening: int = 4) -> dict:
    """Play ``games`` games between two players, alternating who moves first.

    The first ``opening`` moves of every game are random, so a pair of greedy players does not
    replay the same game every time. Returns A's wins, draws, and losses.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    boards = Boards(games, device)
    a_turn = torch.arange(games, device=device) % 2 == 0
    alive = torch.ones(games, dtype=torch.bool, device=device)
    result = torch.zeros(games, device=device)
    while alive.any():
        legal = boards.legal()
        if int(boards.moves.max()) < opening:
            action = torch.multinomial(legal.float(), 1, generator=gen).squeeze(1)
        else:
            action = torch.where(a_turn, player_a(boards, gen), player_b(boards, gen))
        action = torch.where(alive & legal.gather(1, action[:, None]).squeeze(1), action, legal.float().argmax(1))
        won, over = boards.step(action)
        newly = over & alive
        result[newly & won] = torch.where(a_turn[newly & won], 1.0, -1.0)
        alive &= ~over
        a_turn = ~a_turn
    return {"win": int((result == 1).sum()), "draw": int((result == 0).sum()), "loss": int((result == -1).sum())}
# {{/docs-fragment match}}
