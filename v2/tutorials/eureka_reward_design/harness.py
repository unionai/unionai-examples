"""Train one PPO policy on MountainCar with a candidate reward function, then grade it.

This file runs inside a sandbox session, as its own process. It never imports Flyte. It reads
``reward.py`` (the code the model wrote) and ``config.json`` from its working directory, trains
with the candidate reward, and prints one JSON line on stdout:

- ``success``: the share of greedy evaluation episodes that reach the flag, the task metric;
- ``curve``: the training success rate and mean candidate reward over time;
- ``components``: the mean of every reward term the candidate reports, early and late in
  training, which is the feedback the model gets in the next generation.

The policy is graded on the task metric, never on its own reward, so a reward that is easy to
farm without reaching the flag scores zero.
"""

import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn

WD = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()


# {{docs-fragment load}}
def load_reward(path: str):
    """Load ``compute_reward(obs, action, next_obs, terminated) -> (float, dict)`` from the candidate."""
    namespace: dict = {"np": np, "math": __import__("math")}
    with open(path) as f:
        exec(compile(f.read(), "reward.py", "exec"), namespace)
    return namespace["compute_reward"]
# {{/docs-fragment load}}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 2, n_actions: int = 3, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


# MountainCar's observation is position in [-1.2, 0.6] and velocity in [-0.07, 0.07]. Scale it
# to roughly [-1, 1] for the network. The candidate reward always sees the raw values.
LOW, HIGH = np.array([-1.2, -0.07]), np.array([0.6, 0.07])


def scale(obs: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(2 * (obs - LOW) / (HIGH - LOW) - 1, dtype=torch.float32)


def evaluate(net: ActorCritic, episodes: int, seed: int) -> dict:
    """Greedy rollouts on the real task: did the car reach the flag, and how fast."""
    env = gym.make("MountainCar-v0")
    reached, steps, best_pos = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        done, t, top = False, 0, obs[0]
        while not done:
            with torch.no_grad():
                logits, _ = net(scale(obs[None]))
            obs, _, terminated, truncated, _ = env.step(int(logits.argmax()))
            t, top, done = t + 1, max(top, obs[0]), terminated or truncated
        reached.append(bool(terminated))
        steps.append(t)
        best_pos.append(float(top))
    return {"success": float(np.mean(reached)), "steps": float(np.mean(steps)), "best_pos": float(np.mean(best_pos))}


# {{docs-fragment train}}
def train(compute_reward, total_steps: int, num_envs: int, seed: int) -> dict:
    """PPO with the candidate's reward in place of the environment's."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("MountainCar-v0") for _ in range(num_envs)])
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=2.5e-4)
    horizon, epochs, minibatches, gamma, lam, clip = 128, 4, 4, 0.99, 0.95, 0.2
    obs, _ = envs.reset(seed=seed)
    curve, components, finished, reached = [], [], 0, 0
    updates = max(1, total_steps // (horizon * num_envs))
    for update in range(updates):
        b_obs, b_act, b_logp, b_rew, b_done, b_val = [], [], [], [], [], []
        terms: dict[str, list[float]] = {}
        for _ in range(horizon):
            with torch.no_grad():
                logits, value = net(scale(obs))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            next_obs, _, terminated, truncated, info = envs.step(action.numpy())
            # On an autoreset, gymnasium returns the first observation of the next episode. The
            # reward must see where this episode ended, so use the final observation instead.
            real_next = next_obs.copy()
            if "final_obs" in info:
                for i, f in enumerate(info["_final_obs"]):
                    if f:
                        real_next[i] = info["final_obs"][i]
            rewards = np.zeros(num_envs, dtype=np.float32)
            for i in range(num_envs):
                r, parts = compute_reward(obs[i].copy(), int(action[i]), real_next[i].copy(), bool(terminated[i]))
                rewards[i] = float(r)
                for k, v in (parts or {}).items():
                    terms.setdefault(str(k), []).append(float(v))
            if not np.all(np.isfinite(rewards)):
                raise ValueError("reward returned a non-finite value")
            done = np.logical_or(terminated, truncated)
            finished += int(done.sum())
            reached += int(terminated.sum())
            b_obs.append(scale(obs))
            b_act.append(action)
            b_logp.append(dist.log_prob(action))
            b_rew.append(torch.as_tensor(rewards))
            b_done.append(torch.as_tensor(done, dtype=torch.float32))
            b_val.append(value)
            obs = next_obs

        with torch.no_grad():
            _, last_val = net(scale(obs))
        adv, gae = torch.zeros(horizon, num_envs), torch.zeros(num_envs)
        for t in reversed(range(horizon)):
            nxt = last_val if t == horizon - 1 else b_val[t + 1]
            nonterminal = 1.0 - b_done[t]
            delta = b_rew[t] + gamma * nxt * nonterminal - b_val[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        ret = adv + torch.stack(b_val)
        flat = lambda xs: torch.stack(xs).reshape(horizon * num_envs, *xs[0].shape[1:])
        o, a, lp, ad, rt = flat(b_obs), flat(b_act), flat(b_logp), adv.reshape(-1), ret.reshape(-1)
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        idx = np.arange(horizon * num_envs)
        size = len(idx) // minibatches
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), size):
                mb = idx[s:s + size]
                logits, v = net(o[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(a[mb]) - lp[mb]).exp()
                pg = -torch.min(ratio * ad[mb], ratio.clamp(1 - clip, 1 + clip) * ad[mb]).mean()
                loss = pg + 0.5 * ((v - rt[mb]) ** 2).mean() - 0.01 * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        steps = (update + 1) * horizon * num_envs
        curve.append({"step": steps, "train_success": reached / max(finished, 1),
                      "reward": float(torch.stack(b_rew).mean())})
        components.append({k: float(np.mean(v)) for k, v in terms.items()})
        finished, reached = 0, 0
    return {"net": net, "curve": curve, "components": components}
# {{/docs-fragment train}}


def summarize_components(history: list[dict]) -> dict:
    """Each reward term's mean in the first and last quarter of training."""
    keys = sorted({k for h in history for k in h})
    q = max(1, len(history) // 4)
    early, late = history[:q], history[-q:]
    mean = lambda rows, k: float(np.mean([r[k] for r in rows if k in r])) if any(k in r for r in rows) else None
    return {k: {"early": mean(early, k), "late": mean(late, k)} for k in keys}


def main() -> None:
    cfg = json.load(open(os.path.join(WD, "config.json")))
    started = time.time()
    reward_fn = load_reward(os.path.join(WD, "reward.py"))
    out = train(reward_fn, cfg["total_steps"], cfg["num_envs"], cfg["seed"])
    result = {
        **evaluate(out["net"], cfg["eval_episodes"], cfg["seed"]),
        "curve": out["curve"],
        "components": summarize_components(out["components"]),
        "train_s": round(time.time() - started, 1),
    }
    if cfg.get("save_policy"):
        torch.save(out["net"].state_dict(), os.path.join(WD, "policy.pt"))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
