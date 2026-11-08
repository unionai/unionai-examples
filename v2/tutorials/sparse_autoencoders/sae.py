"""TopK sparse autoencoders in plain PyTorch: the model, its training loop, and its metrics.

This file never imports Flyte. The pipeline imports it inside its GPU tasks.

A sparse autoencoder (SAE) rewrites one activation vector of a language model as a sparse sum
of learned directions, its features. A TopK SAE (Gao et al., 2024) keeps exactly the k largest
encoder pre-activations per token and zeroes the rest, so the sparsity is set directly instead
of being tuned through an L1 penalty:

    z     = ReLU(TopK((x - b_dec) @ W_enc + b_enc))
    x_hat = z @ W_dec + b_dec
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# {{docs-fragment model}}
class TopKSAE(nn.Module):
    def __init__(self, d_model: int, width: int, k: int):
        super().__init__()
        self.d_model, self.width, self.k = d_model, width, k
        w_dec = torch.randn(width, d_model)
        w_dec /= w_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w_dec)  # one unit-norm direction per feature
        self.W_enc = nn.Parameter(w_dec.T.clone())  # tied at init, trained separately
        self.b_enc = nn.Parameter(torch.zeros(width))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The k active features per token: their indices, their values, and every pre-activation."""
        pre = self.pre_acts(x)
        vals, idx = pre.topk(self.k, dim=-1)
        return idx, F.relu(vals), pre

    def decode(self, idx: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        # A sum of k decoder rows per token, without materializing the width-sized code.
        return F.embedding_bag(idx, self.W_dec, per_sample_weights=vals, mode="sum") + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx, vals, _ = self.encode(x)
        return self.decode(idx, vals)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
# {{/docs-fragment model}}


def aux_loss(sae: TopKSAE, pre: torch.Tensor, err: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
    """AuxK: the dead features' top pre-activations must explain what the live ones missed.

    This gives dead features a gradient, so fewer of them stay dead (Gao et al., section 2.4).
    """
    n_dead = int(dead.sum())
    if n_dead == 0:
        return pre.new_zeros(())
    k_aux = min(sae.d_model // 2, n_dead)
    vals, idx = pre.masked_fill(~dead, -torch.inf).topk(k_aux, dim=-1)
    err_hat = F.embedding_bag(idx, sae.W_dec, per_sample_weights=F.relu(vals), mode="sum")
    return (err - err_hat).pow(2).sum(-1).mean() / err.pow(2).sum(-1).mean().clamp_min(1e-8)


# {{docs-fragment train}}
def train(
    acts: torch.Tensor,  # [sequences, positions, d_model] float16, on the GPU
    width: int,
    ks: list[int],
    steps: int,
    batch_size: int = 4096,
    lr: float | None = None,
    seed: int = 0,
    log_every: int = 250,
    on_log=None,
) -> tuple[list[TopKSAE], float, list[dict]]:
    """Train one TopK SAE per k on the same batches. Position 0 is skipped (see ``token_index``)."""
    torch.manual_seed(seed)
    device = acts.device
    n_seq, ctx, d = acts.shape
    scale = activation_scale(acts)
    # Gao et al. scale the learning rate as 1/sqrt(width).
    lr = lr or 2e-4 * math.sqrt(16384 / width)
    saes = [TopKSAE(d, width, k).to(device) for k in ks]
    sample = gather(acts, token_index(n_seq, ctx, 32_768, device), scale)
    for sae in saes:
        sae.b_dec.data = sample.median(dim=0).values  # a robust center for the decoder bias
    opts = [torch.optim.Adam(s.parameters(), lr=lr, betas=(0.9, 0.999)) for s in saes]
    decay_from = int(steps * 0.8)
    sched = lambda step: min(1.0, (step + 1) / 100) * min(1.0, (steps - step) / max(steps - decay_from, 1))
    last_fired = [torch.zeros(width, dtype=torch.long, device=device) for _ in saes]
    dead_after = max(1, 1_000_000 // batch_size)  # a feature is dead after 1M tokens without firing
    curve: list[dict] = []
    for step in range(steps):
        x = gather(acts, token_index(n_seq, ctx, batch_size, device), scale)
        point = {"step": step + 1}
        for i, (sae, opt) in enumerate(zip(saes, opts)):
            for g in opt.param_groups:
                g["lr"] = lr * sched(step)
            idx, vals, pre = sae.encode(x)
            x_hat = sae.decode(idx, vals)
            err = x - x_hat
            nmse = err.pow(2).sum(-1).mean() / (x - x.mean(0)).pow(2).sum(-1).mean()
            dead = (step - last_fired[i]) > dead_after
            loss = nmse + aux_loss(sae, pre, err.detach(), dead) / 32
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            last_fired[i][idx[vals > 0]] = step
            point[f"k{sae.k}"] = nmse.item()
            point[f"dead_k{sae.k}"] = float(dead.float().mean())
        if step % log_every == 0 or step == steps - 1:
            curve.append(point)
            if on_log:
                on_log(point)
    return saes, scale, curve
# {{/docs-fragment train}}


def token_index(n_seq: int, ctx: int, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (sequence, position) pairs, never position 0.

    The first token of every context carries a residual norm many times larger than any other
    (an attention sink), so SAE work on Pythia conventionally leaves it out.
    """
    return (torch.randint(0, n_seq, (n,), device=device), torch.randint(1, ctx, (n,), device=device))


def gather(acts: torch.Tensor, index: tuple[torch.Tensor, torch.Tensor], scale: float) -> torch.Tensor:
    return acts[index].float() * scale


def activation_scale(acts: torch.Tensor) -> float:
    """A constant that makes the mean squared norm of an activation equal d_model (as in Gemma Scope)."""
    x = acts[token_index(acts.shape[0], acts.shape[1], 65_536, acts.device)].float()
    return float(math.sqrt(acts.shape[-1] / x.pow(2).sum(-1).mean()))


# {{docs-fragment metrics}}
@torch.no_grad()
def reconstruction_metrics(sae: TopKSAE, acts: torch.Tensor, scale: float, chunk: int = 128) -> dict:
    """Fraction of variance explained, L0, dead features, and each feature's firing rate, on held-out tokens."""
    x_all = acts[:, 1:]
    mean = x_all.reshape(-1, x_all.shape[-1]).float().mean(0) * scale
    sse = sst = l0 = 0.0
    fires = torch.zeros(sae.width, device=acts.device)
    n = 0
    for s in range(0, x_all.shape[0], chunk):
        x = x_all[s:s + chunk].reshape(-1, x_all.shape[-1]).float() * scale
        idx, vals, _ = sae.encode(x)
        x_hat = sae.decode(idx, vals)
        sse += float((x - x_hat).pow(2).sum())
        sst += float((x - mean).pow(2).sum())
        on = vals > 0
        l0 += float(on.sum())
        fires.index_add_(0, idx[on], torch.ones(int(on.sum()), device=acts.device))
        n += x.shape[0]
    density = (fires / n).cpu().numpy()
    return {"fve": 1 - sse / sst, "l0": l0 / n, "dead_frac": float((density == 0).mean()),
            "density": density, "tokens": n}


@torch.no_grad()
def spliced_losses(model, tokens: torch.Tensor, layer: int, saes: list[TopKSAE], scale: float, batch: int = 64) -> dict:
    """Next-token loss with layer ``layer``'s output left alone, replaced by each SAE's reconstruction, or zeroed.

    Position 0 is never touched, the same as in training.
    """
    block = model.gpt_neox.layers[layer]
    state: dict = {"replace": None}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if state["replace"] is None:
            return output
        new = hidden.clone()
        new[:, 1:] = state["replace"](hidden[:, 1:])
        return (new, *output[1:]) if isinstance(output, tuple) else new

    def through(sae):
        def f(h):
            x = h.reshape(-1, h.shape[-1]).float() * scale
            return (sae(x) / scale).reshape(h.shape).to(h.dtype)
        return f

    def loss() -> float:
        total = 0.0
        for s in range(0, tokens.shape[0], batch):
            t = tokens[s:s + batch]
            total += float(model(input_ids=t, labels=t).loss) * t.shape[0]
        return total / tokens.shape[0]

    handle = block.register_forward_hook(hook)
    try:
        out = {"clean": loss()}
        state["replace"] = torch.zeros_like
        out["zero"] = loss()
        for sae in saes:
            state["replace"] = through(sae)
            out[f"k{sae.k}"] = loss()
    finally:
        handle.remove()
    return out


def loss_recovered(clean: float, spliced: float, zero: float) -> float:
    """1.0 when the reconstruction is as good as the real activations, 0.0 when it is no better than zeros."""
    return (zero - spliced) / (zero - clean)
# {{/docs-fragment metrics}}


# {{docs-fragment examples}}
@torch.no_grad()
def feature_examples(
    sae: TopKSAE,
    acts: torch.Tensor,
    tokens: np.ndarray,
    scale: float,
    decode_token,
    n_features: int = 40,
    n_top: int = 12,
    n_sampled: int = 8,
    n_detect: int = 8,
    before: int = 24,
    after: int = 8,
    seed: int = 0,
    chunk: int = 128,
) -> list[dict]:
    """For a random sample of live features: contexts to explain it from, a held-out test set, and a histogram.

    Claude explains a feature from its ``n_top`` strongest contexts plus ``n_sampled`` contexts drawn
    from the rest of its clear firings. Top contexts alone hide what a feature does at moderate
    strength, which is most of its firings (Paulo et al., 2024). The test set for each feature mixes ``n_detect`` contexts where it fires clearly (none of them
    among its top contexts) with ``n_detect`` random contexts where it does not fire at all. Autointerp scores an
    explanation by how well it predicts which is which.
    """
    rng = np.random.default_rng(seed)
    n_seq, ctx, _ = acts.shape
    # Pass 1: how many contexts each feature fires in.
    seq_hits = torch.zeros(sae.width, device=acts.device)
    for s in range(0, n_seq, chunk):
        x = acts[s:s + chunk, 1:].reshape(-1, acts.shape[-1]).float() * scale
        idx, vals, _ = sae.encode(x)
        per_seq = torch.zeros(min(chunk, n_seq - s), sae.width, device=acts.device)
        rows = torch.arange(per_seq.shape[0], device=acts.device).repeat_interleave((ctx - 1) * sae.k)
        per_seq.index_put_((rows, idx.flatten()), (vals.flatten() > 0).float(), accumulate=True)
        seq_hits += (per_seq > 0).float().sum(0)
    # Sample uniformly among features that fire in enough contexts to fill both example sets.
    enough = (seq_hits >= 2 * (n_top + n_sampled + n_detect)) & (seq_hits <= n_seq - 2 * n_detect)
    eligible = torch.nonzero(enough).flatten().cpu().numpy()
    feats = np.sort(rng.choice(eligible, size=min(n_features, len(eligible)), replace=False))
    # Pass 2: every activation of the sampled features, at every position.
    col = torch.full((sae.width,), -1, dtype=torch.long, device=acts.device)
    col[torch.as_tensor(feats, device=acts.device)] = torch.arange(len(feats), device=acts.device)
    f_acts = torch.zeros(n_seq, ctx, len(feats), dtype=torch.float16, device=acts.device)
    for s in range(0, n_seq, chunk):
        block = acts[s:s + chunk, 1:]
        x = block.reshape(-1, acts.shape[-1]).float() * scale
        idx, vals, _ = sae.encode(x)
        c = col[idx]
        keep = (c >= 0) & (vals > 0)
        rows = torch.arange(x.shape[0], device=acts.device).unsqueeze(1).expand_as(idx)[keep]
        flat = torch.zeros(x.shape[0], len(feats), dtype=torch.float16, device=acts.device)
        flat[rows, c[keep]] = vals[keep].half()
        f_acts[s:s + block.shape[0], 1:] = flat.reshape(block.shape[0], ctx - 1, len(feats))

    def window(seq: int, center: int, values: np.ndarray) -> dict:
        lo, hi = max(1, center - before), min(ctx, center + after + 1)
        return {"tokens": [decode_token(int(t)) for t in tokens[seq, lo:hi]],
                "acts": [round(float(v), 3) for v in values[seq, lo:hi]]}

    out = []
    for j, f in enumerate(feats):
        a = f_acts[:, :, j].float()
        seq_max, seq_arg = a.max(dim=1)
        order = torch.argsort(seq_max, descending=True).cpu().numpy()
        seq_max, seq_arg, a_np = seq_max.cpu().numpy(), seq_arg.cpu().numpy(), a.cpu().numpy()
        active = order[seq_max[order] > 0]
        top = active[:n_top]
        rest = active[n_top:]
        # Held-out positives: contexts outside the top where the feature still fires clearly, at
        # least a quarter of its peak. Weaker firings are near the SAE's noise floor.
        clear = rest[seq_max[rest] >= 0.25 * seq_max[top[0]]]
        pool = clear if len(clear) >= n_sampled + n_detect else rest[:4 * (n_sampled + n_detect)]
        picked = rng.choice(pool, size=min(n_sampled + n_detect, len(pool)), replace=False)
        sampled, test_on = np.sort(picked[:n_sampled]), picked[n_sampled:]
        test_on_set = [{**window(int(sq), int(seq_arg[sq]), a_np), "active": True} for sq in test_on]
        quiet = np.nonzero(seq_max == 0)[0]
        test_off = rng.choice(quiet, size=min(n_detect, len(quiet)), replace=False) if len(quiet) else []
        test_off_set = [{**window(int(sq), int(rng.integers(before + 1, ctx - after)), a_np), "active": False}
                        for sq in test_off]
        nz = a_np[a_np > 0]
        counts, edges = np.histogram(nz, bins=24, range=(0, float(nz.max())))
        out.append({
            "feature": int(f),
            "density": float(len(nz) / (n_seq * (ctx - 1))),
            "context_rate": float(len(active) / n_seq),
            "max_act": float(nz.max()),
            "hist": {"counts": counts.tolist(), "edges": [round(float(e), 3) for e in edges]},
            "top": [window(int(sq), int(seq_arg[sq]), a_np) for sq in top],
            "sampled": sorted((window(int(sq), int(seq_arg[sq]), a_np) for sq in sampled), key=lambda w: -max(w["acts"])),
            "test": test_on_set + test_off_set,
        })
    return out
# {{/docs-fragment examples}}


def density_histogram(density: np.ndarray, lo: float = -6.0, hi: float = 0.0, step: float = 0.5) -> dict:
    """How many features fire at each log10 rate, with dead features counted separately."""
    alive = density[density > 0]
    edges = np.arange(lo, hi + step, step)
    counts, _ = np.histogram(np.log10(alive), bins=edges)
    return {"edges": edges.tolist(), "counts": counts.tolist(), "dead": int((density == 0).sum())}
