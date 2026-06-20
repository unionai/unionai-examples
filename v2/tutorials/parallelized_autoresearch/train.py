"""TinyGPT training code for the autoresearch MLE agent.

The model (causal self-attention + MLP blocks), optimizer, and training loop are
the same single-GPU recipe used in `karpathy/autoresearch
<https://github.com/karpathy/autoresearch>`_'s ``train.py``. Upstream, an agent
*rewrites this whole file* each experiment. Here we expose the same knobs through
an :class:`~autoresearch_types.ExperimentConfig` so the agent can drive
architecture / optimization search as **structured tool arguments** — which is
exactly what lets the runtime right-size compute and self-heal from OOM per call.

Metric: **val_bpb** (validation bits per byte) from :func:`prepare.evaluate_bpb`
— lower is better, and comparable across architecture changes.
"""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from autoresearch_types import ExperimentConfig, ExperimentResult

import prepare

# Context length (must match prepare.MAX_SEQ_LEN).
MAX_SEQ_LEN = prepare.MAX_SEQ_LEN


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc2(F.gelu(self.fc1(x)))
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Causal LM with the forward signature expected by ``evaluate_bpb``."""

    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(B, T)

        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unknown reduction={reduction!r}")


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_training(config: ExperimentConfig) -> ExperimentResult:
    """Train one TinyGPT variant under ``config`` and return its ``val_bpb``.

    Expects ``AUTORESEARCH_CACHE`` to point at a directory containing ``data/``
    and ``tokenizer/`` (see ``prepare.py``). Keeps a small default eval budget so
    workshop runs finish quickly; override with ``AUTORESEARCH_EVAL_TOKENS``.
    """
    os.environ.setdefault("AUTORESEARCH_EVAL_TOKENS", str(32 * MAX_SEQ_LEN))

    assert prepare.MAX_SEQ_LEN == MAX_SEQ_LEN, "Bundle MAX_SEQ_LEN must match train.py block size"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = prepare.Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    model = TinyGPT(
        vocab_size=vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=MAX_SEQ_LEN,
        dropout=config.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    train_loader = prepare.make_dataloader(
        tokenizer, config.device_batch_size, MAX_SEQ_LEN, "train", device=device
    )

    t0 = time.time()
    model.train()
    steps = 0
    while steps < config.max_steps and time.time() - t0 < config.time_budget_sec:
        x, y, _ = next(train_loader)
        x = x.to(device)
        y = y.to(device)
        loss = model(x, y, reduction="mean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1

    val_bpb = float(prepare.evaluate_bpb(model, tokenizer, config.device_batch_size, device=device))
    n_params = _count_params(model)
    model_name = f"TinyGPT-L{config.n_layer}H{config.n_head}D{config.n_embd}"

    return ExperimentResult(
        title=config.title,
        val_bpb=val_bpb,
        model_name=model_name,
        n_params=n_params,
        steps=steps,
        device=device.type,
        config=config,
        notes=(
            f"val_bpb (lower better); device={device.type}; steps={steps}/{config.max_steps}; "
            f"params={n_params:,}; batch={config.device_batch_size}; lr={config.learning_rate}; "
            f"time_budget_s={config.time_budget_sec}"
        ),
    )


if __name__ == "__main__":
    # Local smoke test against ~/.cache/autoresearch (run prepare.py first).
    result = run_training(ExperimentConfig(title="baseline-smoke-test", time_budget_sec=20))
    print(result)
