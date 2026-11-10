"""The distilled quality classifier: a frozen text encoder and a small regression head.

This is the FineWeb-Edu classifier's recipe at a smaller scale: Snowflake's arctic-embed-m
encoder stays frozen, and only a head on its [CLS] vector (a dense layer, tanh, and one linear
output, the same shape as BERT's pooler plus a regression layer) learns to predict the LLM's
0-5 score. Because the encoder is frozen, each document is encoded once and the head trains in
seconds.
"""

from __future__ import annotations

import time

# Bump when the encoder, the head, or its training changes. Tasks that use this module salt
# their cache with it, because cache="auto" hashes only a task's own source.
VERSION = "arctic-embed-m/balanced-head/2"
ENCODER = "Snowflake/snowflake-arctic-embed-m"
PUBLIC_CLASSIFIER = "HuggingFaceFW/fineweb-edu-classifier"
MAX_TOKENS = 512

_models: dict = {}


def _load(name: str, head: bool = False):
    """Load a tokenizer and model onto the GPU once per container."""
    if name not in _models:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        cls = AutoModelForSequenceClassification if head else AutoModel
        started = time.monotonic()
        tok = AutoTokenizer.from_pretrained(name)
        model = cls.from_pretrained(name, dtype=torch.float16).cuda().eval()
        _models[name] = (tok, model, time.monotonic() - started)
    return _models[name]


# {{docs-fragment encode}}
def _batches(tok, texts: list[str], batch_size: int):
    """Tokenize, then batch documents of similar length together so padding stays small."""
    enc = tok(texts, truncation=True, max_length=MAX_TOKENS)
    order = sorted(range(len(texts)), key=lambda i: len(enc["input_ids"][i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        yield idx, tok.pad({"input_ids": [enc["input_ids"][i] for i in idx],
                            "attention_mask": [enc["attention_mask"][i] for i in idx]}, return_tensors="pt")


def embed(texts: list[str], batch_size: int = 128):
    """The encoder's last-layer [CLS] vector for every text, in input order (float32, N x 768)."""
    import numpy as np
    import torch

    tok, model, _ = _load(ENCODER)
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for idx, batch in _batches(tok, texts, batch_size):
            hidden = model(**{k: v.cuda() for k, v in batch.items()}).last_hidden_state[:, 0]
            out[idx] = hidden.float().cpu().numpy()
    return out
# {{/docs-fragment encode}}


def public_scores(texts: list[str], batch_size: int = 128):
    """Scores from the public FineWeb-Edu classifier, distilled from Llama-3-70B labels."""
    import numpy as np
    import torch

    tok, model, _ = _load(PUBLIC_CLASSIFIER, head=True)
    out = np.zeros(len(texts), dtype=np.float32)
    with torch.inference_mode():
        for idx, batch in _batches(tok, texts, batch_size):
            out[idx] = model(**{k: v.cuda() for k, v in batch.items()}).logits.squeeze(-1).float().cpu().numpy()
    return out


def warm_up(name: str = ENCODER) -> float:
    """Load a model if this container has not yet. Returns the seconds this call spent loading."""
    if name in _models:
        return 0.0
    return _load(name, head=name == PUBLIC_CLASSIFIER)[2]


# {{docs-fragment head}}
def make_head(dim: int = 768):
    from torch import nn

    return nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))


def train_head(x, y, epochs: int = 100, lr: float = 3e-4, batch_size: int = 256, seed: int = 0):
    """Fit the head to the LLM scores with a class-balanced MSE.

    Web text is mostly scores 0-2, so a plain MSE learns to predict "about 1" for everything and
    never reaches the threshold. Weighting each document by the inverse frequency of its score
    (capped at 20x) makes the rare high scores count. 10% of the rows pick the best epoch."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    counts = np.bincount(np.rint(y).astype(int), minlength=6).astype(float)
    weight = np.minimum(counts.max() / np.maximum(counts, 1), 20.0)[np.rint(y).astype(int)]
    perm = rng.permutation(len(x))
    n_val = max(len(x) // 10, 1)
    val, tr = perm[:n_val], perm[n_val:]

    def t(a, idx):
        return torch.tensor(a[idx], dtype=torch.float32).cuda()

    xt, yt, wt = t(x, tr), t(y, tr), t(weight, tr)
    xv, yv, wv = t(x, val), t(y, val), t(weight, val)
    head = make_head(x.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    best, best_state, curve = float("inf"), None, []
    for epoch in range(epochs):
        head.train()
        for s in torch.randperm(len(xt)).split(batch_size):
            loss = (wt[s] * (head(xt[s]).squeeze(-1) - yt[s]) ** 2).sum() / wt[s].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = ((wv * (head(xv).squeeze(-1) - yv) ** 2).sum() / wv.sum()).item()
        curve.append(v)
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head.cpu(), curve
# {{/docs-fragment head}}


def predict(head, x):
    import torch

    head = head.cuda().eval()
    with torch.no_grad():
        return head(torch.tensor(x).cuda()).squeeze(-1).float().cpu().numpy()


# {{docs-fragment agreement}}
def agreement(reference, predicted, threshold: float = 3.0) -> dict:
    """How well ``predicted`` scores track ``reference`` scores (both 0-5, either may be float)."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)

    def ranks(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):  # average ranks over ties (scores are mostly integers)
            tie = v == val
            r[tie] = r[tie].mean()
        return r

    ri, pi = np.clip(np.rint(ref), 0, 5).astype(int), np.clip(np.rint(pred), 0, 5).astype(int)
    tp = int(((ref >= threshold) & (pred >= threshold)).sum())
    fp = int(((ref < threshold) & (pred >= threshold)).sum())
    fn = int(((ref >= threshold) & (pred < threshold)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    confusion = np.zeros((6, 6), dtype=int)
    np.add.at(confusion, (ri, pi), 1)
    return {
        "n": int(len(ref)),
        "spearman": float(np.corrcoef(ranks(ref), ranks(pred))[0, 1]),
        "pearson": float(np.corrcoef(ref, pred)[0, 1]),
        "mae": float(np.abs(ref - pred).mean()),
        "exact": float((ri == pi).mean()),
        "within_one": float((np.abs(ri - pi) <= 1).mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion": confusion.tolist(),  # rows: reference score, columns: predicted score
    }
# {{/docs-fragment agreement}}
