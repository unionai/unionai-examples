"""gpt-oss attention for training: exact attention sinks and sliding windows with flex attention's memory profile.

gpt-oss adds a learned per-head "sink" logit to every softmax row (attention mass that goes nowhere), and every other
layer attends over a 128-token sliding window. Transformers' eager implementation does both but materializes the
full n x n score matrix, which doesn't fit at agent-trajectory lengths. This implementation runs flex attention with
one extra key in front of the sequence whose score is exactly the sink:

    q' = [q, 1]        k'_sink = [0, sink / scale]        k'_j = [k_j, 0]        v'_sink = 0

so ``scale * q' . k'_sink = sink`` for every query, the sink takes its share of the softmax, and contributes nothing
to the output. The mask lets every query see the sink key, and otherwise is causal (plus the window on sliding layers).

Register it before loading the model, then load with ``attn_implementation="sink_flex"``.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

_HEAD_PAD = 128  # pad q/k/v head dims to a size flex attention's kernels like


@functools.lru_cache(maxsize=64)
def _block_mask(n: int, window: int | None, device: str):
    from torch.nn.attention.flex_attention import create_block_mask

    def mask(b, h, q, kv):
        j = kv - 1  # key index in the original sequence; kv == 0 is the sink
        ok = (j <= q) & (kv > 0)
        if window is not None:
            ok = ok & (q - j < window)
        return ok | (kv == 0)

    return create_block_mask(mask, B=None, H=None, Q_LEN=n, KV_LEN=n + 1, device=device)


@functools.lru_cache(maxsize=1)
def _flex():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False)


def sink_flex_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None,
                        s_aux=None, **kwargs):
    """Same signature as transformers' attention interfaces; returns (output [b, n, h, d], None)."""
    b, h, n, d = query.shape
    kvh = key.shape[1]
    sinks = (s_aux if s_aux is not None else module.sinks).to(query.dtype)  # (h,)
    rep = h // kvh
    key = key.repeat_interleave(rep, dim=1)  # the sink differs per query head, so expand KV heads first
    value = value.repeat_interleave(rep, dim=1)
    pad = _HEAD_PAD - d - 1
    one = torch.ones(b, h, n, 1, dtype=query.dtype, device=query.device)
    q2 = torch.cat([query, one, query.new_zeros(b, h, n, pad)], dim=-1)
    k_sink = torch.cat([key.new_zeros(b, h, 1, d), (sinks / scaling).view(1, h, 1, 1).expand(b, h, 1, 1),
                        key.new_zeros(b, h, 1, pad)], dim=-1)
    k2 = torch.cat([k_sink, torch.cat([key, key.new_zeros(b, h, n, 1 + pad)], dim=-1)], dim=2)
    v2 = torch.cat([value.new_zeros(b, h, 1, d), value], dim=2)
    v2 = F.pad(v2, (0, _HEAD_PAD - d))
    out = _flex()(q2, k2, v2, block_mask=_block_mask(n, sliding_window, str(query.device)), scale=scaling)
    return out[..., :d].transpose(1, 2).contiguous(), None


def register() -> str:
    from transformers import AttentionInterface

    AttentionInterface.register("sink_flex", sink_flex_attention)
    return "sink_flex"
