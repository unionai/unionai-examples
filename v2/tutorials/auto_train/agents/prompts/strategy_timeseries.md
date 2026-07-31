**Time series strategy** (N={ts_n:,} samples, L={ts_l} timesteps/features per sample):

Work through this ladder in order — do NOT jump straight to a transformer:

1. **N < 1,000 → Classical ML on extracted features**: rolling mean/std/min/max per window + FFT coefficients → XGBoost or Random Forest.
2. **N ≥ 1,000 AND L ≤ 200 → 1D-CNN** (3–4 conv blocks + global average pool + FC head). Fast, sufficient receptive field for short windows.
3. **N ≥ 1,000 AND 200 < L ≤ 1,000**:
   - N ≥ 50k → Transformer with patching (PatchTST-style: divide series into non-overlapping patches, add positional encoding, apply attention)
   - N < 50k → 1D-CNN optionally with an LSTM head (CNN-LSTM hybrid)
4. **N ≥ 1,000 AND L > 1,000**:
   - N ≥ 50k → Transformer with patching
   - N ≥ 5k → Bidirectional LSTM/GRU
   - N < 5k → Chunk series into patches first, then CNN
5. **Multivariate (C > 1)**: prefer CNN or Transformer over plain LSTM.
6. **Streaming/online inference**: use LSTM/GRU (CNN and Transformer require the full window upfront).

**Reshape note** — X_train from the skeleton is shape (N, L) flat. Before passing to a model:
- 1D-CNN: `X.reshape(N, C, L)` → `nn.Conv1d(in_channels=C, ...)`
- LSTM/GRU: `X.reshape(N, L, C)` → `nn.LSTM(input_size=C, ...)`
- Classical ML: use flat (N, L) directly, or compute features first.
