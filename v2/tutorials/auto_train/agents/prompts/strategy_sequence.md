**Sequence strategy** — work through this decision ladder in order, do NOT jump straight to a transformer:

1. **Positional tabular features** (if sequence is fixed-length and short, e.g. ≤200 chars):
   Split each character position into a separate categorical feature column → LightGBM/XGBoost.
   e.g. for a 60-char DNA sequence: `df[[f'pos_{{i}}' for i in range(60)]] = df['seq'].apply(list, result_type='expand')`
   Try this FIRST — a 0.90+ result here means you do not need a transformer.

2. **k-mer frequency features** (works for any sequence length):
   `CountVectorizer(analyzer='char', ngram_range=(3, 6))` → LightGBM or logistic regression.
   Fast, no GPU needed.

3. **CNN on one-hot encoded sequences** (if the above plateau):
   One-hot encode each position → 1D CNN. Captures local motifs without a pretrained model.

4. **Frozen domain-specific transformer + linear probe** (only when N≥5k and CNN also plateaus):
   Extract CLS/mean-pool embeddings once at FP16 with `torch.no_grad()`, cache to disk, train a linear head.
   Choose a model pretrained on the same domain (DNA, protein, text, etc.).

5. **Two-phase fine-tuning** (only when frozen probe plateaus AND N≥5k):
   Phase 1: frozen backbone, cache embeddings, train head (3–5 epochs).
   Phase 2: `backbone.train().float()`, new DataLoader of raw sequences, backbone LR=1e-5 / head LR=1e-4, batch=16–32.

Start at step 1. Skip to a later step only if the current approach has clearly plateaued.
