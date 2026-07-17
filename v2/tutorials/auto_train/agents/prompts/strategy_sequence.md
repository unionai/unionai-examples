**Sequence strategy** — two sub-types; pick the right ladder based on the sequence type:

---

### Biological sequences (DNA, RNA, protein — characters are ACGT / amino acids)

Work up this ladder, but skip straight to the tier specified in **"Baseline starting point"**:

1. **k-mer TF-IDF** (ngram_range=(3,6)) → LightGBM: fast baseline for N<5k or short fixed-length sequences.
2. **1D-CNN on one-hot**: only if k-mer clearly underperforms and N<5k.
3. **Frozen domain-specific language model + linear probe** (N≥5k — start here if N≥5k):
   - DNA/nucleotide → `zhihan1996/DNABERT-2-117M` or `InstaDeepAI/nucleotide-transformer-v2-100m`
   - Protein → `facebook/esm2_t6_8M_UR50D`
   - Extract CLS or mean-pool embeddings at FP16 with `torch.no_grad()`, cache to disk, train a linear head.
   - **Never use a general NLP model (DistilBERT, RoBERTa) on biological sequences** — they have no biological pretraining and will produce random embeddings.
4. **Two-phase fine-tuning** (frozen probe plateau AND N≥5k): `backbone.train().float()`, backbone LR=1e-5 / head LR=1e-4, batch=16–32.

---

### Natural language / NLP text sequences (human-readable text — product descriptions, reviews, reports, etc.)

1. **TF-IDF + LightGBM**: reasonable for N<5k; use `analyzer='word'` for normal text, `analyzer='char_wb'` for noisy/short text.
2. **Frozen pre-trained text transformer + linear probe** (N≥5k — start here if N≥5k):
   - General text → `distilbert-base-uncased` or `roberta-base`
   - Domain-specific text → check for a domain-pretrained BERT (e.g. `ProsusAI/finbert` for finance, `allenai/scibert_scivocab_uncased` for science)
   - Extract CLS embeddings at FP16 with `torch.no_grad()`, max_length=128–256, cache to disk, train a linear head.
3. **Full fine-tuning** (frozen probe plateau AND N≥5k): unfreeze all layers, LR=2e-5, warmup 10%, cosine decay.

---

**Key rule**: never use a text NLP model (DistilBERT, BERT, RoBERTa) on biological sequences, and never use a biological model (DNABERT-2, ESM-2) on natural language text. The pretraining domain must match the data.
