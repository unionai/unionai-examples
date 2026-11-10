"""Document-level filters and signatures for web text: Gopher, C4 and FineWeb heuristics, MinHash, n-grams.

Pure Python and NumPy, so the same code runs in a shard task, a notebook, or a test. Every
function works on one document at a time; the shard task loops over its documents.

    words = tokens(text)                       # normalized word tokens
    reason = heuristic_reject(text, config)    # None, or the name of the first rule that fired
    hashes = token_hashes(words)               # one uint64 per token
    sig = minhash(hashes, config)              # MinHash signature over word 5-grams
    keys = band_keys(sig, config)              # LSH band keys for near-duplicate search
    grams = ngram_hashes(hashes, 13)           # 13-gram hashes for decontamination

The rules and their thresholds follow DataTrove's implementations of the Gopher quality and
repetition filters, the document-level C4 rules, and FineWeb's own three extra rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FILTERS_VERSION = "gopher-c4-fineweb/1"


# {{docs-fragment config}}
@dataclass
class FilterConfig:
    """Every threshold the CPU stages use. It is an input of each shard task, so changing any
    value here gives new cache keys, and leaving it alone lets a rerun reuse every shard."""

    # Gopher quality (Rae et al. 2021)
    min_words: int = 50
    max_words: int = 100_000
    min_avg_word_len: float = 3.0
    max_avg_word_len: float = 10.0
    max_symbol_word_ratio: float = 0.1
    max_bullet_lines_ratio: float = 0.9
    max_ellipsis_lines_ratio: float = 0.3
    max_non_alpha_words_ratio: float = 0.8
    min_stop_words: int = 2
    # Gopher repetition
    max_dup_line_frac: float = 0.3
    max_dup_para_frac: float = 0.3
    max_dup_line_char_frac: float = 0.2
    max_dup_para_char_frac: float = 0.2
    # FineWeb's extra rules
    min_line_punct_ratio: float = 0.12
    max_short_line_ratio: float = 0.67
    max_newline_word_ratio: float = 0.3
    # MinHash: 14 bands of 8 hashes over word 5-grams, as in FineWeb
    shingle: int = 5
    bands: int = 14
    rows: int = 8
    seed: int = 1
    # Decontamination
    contam_ngram: int = 13
    version: str = FILTERS_VERSION
# {{/docs-fragment config}}


# Gopher's n-gram repetition limits: (n, max share of characters).
TOP_NGRAM_MAX_FRAC = ((2, 0.20), (3, 0.18), (4, 0.16))
DUP_NGRAM_MAX_FRAC = ((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10))


STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
END_PUNCT = (".", "!", "?", '"', "'", "”", "’", ":", ";", ")", "]")
BULLETS = ("•", "●", "○", "■", "□", "▪", "▫", "*", "-", "–", "—")
_WORD = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    """Lowercased word tokens, the unit for n-grams, MinHash and decontamination."""
    return _WORD.findall(text.lower())


def normalize(text: str) -> str:
    """The form used for exact duplicates: lowercase, whitespace collapsed."""
    return " ".join(text.lower().split())


def exact_hash(text: str) -> int:
    """A 64-bit hash of the normalized text, for exact duplicates."""
    import xxhash

    return xxhash.xxh3_64_intdigest(normalize(text).encode())


def _dup_frac(items: list[str]) -> tuple[float, float]:
    """Gopher: the share of elements, and of their characters, that repeat an earlier element."""
    if not items:
        return 0.0, 0.0
    seen: set[str] = set()
    dup, dup_chars = 0, 0
    for it in items:
        if it in seen:
            dup += 1
            dup_chars += len(it)
        seen.add(it)
    total_chars = sum(len(it) for it in items) or 1
    return dup / len(items), dup_chars / total_chars


# {{docs-fragment heuristics}}
def heuristic_reject(text: str, c: FilterConfig, words: list[str], hashes) -> str | None:
    """Return the name of the first rule the document fails, or None if it passes them all.
    ``words`` and ``hashes`` are the document's tokens and their hashes."""
    raw_words = text.split()
    n = len(raw_words)
    if n < c.min_words:
        return "gopher: too few words"
    if n > c.max_words:
        return "gopher: too many words"
    avg = sum(len(w) for w in raw_words) / n
    if not c.min_avg_word_len <= avg <= c.max_avg_word_len:
        return "gopher: mean word length"
    if (text.count("#") + text.count("...") + text.count("…")) / n > c.max_symbol_word_ratio:
        return "gopher: symbol-to-word ratio"
    lines = [ln.strip() for ln in text.split("\n")]
    nonempty = [ln for ln in lines if ln]
    if nonempty:
        if sum(ln.startswith(BULLETS) for ln in nonempty) / len(nonempty) > c.max_bullet_lines_ratio:
            return "gopher: bullet lines"
        if sum(ln.endswith(("...", "…")) for ln in nonempty) / len(nonempty) > c.max_ellipsis_lines_ratio:
            return "gopher: ellipsis lines"
    if sum(not any(ch.isalpha() for ch in w) for w in raw_words) / n > 1 - c.max_non_alpha_words_ratio:
        return "gopher: non-alphabetic words"
    if len(STOP_WORDS.intersection(words)) < c.min_stop_words:
        return "gopher: missing stop words"

    # C4: placeholder text and code
    low = text.lower()
    if "lorem ipsum" in low:
        return "c4: lorem ipsum"
    if "{" in text:
        return "c4: curly bracket"

    # Gopher repetition: whole lines and paragraphs, then n-grams
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    line_frac, line_chars = _dup_frac(nonempty)
    para_frac, para_chars = _dup_frac(paras)
    if line_frac > c.max_dup_line_frac or para_frac > c.max_dup_para_frac:
        return "gopher: duplicate lines"
    if line_chars > c.max_dup_line_char_frac or para_chars > c.max_dup_para_char_frac:
        return "gopher: duplicate line characters"
    reason = _ngram_repetition(words, hashes)
    if reason:
        return reason

    # FineWeb's three rules
    if nonempty:
        if sum(ln.endswith(END_PUNCT) for ln in nonempty) / len(nonempty) < c.min_line_punct_ratio:
            return "fineweb: lines without end punctuation"
        if sum(len(ln) <= 30 for ln in nonempty) / len(nonempty) > c.max_short_line_ratio:
            return "fineweb: short lines"
    if text.count("\n") / n > c.max_newline_word_ratio:
        return "fineweb: newline-to-word ratio"
    return None
# {{/docs-fragment heuristics}}


def _ngram_repetition(words: list[str], h) -> str | None:
    """Gopher's n-gram rules, measured in characters of the words involved."""
    import numpy as np

    lengths = np.fromiter((len(w) for w in words), dtype=np.int64, count=len(words))
    total = int(lengths.sum()) or 1
    for n, limit in TOP_NGRAM_MAX_FRAC:
        grams = ngram_hashes(h, n)
        if len(grams) == 0:
            continue
        _, first, counts = np.unique(grams, return_index=True, return_counts=True)
        top = counts.argmax()
        if counts[top] > 1 and lengths[first[top]:first[top] + n].sum() * counts[top] / total > limit:
            return f"gopher: top {n}-gram"
    for n, limit in DUP_NGRAM_MAX_FRAC:
        grams = ngram_hashes(h, n)
        if len(grams) == 0:
            continue
        repeat = np.ones(len(grams), dtype=np.int64)
        repeat[np.unique(grams, return_index=True)[1]] = 0  # first occurrences are not repeats
        covered = np.convolve(repeat, np.ones(n, dtype=np.int64))[: len(words)] > 0
        if lengths[covered].sum() / total > limit:
            return f"gopher: duplicate {n}-grams"
    return None


# ── hashing ──────────────────────────────────────────────────────────────────

_P = 0x100000001B3  # FNV prime, used as the base of the rolling n-gram hash


def token_hashes(words: list[str]):
    """One stable 64-bit hash per token (Python's hash() is salted per process, so not that)."""
    import numpy as np
    import xxhash

    return np.fromiter((xxhash.xxh3_64_intdigest(w.encode()) for w in words), dtype=np.uint64, count=len(words))


def ngram_hashes(h, n: int):
    """A 64-bit hash for every contiguous n-gram of token hashes (polynomial, wrapping)."""
    import numpy as np

    if len(h) < n:
        return np.empty(0, dtype=np.uint64)
    m = len(h) - n + 1
    out = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            out = out * np.uint64(_P) + h[j:j + m]
    return out


def _perms(c: FilterConfig):
    import numpy as np

    rng = np.random.default_rng(c.seed)
    k = c.bands * c.rows
    a = rng.integers(1, 2**63, size=k, dtype=np.uint64) | np.uint64(1)  # odd multipliers
    b = rng.integers(0, 2**63, size=k, dtype=np.uint64)
    return a, b


_PERM_CACHE: dict = {}


# {{docs-fragment minhash}}
def minhash(h, c: FilterConfig):
    """MinHash signature over word shingles: for each of bands*rows hash functions
    (multiply-shift: (a*x + b) >> 32), the minimum over the document's shingles."""
    import numpy as np

    key = (c.seed, c.bands, c.rows)
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = _perms(c)
    a, b = _PERM_CACHE[key]
    shingles = np.unique(ngram_hashes(h, c.shingle))
    with np.errstate(over="ignore"):
        return ((a[:, None] * shingles[None, :] + b[:, None]) >> np.uint64(32)).min(axis=1)


def band_keys(sig, c: FilterConfig):
    """One 64-bit key per band of ``rows`` signature values. Two documents that share any band
    key are near-duplicate candidates. With 14 bands of 8, a pair with Jaccard similarity s
    becomes a candidate with probability 1 - (1 - s**8)**14: about 50% at s = 0.69, 77% at
    0.75, 99% at 0.85, and about 5% at 0.5."""
    import numpy as np

    rows = sig.reshape(c.bands, c.rows)
    key = np.arange(c.bands, dtype=np.uint64)  # different bands never collide
    with np.errstate(over="ignore"):
        for j in range(c.rows):
            key = key * np.uint64(_P) + rows[:, j]
    return key
# {{/docs-fragment minhash}}
