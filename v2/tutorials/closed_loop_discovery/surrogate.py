"""The surrogate and the acquisition function.

The surrogate is a bootstrap ensemble of small MLPs on Morgan fingerprints. Each member is
trained on a resample of the measured molecules, so the members agree where the data is dense
and disagree where it is thin: the spread of their predictions is the model's uncertainty.

Acquisition is a lower confidence bound on the gap, ``mean - beta * std``: a molecule is worth
measuring if it is predicted to have a small gap, or if the model is unsure about it. A greedy
diversity filter then skips any molecule too similar (Tanimoto) to one already in the batch,
so a batch does not spend its budget on 64 near-copies of the same scaffold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Part of the cache key of every task that uses this module: bump it when the model, its
# features, or the acquisition changes, so cached proposals are not reused.
SURROGATE_VERSION = "mlp-ensemble-lcb/1"
FP_BITS = 2048


def fingerprints(smiles: list[str]) -> np.ndarray:
    """Morgan fingerprints (radius 2) as a packed uint8 matrix, 256 bytes per molecule."""
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    out = np.zeros((len(smiles), FP_BITS), dtype=np.uint8)
    for i, s in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(s)), out[i])
    return np.packbits(out, axis=1)


def unpack(packed: np.ndarray) -> np.ndarray:
    import numpy as np

    return np.unpackbits(packed, axis=1).astype(np.float32)


# {{docs-fragment select}}
def fit_ensemble(X: np.ndarray, y: np.ndarray, members: int = 8, seed: int = 0) -> list:
    import numpy as np
    from sklearn.neural_network import MLPRegressor

    rng = np.random.default_rng(seed)
    models = []
    for m in range(members):
        idx = rng.choice(len(y), len(y), replace=True)
        net = MLPRegressor(hidden_layer_sizes=(256, 64), alpha=1e-3, learning_rate_init=1e-3,
                           max_iter=300, random_state=seed * 100 + m)
        models.append(net.fit(X[idx], y[idx]))
    return models


def predict(models: list, X: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    preds = np.stack([np.concatenate([m.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)]) for m in models])
    return preds.mean(0), preds.std(0)


def tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tanimoto similarity of one packed fingerprint ``a`` to each row of ``b``."""
    import numpy as np

    inter = np.unpackbits(a & b, axis=-1).sum(-1)
    union = np.unpackbits(a | b, axis=-1).sum(-1)
    return inter / np.maximum(union, 1)


def select_batch(score: np.ndarray, packed: np.ndarray, exclude: set[int], k: int, max_similarity: float) -> list[int]:
    """Greedy: best score first, skipping excluded molecules and near-duplicates of earlier picks."""
    import numpy as np

    picked: list[int] = []
    for i in np.argsort(score):  # lowest score (best) first
        i = int(i)
        if i in exclude:
            continue
        if picked and tanimoto(packed[i], packed[picked]).max() > max_similarity:
            continue
        picked.append(i)
        if len(picked) == k:
            break
    return picked
# {{/docs-fragment select}}
