# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.0",
#    "scikit-learn",
#    "numpy",
# ]
# main = "incremental_sgd"
# params = "chunks=10"
# ///

"""Resume scikit-learn incremental training (`partial_fit`) across task retries.

The estimator and the number of chunks already trained are pickled together
after each chunk, so a retry unpickles them and continues from the next chunk.
"""

# {{docs-fragment env}}
import pathlib
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_sklearn_partial",
    image=flyte.Image.from_debian_base().with_pip_packages("scikit-learn"),
    resources=flyte.Resources(cpu="2", memory="1Gi"),
)

RETRIES = 3
# {{/docs-fragment env}}


# {{docs-fragment task}}
@env.task(retries=RETRIES)
async def incremental_sgd(chunks: int = 10) -> float:
    checkpoint = flyte.ctx().checkpoint

    # Resume the estimator and progress from the previous attempt, if any.
    prev = await checkpoint.load()
    if prev:
        bundle = pickle.loads(prev.read_bytes())
        start = bundle["chunks_done"]
        clf = bundle["clf"]
    else:
        start = 0
        clf = SGDClassifier(max_iter=1, tol=None, random_state=0)

    bundle_path = pathlib.Path("sklearn_partial") / "sgd_bundle.pkl"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    classes = np.array([0, 1])

    failure_interval = chunks // RETRIES
    for i in range(start, chunks):
        x = rng.standard_normal((32, 8))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        clf.partial_fit(x, y, classes=classes)

        if i > start and i % failure_interval == 0:
            # Simulate a failure so the next attempt resumes from the checkpoint
            raise RuntimeError(f"Simulated failure at chunk {i}")

        # Pickle the estimator plus progress and save it to object storage.
        bundle_path.write_bytes(pickle.dumps({"clf": clf, "chunks_done": i + 1}))
        await checkpoint.save(bundle_path)

    x_test = rng.standard_normal((64, 8))
    y_test = (x_test[:, 0] + x_test[:, 1] > 0).astype(int)
    return float(clf.score(x_test, y_test))
# {{/docs-fragment task}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(incremental_sgd, chunks=10)
    print(run.url)
