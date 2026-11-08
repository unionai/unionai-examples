"""Names, models, benchmarks and images shared by the trainer and the evaluator.

The trainer and the evaluator are deployed and run separately. They share only what is in
this file: the artifact names that connect them and the container images.
"""

import flyte

# {{docs-fragment names}}
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"

# The trainer publishes every checkpoint as a new version of this artifact, partitioned by
# training run, step and profile ("full" or "smoke").
CHECKPOINT_ARTIFACT = "math-sft-checkpoint"
# Every eval run publishes its scores as a new version of this artifact, same partitions.
RESULTS_ARTIFACT = "math-sft-checkpoint-evals"
# The checkpoint that wins the ship rule is promoted to this artifact.
RELEASE_ARTIFACT = "math-sft-release-candidate"
# {{/docs-fragment names}}

# {{docs-fragment benchmarks}}
# One entry per benchmark slice. Bump a benchmark's version to re-score only that benchmark on
# every checkpoint; the other benchmarks stay cached. `limit` is the full-profile sample size.
BENCHMARKS = {
    "gsm8k": {"kind": "generate", "version": "1", "limit": 0, "label": "GSM8K (exact match)", "role": "target"},
    # v2 changed the ARC prompt (see ARC_PROMPTS in benchmarks.py); bumping the version re-scored
    # ARC-Easy on every checkpoint and left the other benchmarks cached.
    "arc_easy": {"kind": "choice", "version": "2", "limit": 0, "label": "ARC-Easy (MC)", "role": "reasoning"},
    "mmlu": {"kind": "choice", "version": "1", "limit": 2000, "label": "MMLU (MC)", "role": "guard"},
    # Not in the default suite. Add it with `python evals.py --backfill <run> arc_challenge`.
    "arc_challenge": {"kind": "choice", "version": "2", "limit": 0, "label": "ARC-Challenge (MC)", "role": "reasoning"},
}
DEFAULT_SUITE = ["gsm8k", "arc_easy", "mmlu"]
SMOKE_LIMIT = 40

# The ship rule: the best GSM8K checkpoint whose MMLU score is at most this many points below
# the base model's.
GUARD_BENCHMARK = "mmlu"
GUARD_MAX_DROP = 2.0
# {{/docs-fragment benchmarks}}

# {{docs-fragment images}}
# One GPU image trains the model, generates with vLLM, and scores multiple choice with transformers.
gpu_image = (
    flyte.Image.from_debian_base(name="ckpt-evals-gpu", python_version=(3, 12))
    .with_pip_packages("vllm==0.33.0", "transformers==5.19.0", "datasets>=4.0.0", "accelerate")
)

# Drivers, reports and artifact bookkeeping need nothing but flyte.
driver_image = flyte.Image.from_debian_base(name="ckpt-evals-driver", python_version=(3, 12))
# {{/docs-fragment images}}
