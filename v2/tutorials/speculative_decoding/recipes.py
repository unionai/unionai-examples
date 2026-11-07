"""Training and serving recipes for three draft model families.

All three train with speculators in online mode, from the same target-generated data. They
differ in what they predict and how:

- EAGLE-3 drafts one token at a time with a single transformer layer, autoregressively.
- DFlash drafts a whole block of tokens in one forward pass, with a small block-diffusion
  model: every position in the block is predicted in parallel from the target's features.
- DSpark is DFlash plus two heads: a Markov head that models dependencies between tokens
  inside the block, and a confidence head that predicts which positions will be accepted.

Each recipe pins the target layers the drafter reads. The hidden-state server must capture
exactly those layers, and the serving engine must use the same draft length.
"""

from dataclasses import dataclass, field

from config import PUBLIC_DRAFTER


# {{docs-fragment recipe}}
@dataclass(frozen=True)
class Recipe:
    method: str  # speculators' --speculator-type, and vLLM's speculative method
    init_from: str  # Hugging Face head to fine-tune; "" trains from scratch
    target_layers: tuple[int, ...]  # Target layers the drafter reads
    num_speculative_tokens: int  # Draft tokens per verification step when serving
    lr: float
    epochs: int
    train_args: tuple[str, ...] = field(default_factory=tuple)  # Method-specific flags
    vllm_memory: float = 0.45  # Share of a shared GPU for the hidden-state server
    seq_len: int = 4096  # Training sequence length, in tokens
# {{/docs-fragment recipe}}


# {{docs-fragment eagle3}}
EAGLE3 = Recipe(
    method="eagle3",
    init_from=PUBLIC_DRAFTER,
    target_layers=(2, 18, 33),
    num_speculative_tokens=3,
    lr=5e-5,
    epochs=2,
    # Training-time test: unroll the draft on its own predictions for 3 steps.
    train_args=("--ttt-steps", "3"),
)
# {{/docs-fragment eagle3}}

# {{docs-fragment dflash}}
DFLASH = Recipe(
    method="dflash",
    init_from="RedHatAI/Qwen3-8B-speculator.dflash",
    target_layers=(2, 10, 18, 26, 34),
    # A block of 16 positions: the bonus token plus 15 drafted tokens.
    num_speculative_tokens=15,
    lr=1e-4,
    epochs=2,
    train_args=(
        "--block-size", "16",
        # Anchors are the positions a block is drafted from; fewer anchors per sequence
        # trade throughput for memory on a 48 GB GPU.
        "--max-anchors", "512",
        # D-PACE weights each block position by how often it can be accepted.
        "--loss-fn", "ce",
        "--per-position-loss-weight", "dpace",
        "--gradient-checkpointing",
    ),
    # Block drafters predict every block position over the vocabulary, so on a shared 48 GB
    # GPU they train on shorter sequences to keep the logits in memory.
    seq_len=2048,
)
# {{/docs-fragment dflash}}

# {{docs-fragment dspark}}
DSPARK = Recipe(
    method="dspark",
    init_from="",  # No permissively licensed DSpark head for Qwen3-8B: train from scratch.
    target_layers=(2, 10, 18, 26, 34),
    # A block of 8 positions: the bonus token plus 7 drafted tokens.
    num_speculative_tokens=7,
    lr=3e-4,
    epochs=5,  # From random weights: more passes over the data.
    train_args=(
        "--block-size", "8",
        # From scratch, every draft weight trains, with fp32 Adam state. Three decoder layers
        # and 256 anchors fit next to the hidden-state server on one 48 GB GPU.
        "--max-anchors", "256",
        "--num-layers", "3",
        "--draft-vocab-size", "32000",
        # The Markov head adds a low-rank dependency between neighboring block tokens.
        "--markov-rank", "256",
        "--markov-head-type", "vanilla",
        # The confidence head predicts acceptance, so serving can stop a draft early.
        "--enable-confidence-head",
        "--confidence-head-with-markov",
        "--confidence-head-alpha", "1.0",
        "--loss-fn", '{"ce": 0.1, "tv": 0.9}',
        "--gradient-checkpointing",
    ),
    # Block drafters predict every block position over the vocabulary, so on a shared 48 GB
    # GPU they train on shorter sequences to keep the logits in memory.
    seq_len=2048,
)
# {{/docs-fragment dspark}}

RECIPES = {r.method: r for r in (EAGLE3, DFLASH, DSPARK)}
