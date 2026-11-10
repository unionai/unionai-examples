# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
# ]
# main = "discover"
# params = "rounds=2 batch_size=16 pool_size=2000"
# ///
"""Closed-loop molecular discovery: a surrogate proposes, a quantum chemistry oracle measures.

    build_pool                     20,000 drug-like molecules from ZINC250k, with fingerprints
    round 0: measure 64 random molecules, shared by both arms
    for each round:
        propose (warm CPU)         retrain an MLP ensemble, pick 64 by a confidence bound
        approval (condition)       a scientist approves the batch, or it auto-approves on timeout
        N x measure (CPU pool)     GFN2-xTB HOMO-LUMO gap, cached per molecule
        and 64 random molecules through the same oracle, as the control arm
    report best gap found against oracle calls, and publish the top molecules as a dataset

The objective is to find molecules with a small HOMO-LUMO gap, a cheap proxy for organic
semiconductors and dyes. The oracle is real compute that can fail: conformer embedding, a
GFN2-xTB geometry optimization, and a single point. Each round is a cached task whose output
is the whole loop state, so a driver that crashes resumes from the last finished round, and a
molecule measured once is never computed again, in this run or the next.

    uv run discovery.py              # 10 rounds of 64 over 20,000 molecules, about 45 minutes
    uv run discovery.py --smoke      # 2 rounds of 16 over 2,000 molecules, 60 s approval timeout
"""

import asyncio
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace

import flyte
import flyte.artifacts as artifacts
import flyte.errors
import flyte.report
import reports
from flyte.io import File
from oracle import ORACLE_VERSION, Measurement
from surrogate import SURROGATE_VERSION

def round_s(x: float) -> float:
    return round(x, 1)


ZINC_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/"
    "models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)

# {{docs-fragment envs}}
# RDKit for molecules (its drawing code needs the X rendering libraries), tblite and ASE for
# GFN2-xTB, scikit-learn for the surrogate, and unionai-reuse for the reusable environments.
# tblite 0.8.0 ships Linux wheels for x86_64 only, so the image is built for linux/amd64, even
# by a local builder on an arm64 machine.
image = (
    flyte.Image.from_debian_base(name="closed-loop-discovery", python_version=(3, 12), platform=("linux/amd64",))
    .with_apt_packages("libxrender1", "libxext6", "libexpat1")
    .with_pip_packages("rdkit==2026.9.1", "tblite==0.8.0", "ase==3.29.0", "scikit-learn==1.9.1", "numpy==2.5.4",
                       "unionai-reuse==0.2.1")
)
CODE = ["oracle.py", "surrogate.py", "reports.py", "viz.py"]

# The oracle: a warm pool of small CPU pods, one molecule per call. Every call is cached on
# its SMILES string, so a molecule is measured once, ever, whichever arm or run asks for it.
# "auto" caching hashes the task's own source, not the modules it imports, so the tasks that
# use oracle.py and surrogate.py add those modules' versions to their cache keys (see below).
oracle_env = flyte.TaskEnvironment(
    name="discovery-oracle",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 12), concurrency=1, idle_ttl=600, scaledown_ttl=120),
    env_vars={"OMP_NUM_THREADS": "2", "OMP_STACKSIZE": "1G"},
    include=CODE,
)

# The surrogate retrains every round on a single warm replica, which keeps its imports and
# the downloaded pool between rounds.
model_env = flyte.TaskEnvironment(
    name="discovery-model",
    image=image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    reusable=flyte.ReusePolicy(replicas=1, concurrency=1, idle_ttl=900),
    include=CODE,
)

driver_env = flyte.TaskEnvironment(
    name="discovery-driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    include=CODE,
    depends_on=[oracle_env, model_env],
)
# {{/docs-fragment envs}}


# {{docs-fragment state}}
@dataclass
class Pick:
    smiles: str
    mean: float  # predicted gap, eV
    std: float  # ensemble spread, eV
    score: float  # the acquisition score, mean - beta * std


@dataclass
class Proposal:
    round: int
    batch: list[Pick]  # the active learning batch
    control: list[Pick]  # the random control batch, with the same model's predictions
    train_size: int
    pool_median: float  # median predicted gap over the whole pool
    fit_seconds: float


@dataclass
class RoundLog:
    round: int
    decision: str  # "seed library", "approved", "rejected", "auto-approved", "no gate"
    gate_seconds: float = 0.0
    proposal: Proposal | None = None
    active: list[Measurement] = field(default_factory=list)
    control: list[Measurement] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class LoopState:
    rounds: list[RoundLog] = field(default_factory=list)
    vetoed: list[str] = field(default_factory=list)  # molecules a scientist rejected

    def measured(self, arm: str) -> list[Measurement]:
        """Every measurement of one arm ("active" or "control") so far."""
        return [m for r in self.rounds for m in getattr(r, arm)]
# {{/docs-fragment state}}


# {{docs-fragment pool}}
@model_env.task(cache="auto")
async def build_pool(pool_size: int = 20_000, seed: int = 0) -> File:
    """A random sample of neutral ZINC250k molecules, with their Morgan fingerprints."""
    import csv
    import io
    import urllib.request

    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from surrogate import fingerprints

    RDLogger.DisableLog("rdApp.*")
    text = urllib.request.urlopen(ZINC_URL, timeout=120).read().decode()
    rows = [r["smiles"].strip() for r in csv.DictReader(io.StringIO(text))]
    random.Random(seed).shuffle(rows)

    # Gas-phase GFN2-xTB gives zwitterions and ions unphysically small gaps, so the pool keeps
    # neutral molecules only: neutralize what can be, and drop charged atoms other than nitro groups.
    uncharger, nitro = rdMolStandardize.Uncharger(), Chem.MolFromSmarts("[N+](=O)[O-]")
    pool, seen = [], set()
    for s in rows:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        mol = uncharger.uncharge(mol)
        charged = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge())
        if Chem.GetFormalCharge(mol) != 0 or charged > 2 * len(mol.GetSubstructMatches(nitro)):
            continue
        smiles = Chem.MolToSmiles(mol)
        if smiles not in seen:
            seen.add(smiles)
            pool.append(smiles)
        if len(pool) == pool_size:
            break

    path = f"{tempfile.mkdtemp()}/pool.npz"
    np.savez_compressed(path, smiles=np.array(pool), fps=fingerprints(pool))
    print(f"pool: {len(pool)} molecules from {len(rows)} in ZINC250k")
    return await File.from_local(path)
# {{/docs-fragment pool}}


async def load_pool(pool: File):
    import numpy as np

    data = np.load(await pool.download())
    return [str(s) for s in data["smiles"]], data["fps"]


# {{docs-fragment measure}}
@oracle_env.task(cache=flyte.Cache("auto", salt=ORACLE_VERSION), retries=2)
async def measure(smiles: str) -> Measurement:
    """Measure one molecule's HOMO-LUMO gap with GFN2-xTB. A failed calculation is a result, not an error."""
    from oracle import measure_gap

    m = await asyncio.to_thread(measure_gap, smiles)
    m.measured_in = flyte.ctx().action.run_name or ""
    print(f"{smiles}: {'%.3f eV' % m.gap if m.ok else 'FAILED'} after {m.attempts} attempt(s) in {m.seconds:.1f}s {m.error}")
    return m
# {{/docs-fragment measure}}


# {{docs-fragment propose}}
@model_env.task(cache=flyte.Cache("auto", salt=SURROGATE_VERSION))
async def propose(
    round_num: int, pool: File, train_smiles: list[str], train_gaps: list[float], exclude: list[str],
    control: list[str], batch_size: int, beta: float = 1.0, max_similarity: float = 0.6,
) -> Proposal:
    """Retrain the ensemble on everything measured so far and pick the next batch by a lower confidence bound."""
    import numpy as np
    import surrogate

    started = time.time()
    smiles, packed = await load_pool(pool)
    index = {s: i for i, s in enumerate(smiles)}
    X = surrogate.unpack(packed)
    models = surrogate.fit_ensemble(X[[index[s] for s in train_smiles]], np.array(train_gaps), seed=round_num)
    mean, std = surrogate.predict(models, X)
    score = mean - beta * std  # small predicted gap, or high uncertainty
    chosen = surrogate.select_batch(score, packed, {index[s] for s in exclude}, batch_size, max_similarity)

    def pick(i: int) -> Pick:
        return Pick(smiles[i], float(mean[i]), float(std[i]), float(score[i]))

    print(f"round {round_num}: trained on {len(train_smiles)}, batch predicted {mean[chosen].mean():.2f} eV")
    return Proposal(round_num, [pick(i) for i in chosen], [pick(index[s]) for s in control], len(train_smiles),
                    float(np.median(mean)), round_s(time.time() - started))
# {{/docs-fragment propose}}


# {{docs-fragment approval}}
async def approval(round_num: int, proposal: Proposal, state: LoopState, timeout_s: int) -> tuple[str, float]:
    """Ask a scientist to approve the batch before it goes to the lab. Silence counts as approval."""
    if timeout_s <= 0:
        return "no gate", 0.0
    started = time.time()
    condition = await flyte.new_condition.aio(
        f"approve-round-{round_num}",
        prompt=reports.approval_prompt(proposal, state, timeout_s),
        prompt_type="markdown",
        data_type=bool,
        description=f"Send round {round_num}'s {len(proposal.batch)} molecules to the oracle?",
        timeout=timeout_s,
    )
    try:
        approved = await condition.wait.aio()
        decision = "approved" if approved else "rejected"
    except flyte.errors.ConditionTimedoutError:
        decision = "auto-approved"  # nobody answered: an unattended run keeps going
    return decision, round_s(time.time() - started)
# {{/docs-fragment approval}}


# {{docs-fragment round}}
@driver_env.task(cache=flyte.Cache("auto", salt=f"{ORACLE_VERSION}+{SURROGATE_VERSION}"), retries=1)
async def run_round(
    round_num: int, state: LoopState, pool: File, batch_size: int, beta: float, max_similarity: float,
    approval_timeout_s: int, seed: int, campaign: str,
) -> LoopState:
    """One turn of the loop. The output is the whole state, so a finished round is never run twice.

    The state includes the approval decision, so a cached round is not asked about again. The
    campaign name is part of the cache key: a run with a new campaign asks for approval afresh.
    """
    started = time.time()
    smiles, _ = await load_pool(pool)
    rng = random.Random(seed * 1_000 + round_num)
    if round_num == 0:
        # The seed library: random molecules, measured once and shared by both arms.
        seed_batch = rng.sample(smiles, batch_size)
        results = list(await asyncio.gather(*[measure(s) for s in seed_batch]))
        log = RoundLog(0, "seed library", active=results, control=results)
        return replace(state, rounds=[replace(log, seconds=round_s(time.time() - started))])

    active, control = state.measured("active"), state.measured("control")
    seen_control = {m.smiles for m in control}
    control_batch = rng.sample([s for s in smiles if s not in seen_control], batch_size)
    train = [m for m in active if m.ok]
    proposal = await propose(
        round_num, pool, [m.smiles for m in train], [m.gap for m in train],
        exclude=[m.smiles for m in active] + state.vetoed, control=control_batch,
        batch_size=batch_size, beta=beta, max_similarity=max_similarity,
    )

    decision, waited = await approval(round_num, proposal, state, approval_timeout_s)
    log = RoundLog(round_num, decision, waited, proposal)
    if decision == "rejected":
        state = replace(state, vetoed=state.vetoed + [p.smiles for p in proposal.batch])
    else:
        # Both arms go to the oracle at once. A molecule in both is computed once: the second
        # call is a cache hit.
        batch = [p.smiles for p in proposal.batch]
        results = await asyncio.gather(*[measure(s) for s in batch + control_batch])
        log.active, log.control = list(results[: len(batch)]), list(results[len(batch):])
    log.seconds = round_s(time.time() - started)
    return replace(state, rounds=state.rounds + [log])
# {{/docs-fragment round}}


# {{docs-fragment driver}}
@driver_env.task(report=True, produces_artifacts=True, retries=2)
async def discover(
    rounds: int = 10,
    batch_size: int = 64,
    pool_size: int = 20_000,
    beta: float = 1.0,
    max_similarity: float = 0.6,
    approval_timeout_s: int = 180,
    hit_gap: float = 1.0,
    seed: int = 0,
    campaign: str = "zinc-small-gap",
) -> File:
    """Run the closed loop and the random control, report both, and publish the best molecules.

    Rounds are cached, approvals included: rerunning with the same inputs replays the campaign
    without asking anyone. Pass a new ``campaign`` name to start over and be asked again.
    """
    pool = await build_pool(pool_size, seed)
    state = LoopState()
    for r in range(rounds + 1):
        with flyte.group(f"round-{r}"):
            state = await run_round(
                r, state, pool, batch_size, beta, max_similarity, approval_timeout_s, seed, campaign)
        await flyte.report.replace.aio(reports.live(state, rounds, hit_gap, pool_size), do_flush=True)

    top = reports.top_molecules(state, 50)
    # The rounds' own durations: a resumed driver replays cached rounds in seconds, and this
    # still reports how long the loop took when it ran.
    loop_seconds = sum(r.seconds for r in state.rounds)
    await flyte.report.replace.aio(reports.final(state, hit_gap, pool_size, top, loop_seconds))
    flyte.report.get_tab("Surrogate").replace(reports.surrogate_tab(state))
    flyte.report.get_tab("Batches").replace(reports.batches_tab(state))
    flyte.report.get_tab("Oracle").replace(reports.oracle_tab(state, flyte.ctx().action.run_name or ""))
    await flyte.report.flush.aio()

    path = f"{tempfile.mkdtemp()}/top_candidates.csv"
    reports.write_csv(top, path)
    summary = reports.summary(state, hit_gap)
    card = await artifacts.Card.create_from.aio(content=reports.card(state, top, hit_gap), format="html", card_type="data")
    metadata = artifacts.Metadata(
        name="closed-loop-top-candidates",
        description=f"The {len(top)} smallest-gap molecules found in {summary['calls']} GFN2-xTB calculations "
                    f"over {pool_size:,} ZINC molecules; best {top[0]['gap_ev']:.2f} eV",
        kind="data",
        card=card,
        attrs={k: str(v) for k, v in summary.items()} | {"objective": "minimize GFN2-xTB HOMO-LUMO gap"},
    )
    return artifacts.new(await File.from_local(path), metadata)
# {{/docs-fragment driver}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    if "--smoke" in sys.argv:
        run = flyte.run(discover, rounds=2, batch_size=16, pool_size=2_000, approval_timeout_s=60)
    else:
        run = flyte.run(discover)
    print(run.url)
# {{/docs-fragment run}}
