"""The oracle: a GFN2-xTB calculation of one molecule's HOMO-LUMO gap.

This is what a lab measurement costs in this tutorial: real compute that can fail. For one
SMILES string it

1. embeds several 3D conformers with RDKit (ETKDG) and relaxes them with the MMFF force field,
2. optimizes the lowest-energy conformers with GFN2-xTB, a semi-empirical quantum chemistry
   method, through tblite and ASE,
3. checks that no bond broke during the optimization, and
4. reads the HOMO and LUMO energies of the lowest-energy optimized structure.

Each step can fail: a strained ring system may not embed, the self-consistent field (SCF) may
not converge, or the geometry may fall apart. A failed attempt is retried with a different
embedding strategy and a higher electronic temperature, which smooths orbital occupations and
helps the SCF converge. If every attempt fails, the molecule is reported as a failed
measurement with the reason, never as an exception, so one bad molecule never fails a round.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Part of the cache key of every measurement: bump it when the protocol below changes, so the
# cache never returns a gap computed the old way.
ORACLE_VERSION = "gfn2-xtb-8conf-best3/1"
HARTREE_TO_EV = 27.211386
BOHR = 0.52917721

# (embedding strategy, electronic temperature in K): the first attempt is the standard
# protocol, the later ones trade a little accuracy for robustness.
ATTEMPTS = [("etkdg", 300.0), ("random-coords", 1000.0), ("random-coords", 3000.0)]


@dataclass
class Measurement:
    smiles: str
    ok: bool = False
    gap: float = 0.0  # HOMO-LUMO gap in eV
    homo: float = 0.0
    lumo: float = 0.0
    energy: float = 0.0  # GFN2-xTB total energy, Hartree
    atoms: int = 0
    attempts: int = 0
    seconds: float = 0.0
    error: str = ""
    measured_in: str = ""  # the run that computed it; a cache hit keeps the original run's name


def _embed(mol, strategy: str, n_conf: int, seed: int) -> list[int]:
    from rdkit.Chem import AllChem

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if strategy == "random-coords":
        params.useRandomCoords = True
        params.maxIterations = 2000
    return list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params))


def _bonds_intact(mol, positions) -> bool:
    """True if every bond in the molecular graph is still shorter than 1.4x its covalent length."""
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        r = sum((positions[i][k] - positions[j][k]) ** 2 for k in range(3)) ** 0.5
        ref = table.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) + table.GetRcovalent(mol.GetAtomWithIdx(j).GetAtomicNum())
        if r > 1.4 * ref:
            return False
    return True


# {{docs-fragment attempts}}
def _attempt(mol, numbers, strategy: str, temperature: float, n_conf: int, keep: int, seed: int) -> dict:
    import numpy as np
    from ase import Atoms
    from ase.optimize import BFGS
    from rdkit.Chem import AllChem
    from tblite.ase import TBLite
    from tblite.interface import Calculator

    conf_ids = _embed(mol, strategy, n_conf, seed)
    if not conf_ids:
        raise RuntimeError("conformer embedding failed")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        mm = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    else:
        mm = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
    ranked = sorted(range(len(conf_ids)), key=lambda k: mm[k][1])[:keep]

    best = None
    for k in ranked:
        atoms = Atoms(numbers=numbers, positions=mol.GetConformer(conf_ids[k]).GetPositions())
        atoms.calc = TBLite(method="GFN2-xTB", electronic_temperature=temperature, max_iterations=250, verbosity=0)
        BFGS(atoms, logfile=None).run(fmax=0.05, steps=300)
        energy = atoms.get_potential_energy()
        if best is None or energy < best[0]:
            best = (energy, atoms.get_positions().copy())
    if not _bonds_intact(mol, best[1]):
        raise RuntimeError("a bond broke during the xTB optimization")

    calc = Calculator("GFN2-xTB", np.array(numbers), best[1] / BOHR)
    calc.set("verbosity", 0)
    calc.set("temperature", temperature * 3.166811563e-6)  # K to Hartree
    res = calc.singlepoint()
    eps, occ = res.get("orbital-energies"), res.get("orbital-occupations")
    n_occ = int(round(float(occ.sum()) / 2))
    homo, lumo = float(eps[n_occ - 1]) * HARTREE_TO_EV, float(eps[n_occ]) * HARTREE_TO_EV
    return {"gap": lumo - homo, "homo": homo, "lumo": lumo, "energy": float(res.get("energy"))}


def measure_gap(smiles: str, n_conf: int = 8, keep: int = 3, max_atoms: int = 90) -> Measurement:
    """Compute the GFN2-xTB HOMO-LUMO gap of one molecule, retrying failed attempts."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    started = time.time()
    out = Measurement(smiles=smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out.error = "invalid SMILES"
        return out
    mol = Chem.AddHs(mol)
    out.atoms = mol.GetNumAtoms()
    if out.atoms > max_atoms:
        out.error = f"too large ({out.atoms} atoms)"
        return out
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    errors = []
    for i, (strategy, temperature) in enumerate(ATTEMPTS):
        out.attempts = i + 1
        try:
            result = _attempt(Chem.Mol(mol), numbers, strategy, temperature, n_conf, keep, seed=0xF00D + i)
        except Exception as e:  # SCF not converged, embedding failed, broken geometry, ...
            errors.append(f"attempt {i + 1}: {str(e).strip().splitlines()[0][:120] if str(e).strip() else type(e).__name__}")
            continue
        out.ok, out.error = True, "; ".join(errors)
        out.gap, out.homo, out.lumo, out.energy = result["gap"], result["homo"], result["lumo"], result["energy"]
        break
    else:
        out.error = "; ".join(errors)
    out.seconds = round(time.time() - started, 2)
    return out
# {{/docs-fragment attempts}}
