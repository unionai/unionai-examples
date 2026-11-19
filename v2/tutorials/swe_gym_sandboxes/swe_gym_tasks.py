"""SWE-Gym tasks: which instances, how to build each one's testbed, and how to grade a patch.

Modal's example gets all three from the `swegym` harness (`MAP_REPO_VERSION_TO_SPECS`,
`make_eval_script_list`, `get_logs_eval`) and from prebuilt Docker images
(`xingyaoww/sweb.eval.x86_64.<id>`). A Union sandbox can't start from a custom image yet,
so this file carries the part of the harness the subset needs, as data:

    SPECS        per repo and version: the Python, the pre-install steps and the install line,
                 copied from swegym/harness/constants.py (SWE-Bench-Package @16dd480)
    eval_script  apply the hidden test patch and run the instance's tests (make_eval_script_list)
    grade_log    parse pytest's -rA summary and decide `resolved` (get_logs_eval + get_eval_report)
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlopen

# The same 293-task file Modal's example reads; we pick from its 31-task subset.
DATA_URL = (
    "https://raw.githubusercontent.com/NVIDIA-NeMo/ProRL-Agent-Server/"
    "6a1ead6bfac054fce6c1e62d1a77b330d96c58db/"
    "examples/swegym_slime_grpo/swegym_train_293.jsonl"
)

# Eight of Modal's 31 task ids whose testbeds build from source in a few minutes, with no
# conda (pandas, dask and MONAI need conda or torch, so they wait for custom images).
DEFAULT_IDS = [
    "getmoto__moto-7023",
    "python__mypy-16781",
    "getmoto__moto-7365",
    "getmoto__moto-6913",
    "python__mypy-10683",
    "python__mypy-11134",
    "iterative__dvc-6683",
    "facebookresearch__hydra-1560",
]
SMOKE_IDS = ["getmoto__moto-7023", "python__mypy-16781"]

CACHE = Path.home() / ".cache" / "swe-gym-union"


def load_tasks(ids: List[str]) -> List[Dict[str, Any]]:
    """The SWE-Gym instances with these ids, in this order (downloaded once, then cached)."""
    path = CACHE / "swegym_train_293.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(DATA_URL, timeout=60) as r:
            path.write_bytes(r.read())
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {row["metadata"]["instance_id"]: row["metadata"]["instance"] for row in rows}
    return [by_id[i] for i in ids]


# ---------------------------------------------------------------------------------------------
# Testbeds: what a SWE-Gym image has, built from source in a sandbox.
# ---------------------------------------------------------------------------------------------

_MYPY_PRE = ["git submodule update --init mypy/typeshed || true"]
_MYPY_INSTALL = (
    "python -m pip install -r test-requirements.txt; python -m pip install -e .; "
    "pip install pytest pytest-xdist; hash -r"
)


def _mypy(version: str) -> Dict[str, Any]:
    major_minor = tuple(int(x) for x in version.split(".")[:2])
    if major_minor >= (1, 7):
        return {"python": "3.12", "pre": _MYPY_PRE, "install": _MYPY_INSTALL, "test": "pytest -rA -k"}
    if major_minor >= (1, 0):
        return {"python": "3.11", "pre": _MYPY_PRE, "install": _MYPY_INSTALL, "test": "pytest -n0 -rA -k"}
    if major_minor >= (0, 940):
        return {"python": "3.10", "pre": _MYPY_PRE, "install": _MYPY_INSTALL, "test": "pytest -n0 -rA -k"}
    return {
        "python": "3.9",
        # types-typing-extensions is yanked; the harness pins it by hand.
        "pre": [*_MYPY_PRE, "sed -i '1i types-typing-extensions==3.7.3' test-requirements.txt"],
        "install": _MYPY_INSTALL,
        "test": "pytest -n0 -rA -k",
    }


def _moto(version: str) -> Dict[str, Any]:
    return {"python": "3.12", "pre": [], "install": "make init", "test": "pytest -n0 -rA"}


def _dvc(version: str) -> Dict[str, Any]:
    install = (
        'python -m pip install --upgrade pip wheel GitPython; python -m pip install "cython<3.0.0" && '
        "python -m pip install --no-build-isolation pyyaml==5.4.1; "
        "python -m pip install git+https://github.com/iterative/mock-ssh-server.git || true; "
        "python -m pip install -r tests/requirements.txt || true; "
        "python -m pip install -r test-requirements.txt || true; "
        'python -m pip install -e ".[tests,dev,all_remotes,all,testing]"; '
        'python -m pip install "numpy<=1.20"; python -m pip install "pytest<8";'
    )
    return {"python": "3.9", "pre": [], "apt": ["cmake"], "install": install, "test": "pytest -rA"}


def _hydra(version: str) -> Dict[str, Any]:
    install = (
        "sed -i 's|isort@git+git://github.com/timothycrosley/isort|"
        "isort@git+https://github.com/timothycrosley/isort|g' "
        "requirements/dev.txt; "
        '{ tail -n1 requirements/requirements.txt | grep -q "." && echo ""; } >> requirements/requirements.txt; '
        'echo "pip==24.0" >> requirements/requirements.txt; pip install "pip==24.0"; '
        "pip install -r requirements/dev.txt; pip install -e .;"
    )
    return {
        "python": "3.8",
        "pre": [],
        "apt": ["openjdk-17-jdk-headless"],
        "install": install,
        "test": "pytest -rA --tb=long",
    }


SPECS = {"python/mypy": _mypy, "getmoto/moto": _moto, "iterative/dvc": _dvc, "facebookresearch/hydra": _hydra}


def spec(inst: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return SPECS[inst["repo"]](inst["version"])
    except KeyError:
        raise ValueError(
            f"{inst['instance_id']}: no testbed recipe for {inst['repo']}; add one to SPECS "
            "(or wait for custom sandbox images, which would run its SWE-Gym image as it is)"
        ) from None


#: SWE-Gym images run every command in the `testbed` conda env:
#: `source /opt/miniconda3/bin/activate testbed`. The testbed here is a venv, so this shim
#: stands in for conda at the same path, and the rollout's commands run unchanged.
ACTIVATE_SHIM = """# Union testbed: a venv standing in for SWE-Gym's conda env `testbed`.
conda() { if [ "${1:-}" = activate ]; then . /opt/testbed/bin/activate; fi; }
. /opt/testbed/bin/activate
"""


def build_script(inst: Dict[str, Any]) -> str:
    """Shell that turns the platform's base VM into this instance's testbed: /testbed at the
    base commit with its history after it removed, and its deps installed in the `testbed` env."""
    s = spec(inst)
    apt = " ".join(["git", "make", "gcc", "g++", "patch", *s.get("apt", [])])
    pre = "\n".join(s["pre"])
    return f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
echo "== apt: {apt}"
apt-get update -qq && apt-get install -y -qq {apt} >/dev/null
echo "== python {s["python"]} venv at /opt/testbed"
python3 -m pip install -q uv
uv python install -q {s["python"]}
uv venv -q -p {s["python"]} --seed /opt/testbed
mkdir -p /opt/miniconda3/bin && cat > /opt/miniconda3/bin/activate <<'SHIM'
{ACTIVATE_SHIM}SHIM
echo "== clone {inst["repo"]} @ {inst["base_commit"][:12]}"
git clone -q https://github.com/{inst["repo"]}.git /testbed
cd /testbed
git -c advice.detachedHead=false checkout -q {inst["base_commit"]}
git config --global user.email testbed@example.com && git config --global user.name testbed
{pre}
echo "== install"
set +u; source /opt/miniconda3/bin/activate testbed; set -u
set +e
{s["install"]}
set -e
python -m pip install -q pytest
echo "== drop history after the base commit (the fix is in it)"
git checkout -q --detach HEAD
git remote | xargs -r -n1 git remote remove
git for-each-ref --format='%(refname)' | while read r; do git update-ref -d "$r"; done
git reflog expire --expire=now --all
git gc -q --prune=now
git status --short | head -20
echo "== testbed ready"
"""


def testbed_artifact(inst: Dict[str, Any]) -> str:
    """The artifact a testbed is published as: the instance, and a hash of its recipe, so a
    changed recipe builds afresh rather than reusing a testbed it no longer describes."""
    recipe = hashlib.sha256(build_script(inst).encode()).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", inst["instance_id"].lower()).strip("-")
    return f"swe-testbed-{slug}-{recipe}"


# ---------------------------------------------------------------------------------------------
# Grading: the hidden tests, as the harness runs them.
# ---------------------------------------------------------------------------------------------


def test_files(test_patch: str) -> List[str]:
    """The files the hidden test patch touches."""
    files = []
    for line in test_patch.splitlines():
        if line.startswith(("+++ b/", "--- a/")):
            f = line[6:]
            if f not in files:
                files.append(f)
    return files


def _test_command(inst: Dict[str, Any], test: str) -> str:
    ids = list(inst["FAIL_TO_PASS"]) + list(inst["PASS_TO_PASS"])
    if test.endswith("-k"):
        # mypy's tests are data-driven: run its test modules, picking the cases by name.
        modules = sorted({i.split("::")[0] for i in ids})
        names = sorted({i.split("::")[-1] for i in ids})
        return test[: -len("-k")] + " ".join(map(shlex.quote, modules)) + " -k " + shlex.quote(" or ".join(names))
    return test + " " + " ".join(shlex.quote(f) for f in sorted({i.split("::")[0] for i in ids}))


def eval_script(inst: Dict[str, Any], test_patch_path: str = "/tmp/hidden_test.patch") -> str:
    """Applies the hidden test patch (written to ``test_patch_path``) and runs the tests."""
    s = spec(inst)
    files = " ".join(shlex.quote(f) for f in test_files(inst["test_patch"]))
    base = inst["base_commit"]
    return f"""set -uo pipefail
set +u; source /opt/miniconda3/bin/activate testbed; set -u
cd /testbed
git checkout {base} -- {files} 2>/dev/null
git apply -v {test_patch_path}
{_test_command(inst, s["test"])}
rc=$?
git checkout {base} -- {files} 2>/dev/null
exit $rc
"""


_STATUSES = ("PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL")


def parse_pytest_log(log: str) -> Dict[str, str]:
    """Test id -> status, from pytest's -rA summary (the harness's parse_log_pytest)."""
    out: Dict[str, str] = {}
    for raw in log.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        if line.startswith(_STATUSES):
            parts = (line.replace(" - ", " ") if line.startswith("FAILED") else line).split()
            if len(parts) > 1:
                out[parts[1]] = parts[0]
    return out


def grade_log(inst: Dict[str, Any], log: str, pass_to_pass: List[str] | None = None) -> Dict[str, Any]:
    """The harness's report: resolved when every FAIL_TO_PASS and PASS_TO_PASS test passes."""
    status = parse_pytest_log(log)
    ok = lambda t: status.get(t) in ("PASSED", "XFAIL")  # noqa: E731
    f2p = list(inst["FAIL_TO_PASS"])
    p2p = list(inst["PASS_TO_PASS"] if pass_to_pass is None else pass_to_pass)
    report = {
        "parsed": bool(status),
        "FAIL_TO_PASS": {"success": [t for t in f2p if ok(t)], "failure": [t for t in f2p if not ok(t)]},
        "PASS_TO_PASS": {"success": [t for t in p2p if ok(t)], "failure": [t for t in p2p if not ok(t)]},
    }
    report["resolved"] = (
        bool(status) and not report["FAIL_TO_PASS"]["failure"] and not report["PASS_TO_PASS"]["failure"]
    )
    return report
