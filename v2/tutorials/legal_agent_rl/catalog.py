"""Profile every LAB task without downloading its documents: criteria counts from task.json, document sizes from
GitHub's tree API.

    python catalog.py                # writes lab_tasks.json next to this file

Fetches only the task.json blobs (a few MB) and one tree listing per practice area, so it runs in a minute on a
laptop. ``sweep.pick_tasks`` samples from the result.
"""

from __future__ import annotations

import collections
import json
import subprocess
import urllib.request
import tempfile
import time
import urllib.error
from pathlib import Path

from images import LAB_COMMIT
from lab_harness import LAB_REPO


def build(commit: str = LAB_COMMIT) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        run = lambda *a: subprocess.run(["git", "-C", d, *a], check=True, capture_output=True, text=True).stdout  # noqa: E731
        run("init", "-q")
        run("remote", "add", "origin", LAB_REPO)
        run("sparse-checkout", "set", "--no-cone", "/tasks/**/task.json")
        run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
        run("checkout", "-q", commit)
        sizes = _document_sizes(commit, run)
        rows = []
        for tj in sorted(Path(d, "tasks").rglob("task.json")):
            tid = str(tj.parent.relative_to(Path(d, "tasks")))
            t = json.loads(tj.read_text())
            docs = sizes.get(tid, [])
            rows.append({"id": tid, "area": tid.split("/")[0], "n": len(t["criteria"]), "work_type": t.get("work_type"),
                         "ndeliv": len(t.get("deliverables", {})), "deliverables": sorted(t.get("deliverables", {})),
                         "ndocs": len(docs), "kb": round(sum(s for _, s in docs) / 1024, 1),
                         "exts": dict(collections.Counter(e for e, _ in docs)), "docs_dir": t.get("docs_dir")})
        return rows


def _tree(sha: str) -> dict:
    url = f"https://api.github.com/repos/harveyai/harvey-labs/git/trees/{sha}"
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lab-catalog"})))
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _blobs(sha: str, prefix: str):
    """Every (path, size) under a tree; one recursive call, or one per subtree when GitHub truncates the listing."""
    try:
        url_tree = _tree(f"{sha}?recursive=1")
    except urllib.error.HTTPError:  # GitHub sometimes 500s on a very large recursive listing; walk it instead
        url_tree = {"truncated": True}
    if not url_tree.get("truncated"):
        yield from ((f"{prefix}{e['path']}", e["size"]) for e in url_tree["tree"] if e["type"] == "blob")
        return
    for e in _tree(sha)["tree"]:
        if e["type"] == "blob":
            yield f"{prefix}{e['path']}", e["size"]
        else:
            yield from _blobs(e["sha"], f"{prefix}{e['path']}/")


def _document_sizes(commit: str, run) -> dict[str, list[tuple[str, int]]]:
    """(extension, bytes) per document, per task. A partial clone has no blob sizes, so ask GitHub's tree API, one
    practice area at a time."""
    sizes: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for line in run("ls-tree", commit, "tasks/").splitlines():
        sha, area = line.split()[2], line.split("\t")[1].split("/")[-1]
        for path, size in _blobs(sha, f"{area}/"):
            if "/documents/" in path:
                sizes[path.split("/documents/")[0]].append((Path(path).suffix.lower(), size))
    return sizes


if __name__ == "__main__":
    rows = build()
    out = Path(__file__).with_name("lab_tasks.json")
    out.write_text(json.dumps({"commit": LAB_COMMIT, "tasks": rows}, indent=0))
    print(f"{len(rows)} tasks, {sum(r['n'] for r in rows)} criteria -> {out}")
