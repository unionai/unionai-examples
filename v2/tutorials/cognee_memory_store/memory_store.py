"""Claude-inspired memory store for Flyte tutorials.

This module implements an *auditable, versioned, file-based* memory store that is
meant to be synced via Flyte object storage (flyte.io.Dir).

Design goals (inspired by Claude Managed Agents memory stores):
- Many small focused files ("memories") addressed by path.
- Access modes by prefix (e.g. reference/ is read-only).
- Immutable version history for every mutation.
- Append-only audit log.
- Optimistic concurrency via expected sha256 preconditions.

This is intentionally plain-files + JSON metadata so it stays transparent and
teachable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class MemoryStoreError(RuntimeError):
    pass


class AccessDenied(MemoryStoreError):
    pass


class ConcurrencyError(MemoryStoreError):
    def __init__(self, path: str, expected_sha: str, actual_sha: str):
        super().__init__(
            f"ConcurrencyError for {path!r}: expected_sha={expected_sha} actual_sha={actual_sha}"
        )
        self.path = path
        self.expected_sha = expected_sha
        self.actual_sha = actual_sha


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Topic index helpers
# ---------------------------------------------------------------------------

TOPIC_INDEX_PATH = "memory/_index.json"
TOPIC_MAP_PATH = "user/_topic_map.json"

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

SESSION_REGISTRY_PATH = "user/sessions/_registry.json"


def session_memories_prefix(session: str) -> str:
    return f"user/sessions/{session}/memories"


def session_topic_map_path(session: str) -> str:
    return f"user/sessions/{session}/memories/_topic_map.json"


def session_staging_inbox_prefix(session: str) -> str:
    return f"staging/sessions/{session}/inbox"


def session_staging_archive_prefix(session: str) -> str:
    return f"staging/sessions/{session}/archive"


def register_session(store: "MemoryStore", name: str, label: str = "") -> None:
    """Idempotently register a named session in the session registry."""
    registry = store.read_json(SESSION_REGISTRY_PATH, default={})
    if name not in registry:
        registry[name] = {"created_at_s": time.time(), "label": label or name}
        store.write_json(SESSION_REGISTRY_PATH, registry, actor="system", reason="register-session")


def list_sessions(store: "MemoryStore") -> list[str]:
    """Return sorted list of registered session names."""
    registry = store.read_json(SESSION_REGISTRY_PATH, default={})
    return sorted(registry.keys())


def read_topic_map(store: "MemoryStore", topic_map_path: str = TOPIC_MAP_PATH) -> dict:
    """Return {user_rel_path: topic_slug} from the given topic map path."""
    return store.read_json(topic_map_path, default={})


def upsert_topic_map(store: "MemoryStore", rel_path: str, slug: Optional[str], *, topic_map_path: str = TOPIC_MAP_PATH) -> None:
    """Set or clear the topic association for a promoted memory file.

    Writes directly to the filesystem, bypassing MemoryStore access control,
    because _topic_map.json is machine-managed system metadata.
    """
    m = read_topic_map(store, topic_map_path)
    if slug:
        m[rel_path] = slug
    else:
        m.pop(rel_path, None)
    p = store.root / Path(topic_map_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def load_topic_index(store: "MemoryStore") -> dict:
    """Return {slug: {label, sources, last_updated}} from memory/_index.json."""
    return store.read_json(TOPIC_INDEX_PATH, default={})


def upsert_topic_index(
    store: "MemoryStore",
    slug: str,
    *,
    label: str,
    source_url: Optional[str] = None,
    actor: str = "system",
) -> None:
    """Idempotently add or update one slug entry in memory/_index.json.

    Writes directly to the filesystem, bypassing MemoryStore access control,
    because memory/ is machine-managed and the index is its own metadata.
    """
    index = load_topic_index(store)
    entry = index.get(slug, {"label": label, "sources": [], "last_updated": ""})
    entry["label"] = label
    if source_url and source_url not in entry["sources"]:
        entry["sources"].append(source_url)
    entry["last_updated"] = _utc_ts()
    index[slug] = entry
    index_path = store.root / Path(TOPIC_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON parsing helpers (shared by workflow.py and app.py)
# ---------------------------------------------------------------------------

def _parse_json_object(text: str) -> "dict | None":
    """Parse a JSON object from an LLM response, tolerating code fences and preamble."""
    if not text:
        return None
    t = text.strip()
    if not t or t.lower() == "null":
        return None
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t).strip()
        if t.lower() == "null":
            return None
    start = t.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_json_array(text: str) -> list:
    """Parse a JSON array from an LLM response, tolerating code fences."""
    if not text:
        return []
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t).strip()
    start = t.find("[")
    if start < 0:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[start:])
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _ensure_relative_posix(path: str) -> str:
    """Normalize and validate a memory path.

    - Must be relative (no leading '/').
    - Must not contain '..'.
    - Uses POSIX separators regardless of OS.
    """
    p = Path(path)
    if p.is_absolute() or str(path).startswith("/"):
        raise MemoryStoreError(f"Path must be relative, got {path!r}")

    parts: list[str] = []
    for part in p.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise MemoryStoreError(f"Path traversal is not allowed, got {path!r}")
        parts.append(part)

    if not parts:
        raise MemoryStoreError("Empty path is not allowed")

    return "/".join(parts)


@dataclass(frozen=True)
class MemoryMeta:
    path: str
    sha256: str
    updated_at: str
    updated_by: str
    reason: str
    bytes: int


class MemoryStore:
    """A directory-backed memory store with audit + versioning."""

    # Prefix policy (inspired by Claude's read_only vs read_write).
    # memory/ is machine-managed (written only by ingest_url and sleep_cycle);
    # user proposals must land in user/ only.
    READ_ONLY_PREFIXES = ("memory/",)

    def __init__(self, root: Path):
        self.root = Path(root)
        self._audit_path = self.root / "audit" / "log.jsonl"
        self._meta_root = self.root / "meta"
        self._versions_root = self.root / "versions"

    def ensure_layout(self) -> None:
        (self.root / "audit").mkdir(parents=True, exist_ok=True)
        (self.root / "memory").mkdir(parents=True, exist_ok=True)
        (self.root / "user").mkdir(parents=True, exist_ok=True)
        (self.root / "staging" / "inbox").mkdir(parents=True, exist_ok=True)
        self._meta_root.mkdir(parents=True, exist_ok=True)
        self._versions_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Core path helpers
    # ---------------------------------------------------------------------

    def _abs_memory_path(self, rel_path: str) -> Path:
        rel = _ensure_relative_posix(rel_path)
        return self.root / Path(rel)

    def _abs_meta_path(self, rel_path: str) -> Path:
        rel = _ensure_relative_posix(rel_path)
        return self._meta_root / (rel.replace("/", "__") + ".json")

    def _abs_versions_dir(self, rel_path: str) -> Path:
        rel = _ensure_relative_posix(rel_path)
        return self._versions_root / rel.replace("/", "__")

    def _assert_can_write(self, rel_path: str) -> None:
        rel = _ensure_relative_posix(rel_path)
        if any(rel.startswith(p) for p in self.READ_ONLY_PREFIXES):
            raise AccessDenied(f"Writes to {rel!r} are not allowed (read-only prefix)")

    # ---------------------------------------------------------------------
    # Read / list
    # ---------------------------------------------------------------------

    def exists(self, rel_path: str) -> bool:
        return self._abs_memory_path(rel_path).exists()

    def read_text(self, rel_path: str, default: str = "") -> str:
        p = self._abs_memory_path(rel_path)
        try:
            return p.read_text(encoding="utf-8")
        except FileNotFoundError:
            return default

    def read_json(self, rel_path: str, default: Any = None) -> Any:
        text = self.read_text(rel_path, default="")
        if not text.strip():
            return default
        return json.loads(text)

    def list_paths(self, prefix: str = "") -> list[str]:
        """List memory file paths under a prefix (relative POSIX paths)."""
        prefix_norm = "" if not prefix else _ensure_relative_posix(prefix)
        base = self.root / Path(prefix_norm)
        if not base.exists():
            return []

        out: list[str] = []
        for p in base.rglob("*"):
            if p.is_dir():
                continue

            rel = p.relative_to(self.root).as_posix()
            # Exclude internal bookkeeping.
            if rel.startswith("audit/") or rel.startswith("meta/") or rel.startswith("versions/"):
                continue
            out.append(rel)

        out.sort()
        return out

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------

    def get_meta(self, rel_path: str) -> Optional[MemoryMeta]:
        mp = self._abs_meta_path(rel_path)
        if not mp.exists():
            return None
        try:
            raw = json.loads(mp.read_text(encoding="utf-8"))
            return MemoryMeta(**raw)
        except Exception as e:
            raise MemoryStoreError(f"Failed to read meta for {rel_path!r}: {e}")

    def current_sha(self, rel_path: str) -> str:
        meta = self.get_meta(rel_path)
        if meta is not None:
            return meta.sha256
        if not self.exists(rel_path):
            return ""
        return _sha256_bytes(self._abs_memory_path(rel_path).read_bytes())

    # ---------------------------------------------------------------------
    # Writes (versioned + audited)
    # ---------------------------------------------------------------------

    def write_text(
        self,
        rel_path: str,
        content: str,
        *,
        actor: str = "system",
        reason: str = "",
        expected_sha: Optional[str] = None,
        op: str = "update",
        extra_audit: Optional[dict[str, Any]] = None,
    ) -> MemoryMeta:
        """Write a memory file with optimistic concurrency + audit/versioning.

        expected_sha: if provided, the write is applied only if the current sha matches.
        """
        self.ensure_layout()

        rel = _ensure_relative_posix(rel_path)
        self._assert_can_write(rel)

        p = self._abs_memory_path(rel)
        old_sha = self.current_sha(rel)
        if expected_sha is not None and expected_sha != old_sha:
            raise ConcurrencyError(rel, expected_sha=expected_sha, actual_sha=old_sha)

        p.parent.mkdir(parents=True, exist_ok=True)

        new_sha = _sha256_text(content)
        p.write_text(content, encoding="utf-8")

        # Immutable version snapshot.
        versions_dir = self._abs_versions_dir(rel)
        versions_dir.mkdir(parents=True, exist_ok=True)
        ts = _utc_ts().replace(":", "-")
        version_path = versions_dir / f"{ts}_{new_sha}.txt"
        # Avoid accidental overwrite when timestamps collide.
        if version_path.exists():
            salt = _sha256_bytes(os.urandom(8))[:8]
            version_path = versions_dir / f"{ts}_{new_sha}_{salt}.txt"
        version_path.write_text(content, encoding="utf-8")

        meta = MemoryMeta(
            path=rel,
            sha256=new_sha,
            updated_at=_utc_ts(),
            updated_by=actor,
            reason=reason,
            bytes=len(content.encode("utf-8")),
        )
        self._abs_meta_path(rel).parent.mkdir(parents=True, exist_ok=True)
        self._abs_meta_path(rel).write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")

        self._append_audit(
            {
                "ts": meta.updated_at,
                "op": op,
                "path": rel,
                "old_sha": old_sha,
                "new_sha": new_sha,
                "actor": actor,
                "reason": reason,
                "version_file": version_path.relative_to(self.root).as_posix(),
                **(extra_audit or {}),
            }
        )

        return meta

    def write_json(
        self,
        rel_path: str,
        obj: Any,
        *,
        actor: str = "system",
        reason: str = "",
        expected_sha: Optional[str] = None,
        op: str = "update",
        extra_audit: Optional[dict[str, Any]] = None,
    ) -> MemoryMeta:
        content = json.dumps(obj, indent=2, sort_keys=True)
        return self.write_text(
            rel_path,
            content,
            actor=actor,
            reason=reason,
            expected_sha=expected_sha,
            op=op,
            extra_audit=extra_audit,
        )

    # ---------------------------------------------------------------------
    # Audit
    # ---------------------------------------------------------------------

    def _append_audit(self, event: dict[str, Any]) -> None:
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def audit_tail(self, n: int = 20) -> list[dict[str, Any]]:
        if not self._audit_path.exists():
            return []
        lines = self._audit_path.read_text(encoding="utf-8").splitlines()
        tail = lines[-n:] if n > 0 else lines
        out: list[dict[str, Any]] = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out


def _self_check() -> None:
    """Fast local self-check used by CI-friendly paths in tutorial scripts."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(Path(td))
        store.ensure_layout()

        # Write user memory
        m1 = store.write_text(
            "user/preferences.txt",
            "pref=on",
            actor="self-check",
            reason="unit",
        )
        assert m1.sha256 == store.current_sha("user/preferences.txt")

        # Concurrency
        try:
            store.write_text(
                "user/preferences.txt",
                "pref=off",
                actor="self-check",
                reason="unit",
                expected_sha="deadbeef",
            )
            raise AssertionError("Expected ConcurrencyError")
        except ConcurrencyError:
            pass

        # Access control — memory/ is machine-managed and read-only via the store
        try:
            store.write_text("memory/docs.txt", "nope", actor="x", reason="x")
            raise AssertionError("Expected AccessDenied")
        except AccessDenied:
            pass

        assert store.audit_tail(5), "Expected audit events"


if __name__ == "__main__":
    _self_check()
    print("memory_store self-check: ok")
