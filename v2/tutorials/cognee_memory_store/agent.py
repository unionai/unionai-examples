"""Agent helpers for the Cognee + Flyte memory-store tutorial.

This module defines:
- A staged proposal format (untrusted writes land in staging/ first)
- A validator that rejects obvious memory-poisoning attempts
- Promotion helpers that write into user/ via MemoryStore

The intention is to mimic Claude memory stores best practices:
- Separate untrusted staging from trusted memory
- Audit everything
- Use access modes (reference is read-only)
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

from memory_store import (
    AccessDenied,
    ConcurrencyError,
    MemoryMeta,
    MemoryStore,
    session_staging_inbox_prefix,
    session_staging_archive_prefix,
)


ProposalTarget = Literal["user"]
ProposalFormat = Literal["text", "json"]


class MemoryWriteProposal(BaseModel):
    """An untrusted candidate write.

    - Stored under staging/inbox/<id>.json
    - Must be validated before promotion.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at_s: float = Field(default_factory=lambda: time.time())

    # Where the proposal intends to write
    target: ProposalTarget
    path: str

    # The content to store
    format: ProposalFormat = "text"
    content: str

    # Optional concurrency precondition
    expected_sha256: Optional[str] = None

    # Provenance
    author: str = "unknown"  # user_id, session_id, etc.
    reason: str = ""  # why this memory is being written

    # Debuggable context (kept small)
    source_question: str = ""
    source_answer: str = ""

    # Topic classification — set at staging time, used by sleep_cycle to update topic_map
    topic_slug: Optional[str] = None

    # Session this proposal belongs to — determines storage namespace
    session: str = "default"


class ProposalDecision(BaseModel):
    ok: bool
    reason: str
    normalized_path: str = ""


_DISALLOWED_PATH_PREFIXES = (
    "audit/",
    "meta/",
    "versions/",
)


def classify_proposal_topic(
    content: str,
    source_question: str,
    topic_index: dict,
    api_key: str,
) -> Optional[str]:
    """Return the best-matching topic slug from the index, or None if no match.

    Used at proposal staging time so that sleep_cycle can link promoted user/
    memories to the right Cognee topic dataset without filename heuristics.
    """
    if not topic_index or not api_key:
        return None

    from anthropic import Anthropic

    topic_lines = "\n".join(f"  {s}: {e.get('label', s)}" for s, e in list(topic_index.items())[:20])
    prompt = (
        f"Topics:\n{topic_lines}\n\n"
        f"Question context: {source_question[:500]}\n\n"
        f"Memory content:\n{content[:1000]}\n\n"
        "Which topic slug does this memory belong to? "
        "Return the exact slug string only, or null if none match well.\n"
        "Return JSON only: \"topic_slug_name\" or null"
    )
    try:
        client = Anthropic(api_key=api_key, timeout=15.0)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=40,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip().strip('"').lower()
        return raw if raw in topic_index else None
    except Exception:
        return None


def proposal_inbox_path(proposal_id: str, session: str = "default") -> str:
    return f"{session_staging_inbox_prefix(session)}/{proposal_id}.json"


def proposal_archive_path(proposal_id: str, decision: str, session: str = "default") -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", decision)[:24] or "unknown"
    return f"{session_staging_archive_prefix(session)}/{safe}/{proposal_id}.json"


def stage_proposal(store: MemoryStore, proposal: MemoryWriteProposal) -> MemoryMeta:
    """Write a proposal into the session's staging inbox (untrusted)."""
    return store.write_json(
        proposal_inbox_path(proposal.id, proposal.session),
        proposal.model_dump(),
        actor=proposal.author,
        reason=proposal.reason or "stage-proposal",
        op="stage",
    )


def list_staged_proposals(store: MemoryStore, limit: int = 50, session: str = "default") -> list[MemoryWriteProposal]:
    paths = store.list_paths(session_staging_inbox_prefix(session))
    # Backward compat: also scan old-style staging/inbox for the default session
    if session == "default":
        for p in store.list_paths("staging/inbox"):
            if p not in paths:
                paths.append(p)
    paths = sorted(paths)[:limit]
    out: list[MemoryWriteProposal] = []
    for p in paths:
        try:
            raw = store.read_json(p, default=None)
            if not raw:
                continue
            out.append(MemoryWriteProposal(**raw))
        except Exception:
            continue
    # newest first
    out.sort(key=lambda x: x.created_at_s, reverse=True)
    return out


def _normalize_target_path(proposal: MemoryWriteProposal) -> str:
    session = proposal.session or "default"
    memories_prefix = f"user/sessions/{session}/memories/"
    path = proposal.path.lstrip("/")
    if path.startswith(memories_prefix):
        return path
    # Strip any old-style "user/" prefix before routing to session namespace
    path = path.removeprefix("user/")
    return memories_prefix + path


def validate_proposal(
    store: MemoryStore,
    proposal: MemoryWriteProposal,
) -> ProposalDecision:
    """Cheap, deterministic validator.

    This is intentionally conservative. If it rejects too often, tune it.
    """

    normalized = _normalize_target_path(proposal)

    if any(normalized.startswith(p) for p in _DISALLOWED_PATH_PREFIXES):
        return ProposalDecision(ok=False, reason="attempt to write internal paths")

    if normalized.startswith("memory/"):
        return ProposalDecision(ok=False, reason="memory/ is machine-managed, not user-writable")

    if proposal.target != "user":
        return ProposalDecision(ok=False, reason="invalid target")

    # Basic size bounds: keep memories small and focused.
    if len(proposal.content.encode("utf-8")) > 25_000:
        return ProposalDecision(ok=False, reason="memory too large; split into smaller files")

    # Poisoning / injection heuristics: don't store instructions that would hijack later prompts.
    lc = proposal.content.lower()
    suspicious_markers = [
        "ignore previous",
        "system prompt",
        "developer message",
        "you must obey",
        "exfiltrate",
        "api key",
        "password",
        "ssh-key",
    ]
    if any(m in lc for m in suspicious_markers):
        return ProposalDecision(ok=False, reason="content looks like prompt injection / secret material")

    return ProposalDecision(ok=True, reason="ok", normalized_path=normalized)


def promote_proposal(
    store: MemoryStore,
    proposal: MemoryWriteProposal,
    *,
    actor: str = "promoter",
    promotion_reason: str = "",
) -> MemoryMeta:
    """Promote a validated proposal into trusted memory.

    IMPORTANT: This mutates trusted memory. Caller should run validate_proposal first.
    """

    decision = validate_proposal(store, proposal)
    if not decision.ok:
        raise AccessDenied(f"Proposal {proposal.id} rejected: {decision.reason}")

    path = decision.normalized_path

    try:
        meta = store.write_text(
            path,
            proposal.content,
            actor=actor,
            reason=promotion_reason or proposal.reason or "promote",
            expected_sha=proposal.expected_sha256,
            op="promote",
            extra_audit={"proposal_id": proposal.id, "proposal_author": proposal.author},
        )
    except ConcurrencyError:
        raise

    return meta


def archive_proposal(
    store: MemoryStore,
    proposal: MemoryWriteProposal,
    *,
    actor: str,
    decision: str,
    note: str,
) -> None:
    """Archive the staged proposal with an audit event.

    This does NOT delete the inbox entry (keeps the tutorial simple and append-only).
    """
    store.ensure_layout()
    archive_path = proposal_archive_path(proposal.id, decision, proposal.session)
    store.write_json(
        archive_path,
        {
            **proposal.model_dump(),
            "archived_at_s": time.time(),
            "archived_by": actor,
            "decision": decision,
            "note": note,
        },
        actor=actor,
        reason=f"archive:{decision}",
        op="archive",
        extra_audit={"proposal_id": proposal.id, "decision": decision},
    )
