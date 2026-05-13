# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.1.5",
#    "cognee==1.0.7",  # see workflow.py for version rationale
#    "streamlit>=1.42.0",
#    "pydantic>=2.11.0",
#    "anthropic>=0.40.0",
#    "fastembed>=0.3.0",
#    "packaging>=23.0",
#    "charset-normalizer>=3.0",
# ]
# main = "main"
# ///

"""Cognee + Flyte memory stores — Streamlit app.

Run modes:
  Local dev:       streamlit run app.py
  Serve on Union:  uv run app.py

Memory flow:
  - Chat answered using Cognee semantic retrieval over promoted memories
  - After each answer, Claude proposes a memory to stage (inline card: Accept/Edit/Deny)
  - Accepted proposals → staging/inbox/ → auto-promoted on next sleep cycle
  - Sleep cycle status visible in sidebar; manual trigger available
"""

from __future__ import annotations

import asyncio
import json
import re
import os
import sys
import time
import threading
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

import flyte
import flyte.app
from flyte.io import Dir

from agent import MemoryWriteProposal, classify_proposal_topic, list_staged_proposals, stage_proposal
from workflow import (
    LOCAL_COGNEE_ROOT,
    LOCAL_MEMSTORE_ROOT,
    SHARED_COGNEE_DB_PREFIX,
    SHARED_MEMSTORE_PATH,
    GENERAL_TOPIC_SLUG,
    _setup_cognee_env,
    _route_query_to_topics,
    _topic_db_path,
    init_memory_store,
    ingest_url,
    sleep_cycle,
    summarize_chat_session,
)
from memory_store import (
    MemoryStore,
    TOPIC_INDEX_PATH,
    load_topic_index,
    upsert_topic_index,
    _parse_json_object,
    _parse_json_array,
)

MODEL = os.environ.get("AI_MEMORY_STORE_MODEL", "claude-haiku-4-5-20251001")
DEBUG = os.environ.get("AI_MEMORY_STORE_DEBUG", "").lower() in ("1", "true", "yes")

PREFS_JSON_PATH = "user/preferences.json"
PREFS_TXT_PATH = "user/preferences.txt"

CHAT_CONTEXT_MESSAGES = int(os.environ.get("AI_CHAT_CONTEXT_MESSAGES", "24"))
CHAT_TRANSCRIPT_MAX_LINES = int(os.environ.get("AI_CHAT_TRANSCRIPT_MAX_LINES", "1000"))


Scalar = str | int | float | bool


class ExtractedProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["preference", "memory"]
    reason: str

    # preference
    updates: dict[str, Scalar] | None = None

    # memory
    path: str | None = None
    content: str | None = None

    @field_validator("reason")
    @classmethod
    def _reason_nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("reason is required")
        if len(v) > 240:
            raise ValueError("reason too long")
        return v

    @field_validator("updates")
    @classmethod
    def _validate_updates(cls, v: dict[str, Any] | None) -> dict[str, Scalar] | None:
        if v is None:
            return None
        if not isinstance(v, dict) or not v:
            raise ValueError("updates must be a non-empty object")
        if len(v) > 12:
            raise ValueError("too many preference updates")

        cleaned: dict[str, Scalar] = {}
        for k, val in v.items():
            if not isinstance(k, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_\-]{0,40}", k):
                raise ValueError(f"invalid preference key: {k!r}")

            if isinstance(val, bool):
                cleaned[k] = val
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                cleaned[k] = val
            elif isinstance(val, str):
                sval = val.strip()
                if len(sval) > 400:
                    raise ValueError(f"preference value too long for {k!r}")
                cleaned[k] = sval
            else:
                raise ValueError(f"invalid preference value for {k!r} (must be scalar)")

        return cleaned

    @model_validator(mode="after")
    def _validate_shape(self) -> "ExtractedProposal":
        if self.type == "preference":
            if not self.updates:
                raise ValueError("preference requires updates")
            if self.path is not None or self.content is not None:
                raise ValueError("preference must not include path/content")
        elif self.type == "memory":
            if not self.path or not isinstance(self.path, str) or not self.path.strip():
                raise ValueError("memory requires path")
            if not self.content or not isinstance(self.content, str) or not self.content.strip():
                raise ValueError("memory requires content")
            if self.updates is not None:
                raise ValueError("memory must not include updates")
        return self

THIS_DIR = Path(__file__).resolve().parent
_file_name = Path(__file__).name

app_env = flyte.app.AppEnvironment(
    name="cognee-memory-store-chat",
    image=(
        flyte.Image.from_uv_script(__file__, name="cognee-memory-store-chat", pre=True)
        .with_source_file(THIS_DIR / "memory_store.py", "/root")
        .with_source_file(THIS_DIR / "agent.py", "/root")
        .with_source_file(THIS_DIR / "workflow.py", "/root")
    ),
    args=["streamlit", "run", _file_name, "--server.port", "8080", "--", "--server"],
    port=8080,
    scaling=flyte.app.Scaling(replicas=(1, 1)),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


# ---------------------------------------------------------------------------
# Flyte connection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared state sync
# ---------------------------------------------------------------------------

async def _reset_shared_state() -> None:
    """Wipe the shared memstore and every known per-topic cognee_db on remote storage.

    Old runs can leave behind cognee_db files keyed to an older library/schema
    version (e.g. corrupted Kuzu/Ladybug state), which then crashes the next
    ingest with "Could not map version_code". To avoid that, we overwrite each
    remote prefix with an empty directory at app startup.

    We read the topic index BEFORE wiping memstore so we know which topic DBs
    exist. For a brand-new install (no remote memstore yet) this is a no-op for
    cognee_db and the memstore wipe is also a no-op upload.
    """
    import tempfile

    # 1) Discover existing topic slugs from the remote memstore (best-effort).
    slugs: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as snapshot:
            await Dir(path=SHARED_MEMSTORE_PATH).download(local_path=snapshot)
            slugs = list(load_topic_index(MemoryStore(Path(snapshot))).keys())
    except Exception:
        pass  # Remote memstore doesn't exist yet — nothing to enumerate.

    # 2) Overwrite memstore + each known topic DB + general catchall with empty dirs.
    targets = [SHARED_MEMSTORE_PATH]
    targets.extend(_topic_db_path(s) for s in slugs)
    targets.append(_topic_db_path(GENERAL_TOPIC_SLUG))

    with tempfile.TemporaryDirectory() as empty:
        for path in targets:
            try:
                await Dir.from_local(empty, remote_destination=path)
            except Exception:
                pass  # Best-effort wipe; don't block app startup.

    # 3) Wipe local caches so the next download starts clean.
    if LOCAL_MEMSTORE_ROOT.exists():
        shutil.rmtree(LOCAL_MEMSTORE_ROOT, ignore_errors=True)
    if LOCAL_COGNEE_ROOT.exists():
        shutil.rmtree(LOCAL_COGNEE_ROOT, ignore_errors=True)


async def _download_shared_state() -> None:
    LOCAL_MEMSTORE_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_COGNEE_ROOT.mkdir(parents=True, exist_ok=True)
    await Dir(path=SHARED_MEMSTORE_PATH).download(local_path=str(LOCAL_MEMSTORE_ROOT))
    # Download per-topic cognee DBs based on the current topic index
    store = MemoryStore(LOCAL_MEMSTORE_ROOT)
    topic_index = load_topic_index(store)
    for slug in topic_index:
        local_cognee = LOCAL_COGNEE_ROOT / slug
        local_cognee.mkdir(parents=True, exist_ok=True)
        try:
            await Dir(path=_topic_db_path(slug)).download(local_path=str(local_cognee))
        except Exception:
            pass  # Topic DB may not exist yet if not yet ingested
    # Always download the general catchall DB (user memories not tied to any ingested topic)
    general_cognee = LOCAL_COGNEE_ROOT / GENERAL_TOPIC_SLUG
    general_cognee.mkdir(parents=True, exist_ok=True)
    try:
        await Dir(path=_topic_db_path(GENERAL_TOPIC_SLUG)).download(local_path=str(general_cognee))
    except Exception:
        pass  # General DB may not exist yet if no unclassified memories have been promoted


async def _upload_memstore() -> None:
    await Dir.from_local(str(LOCAL_MEMSTORE_ROOT), remote_destination=SHARED_MEMSTORE_PATH)


def _ensure_seeded() -> None:
    import concurrent.futures

    def _do_seed():
        # Wipe the shared memstore + cognee_db prefixes on every app start, then
        # re-init from scratch. This prevents stale/corrupted state from older
        # cognee versions (e.g. KeyError 'ladybug', kuzu version_code mismatch)
        # from breaking subsequent ingests. See _reset_shared_state for details.
        try:
            asyncio.run(_reset_shared_state())
        except Exception:
            pass  # Best-effort; the seed step below still re-initialises memstore.

        run = flyte.run(init_memory_store)
        run.wait(quiet=True)
        run.sync()
        asyncio.run(_download_shared_state())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_do_seed)
        try:
            fut.result(timeout=120)
        except (concurrent.futures.TimeoutError, Exception):
            # Let the app start even if seeding is still in progress.
            pass


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_llm(system: str, messages: list[dict], timeout_s: float = 30.0, max_tokens: int = 900) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "[error] ANTHROPIC_API_KEY not set"

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        temperature=0,
    )
    return msg.content[0].text


def _extract_proposal_from_message(user_message: str) -> dict | None:
    """Parallel Claude call — detects anything in the user's message worth staging.

    Runs concurrently with _call_llm so it adds zero latency to the response.

    Returns one of:
      {"type": "preference", "updates": {key: value, ...}, "reason": "..."}
      {"type": "memory", "content": "...", "path": "user/<name>.txt", "reason": "..."}
      None
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    import anthropic

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=25.0)
        msg = client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=(
                "You detect whether a user message contains something worth persisting as a memory or preference.\n\n"
                "Return exactly ONE of the following JSON objects (no extra keys), or exactly null.\n\n"
                "Preference schema (dynamic):\n"
                "  {\"type\":\"preference\",\"updates\":{<key>:<scalar>,...},\"reason\":<string>}\n"
                "  - updates must be a small object (1-12 entries)\n"
                "  - keys must match: [a-zA-Z][a-zA-Z0-9_\\-]{0,40}\n"
                "  - values must be JSON scalars only: string/number/boolean (no arrays/objects)\n\n"
                "Memory schema:\n"
                "  {\"type\":\"memory\",\"path\":<string>,\"content\":<string>,\"reason\":<string>}\n\n"
                "Examples:\n"
                "  User: 'I’m working on a Flyte v2 workflow for batch inference this week.'\n"
                "  Return: {\"type\":\"memory\",\"path\":\"user/projects_flyte_batch_inference.txt\",\"content\":\"User is working on a Flyte v2 workflow for batch inference this week.\",\"reason\":\"current project\"}\n\n"
                "  User: 'Always answer in bullet points.'\n"
                "  Return: {\"type\":\"preference\",\"updates\":{\"answer_style\":\"bullet_points\"},\"reason\":\"format preference\"}\n\n"
                "Skip (return null) for: questions, vague chat, or anything already covered by existing preferences.\n\n"
                "Return JSON only, no explanation, no code fences."
            ),
            messages=[{"role": "user", "content": user_message}],
            temperature=0,
        )

        raw = msg.content[0].text
        result = _parse_json_object(raw)

        try:
            proposal = ExtractedProposal.model_validate(result) if result else None
        except ValidationError as e:
            proposal = None
            err = e
        else:
            err = None

        # Debug breadcrumbs for the UI.
        try:
            import streamlit as st

            st.session_state.last_proposal_raw = raw
            st.session_state.last_proposal_parsed = result
            st.session_state.last_proposal_error = (
                (err.errors()[0].get("msg") if err else "")
                if "err" in locals() else ""
            )
        except Exception:
            pass

        if not proposal:
            return None

        out = proposal.model_dump(exclude_none=True)
        if DEBUG:
            print(f"DEBUG proposal={out}", file=sys.stderr)
        return out
    except Exception as e:
        if DEBUG:
            print(f"DEBUG _extract_proposal_from_message error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _retrieve_context(question: str, timeout_s: float = 15.0) -> str:
    store = MemoryStore(LOCAL_MEMSTORE_ROOT)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    topic_index = load_topic_index(store)
    target_slugs = _route_query_to_topics(question, topic_index, api_key)
    if not target_slugs:
        target_slugs = list(topic_index.keys())
    if GENERAL_TOPIC_SLUG not in target_slugs:
        target_slugs.append(GENERAL_TOPIC_SLUG)

    async def _search() -> str:
        import cognee
        all_results = []
        for slug in target_slugs:
            local_cognee = LOCAL_COGNEE_ROOT / slug
            if not local_cognee.exists():
                continue
            _setup_cognee_env(local_cognee)
            try:
                results = await asyncio.wait_for(
                    cognee.search(query_text=question, datasets=[slug]),
                    timeout=timeout_s,
                )
                all_results.extend(results or [])
            except (asyncio.TimeoutError, Exception):
                pass
        return "\n".join(str(r) for r in all_results[:5])

    try:
        cognee_ctx = asyncio.run(_search()).strip()
    except Exception:
        cognee_ctx = ""

    # Raw-file fallback. cognify's entity-extraction LLM calls regularly hit
    # the 8192-token non-streaming output ceiling and leave the graph empty,
    # so a clean cognee.search() can still return nothing useful. The crawler
    # always writes the scraped page text to memory/topic_<slug>/*.txt, so we
    # use those files as ground-truth context when the graph search comes back
    # short. Cap per-slug bytes to keep the prompt bounded.
    raw_parts: list[str] = []
    if len(cognee_ctx) < 400:
        per_slug_cap = 6000
        for slug in target_slugs:
            topic_dir = LOCAL_MEMSTORE_ROOT / "memory" / slug
            if not topic_dir.exists():
                continue
            for fpath in sorted(topic_dir.glob("*.txt")):
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if not text.strip():
                    continue
                raw_parts.append(f"--- {slug}/{fpath.name} ---\n{text[:per_slug_cap]}")
                if sum(len(p) for p in raw_parts) > 20_000:
                    break
            if sum(len(p) for p in raw_parts) > 20_000:
                break

    parts = [p for p in (cognee_ctx, "\n\n".join(raw_parts)) if p]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Preference helpers
# ---------------------------------------------------------------------------

def _load_prefs(store: MemoryStore) -> dict:
    obj = store.read_json(PREFS_JSON_PATH, default=None)
    if isinstance(obj, dict):
        return obj
    text = store.read_text(PREFS_TXT_PATH, default="")
    prefs: dict[str, str] = {}
    for ln in text.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            k, v = k.strip(), v.strip()
            if k:
                prefs[k] = v
    return prefs


def _prefs_to_text(prefs: dict) -> str:
    lines = [f"{k}={v}" for k, v in sorted(prefs.items()) if v is not None and v != ""]
    return "\n".join(lines).strip() + ("\n" if lines else "")


def _chat_transcript_path(session_id: str) -> str:
    return f"user/chat_sessions/{session_id}/transcript.jsonl"


def _chat_summary_path(session_id: str) -> str:
    return f"user/chat_sessions/{session_id}/summary.txt"


def _clip_text(s: str, max_chars: int = 4000) -> str:
    s = s or ""
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _build_llm_messages(history: list[dict], max_messages: int) -> list[dict]:
    # Keep only roles Anthropic expects.
    msgs = [m for m in (history or []) if m.get("role") in ("user", "assistant")]
    msgs = msgs[-max_messages:]
    out: list[dict] = []
    for m in msgs:
        out.append({"role": m["role"], "content": _clip_text(str(m.get("content", "")), 8000)})
    return out


def _append_transcript(store: MemoryStore, session_id: str, entries: list[dict], *, actor: str, reason: str) -> None:
    """Append JSONL entries to the per-session transcript (stored under user/).

    Keeps the transcript bounded so it doesn't grow forever.
    """
    store.ensure_layout()
    path = _chat_transcript_path(session_id)
    existing = store.read_text(path, default="")
    lines = existing.splitlines() if existing.strip() else []

    for e in entries:
        lines.append(json.dumps(e, ensure_ascii=False))

    if len(lines) > CHAT_TRANSCRIPT_MAX_LINES:
        lines = lines[-CHAT_TRANSCRIPT_MAX_LINES :]

    content = "\n".join(lines).strip() + ("\n" if lines else "")
    expected = store.current_sha(path) or None
    store.write_text(path, content, actor=actor, reason=reason, expected_sha=expected, op="append")


def _save_prefs(store: MemoryStore, prefs: dict, *, actor: str, reason: str) -> None:
    expected = store.current_sha(PREFS_JSON_PATH) or None
    store.write_json(PREFS_JSON_PATH, prefs, actor=actor, reason=reason, expected_sha=expected, op="update")
    try:
        expected_txt = store.current_sha(PREFS_TXT_PATH) or None
        store.write_text(
            PREFS_TXT_PATH, _prefs_to_text(prefs),
            actor=actor, reason=f"derived:{reason}", expected_sha=expected_txt, op="update",
        )
    except Exception:
        pass
    asyncio.run(_upload_memstore())




# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session() -> None:
    import streamlit as st

    defaults: dict = {
        "messages": [],
        "last_answer": "",
        "sleep_run": None,
        "sleep_run_id": "",
        "sleep_run_url": "",
        "summary_run": None,
        "summary_run_id": "",
        "summary_run_url": "",
        "ingest_run": None,
        "ingest_run_url": "",
        "promote_run": None,
        # keyed by assistant message index → proposal dict with "status": "pending"|"accepted"|"denied"
        "memory_proposals": {},
        # UI helpers
        "pending_pref_dialog": None,  # assistant message index
        "toast_queue": [],
        "chat_session_id": "",
        # Debugging
        "last_proposal_raw": "",
        "last_proposal_parsed": None,
        "last_proposal_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.get("chat_session_id"):
        st.session_state.chat_session_id = uuid.uuid4().hex[:12]


def _queue_toast(text: str, *, icon: str | None = None) -> None:
    import streamlit as st

    st.session_state.toast_queue.append({"text": text, "icon": icon})


def _drain_toasts(max_toasts: int = 3) -> None:
    import streamlit as st

    q = st.session_state.get("toast_queue") or []
    if not q:
        return

    for item in q[:max_toasts]:
        st.toast(item.get("text", ""), icon=item.get("icon"))

    st.session_state.toast_queue = []


def _maybe_open_preference_dialog(store: MemoryStore) -> None:
    import streamlit as st

    msg_idx = st.session_state.get("pending_pref_dialog")
    if msg_idx is None:
        return

    proposal = st.session_state.memory_proposals.get(msg_idx)
    if not proposal or proposal.get("status") != "pending" or proposal.get("type") != "preference":
        st.session_state.pending_pref_dialog = None
        return

    updates = proposal.get("updates", {})
    if not isinstance(updates, dict) or not updates:
        st.session_state.pending_pref_dialog = None
        return

    @st.dialog("Preference detected")
    def _dialog() -> None:
        st.caption(proposal.get("reason", ""))

        raw = st.text_area(
            "Updates (JSON)",
            value=json.dumps(updates, indent=2, sort_keys=True),
            key=f"pref_dialog_updates_{msg_idx}",
            height=180,
        )

        try:
            edited_obj = json.loads(raw)
        except Exception:
            edited_obj = None

        if isinstance(edited_obj, dict):
            st.json(edited_obj, expanded=True)
        else:
            st.warning("Updates must be valid JSON object")

        col1, col2 = st.columns(2)
        if col1.button("Save preference", type="primary", use_container_width=True):
            if not isinstance(edited_obj, dict):
                st.error("Cannot save: updates JSON must be an object")
                return

            try:
                validated = ExtractedProposal.model_validate(
                    {"type": "preference", "updates": edited_obj, "reason": proposal.get("reason", "preference")}
                )
            except ValidationError as e:
                st.error(f"Cannot save: {e.errors()[0].get('msg', 'invalid updates')}")
                return

            current = _load_prefs(store)
            current.update(validated.updates or {})
            _save_prefs(store, current, actor="chat", reason=proposal.get("reason", "preference"))
            st.session_state.memory_proposals[msg_idx]["status"] = "accepted"
            st.session_state.pending_pref_dialog = None
            _queue_toast("Preference saved", icon="⚙️")
            st.rerun()

        if col2.button("Dismiss", use_container_width=True):
            st.session_state.memory_proposals[msg_idx]["status"] = "denied"
            st.session_state.pending_pref_dialog = None
            st.rerun()

    _dialog()


# ---------------------------------------------------------------------------
# Run polling helper
# ---------------------------------------------------------------------------

def _try_finish_run(run) -> tuple[bool, str, list]:
    """Return (done, phase, outputs). outputs is empty if not done/succeeded."""
    terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"}
    try:
        run.sync()
    except Exception:
        pass
    try:
        phase = getattr(run.phase, "name", str(run.phase))
    except Exception:
        return False, "RUNNING", []
    if phase not in terminal:
        return False, phase, []
    if phase != "SUCCEEDED":
        return True, phase, []
    outs = list(run.outputs())
    if len(outs) == 1 and isinstance(outs[0], (list, tuple)):
        outs = list(outs[0])
    return True, phase, outs


# ---------------------------------------------------------------------------
# Inline memory proposal card (shown below each assistant message)
# ---------------------------------------------------------------------------

def _render_proposal_card(msg_idx: int, proposal: dict, store: MemoryStore) -> None:
    import streamlit as st

    status = proposal.get("status")
    if status != "pending":
        if status == "accepted":
            ptype = proposal.get("type", "memory")
            label = "Preference saved" if ptype == "preference" else "Memory staged — promoted on next sleep cycle"
            st.caption(f"✅ {label}")
        return

    ptype = proposal.get("type", "memory")
    is_pref = ptype == "preference"

    with st.container(border=True):
        if is_pref:
            st.caption("⚙️ Preference detected — accept to save immediately")
            updates = proposal.get("updates", {})
            st.json(updates, expanded=True)
            st.caption(proposal.get("reason", ""))
        else:
            st.caption("💡 Memory suggestion — accept to stage for the next sleep cycle")
            edited = st.text_area(
                "Content (edit before accepting)",
                value=proposal.get("content", ""),
                key=f"proposal_content_{msg_idx}",
                height=80,
                label_visibility="collapsed",
            )
            st.caption(f"`{proposal.get('path', '')}` · {proposal.get('reason', '')}")

        col1, col2, _ = st.columns([1, 1, 5])
        if col1.button("Accept", key=f"accept_{msg_idx}", type="primary"):
            if is_pref:
                current = _load_prefs(store)
                current.update(proposal.get("updates", {}))
                _save_prefs(store, current, actor="chat", reason=proposal.get("reason", "preference"))
                st.toast("Preference saved", icon="⚙️")
            else:
                _api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                _topic_slug = classify_proposal_topic(
                    content=edited,
                    source_question=proposal.get("source_question", ""),
                    topic_index=load_topic_index(store),
                    api_key=_api_key,
                )
                prop = MemoryWriteProposal(
                    target="user",
                    path=proposal["path"],
                    content=edited,
                    author="chat",
                    reason=proposal.get("reason", ""),
                    source_question=proposal.get("source_question", ""),
                    topic_slug=_topic_slug,
                )
                stage_proposal(store, prop)
                asyncio.run(_upload_memstore())
                st.toast("Memory staged — auto-promoted on next sleep cycle", icon="💡")
            st.session_state.memory_proposals[msg_idx]["status"] = "accepted"
            if st.session_state.get("pending_pref_dialog") == msg_idx:
                st.session_state.pending_pref_dialog = None
            st.rerun()
        if col2.button("Deny", key=f"deny_{msg_idx}"):
            st.session_state.memory_proposals[msg_idx]["status"] = "denied"
            if st.session_state.get("pending_pref_dialog") == msg_idx:
                st.session_state.pending_pref_dialog = None
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar sections
# ---------------------------------------------------------------------------

def _render_sleep_section() -> None:
    import streamlit as st

    st.subheader("🌙 Sleep Cycle")

    sleep_run = st.session_state.get("sleep_run")
    sleep_url = st.session_state.get("sleep_run_url", "")

    if sleep_run or sleep_url:
        # IMPORTANT: Do NOT auto-poll with run.sync() on every rerun.
        phase = (
            getattr(getattr(sleep_run, "phase", None), "name", "")
            if sleep_run else ""
        )

        url = getattr(sleep_run, "url", "") if sleep_run else sleep_url
        if url:
            st.info("Sleep cycle running")
            st.code(url, language="text")
        else:
            st.info("Sleep running…")

        col1, col2 = st.columns([1, 1])
        if col1.button("Refresh status", use_container_width=True, disabled=not bool(sleep_run)):
            try:
                done, phase, _ = _try_finish_run(sleep_run)
                if done:
                    st.session_state.sleep_run = None
                    st.session_state.sleep_run_id = ""
                    st.session_state.sleep_run_url = ""
                    if phase == "SUCCEEDED":
                        st.success("Sleep cycle complete — memory consolidated")
                        asyncio.run(_download_shared_state())
                    else:
                        st.error(f"Sleep cycle ended: {phase}")
                    st.rerun()
                else:
                    st.toast(f"Sleep still running: {phase or 'RUNNING'}")
            except Exception as e:
                st.warning(f"Could not refresh: {e}")

        if col2.button("Clear", use_container_width=True):
            st.session_state.sleep_run = None
            st.session_state.sleep_run_id = ""
            st.session_state.sleep_run_url = ""
            st.rerun()

        if not sleep_run:
            st.caption("Run handle not available in this session; open the Union UI link to monitor status.")

    else:
        st.caption("Runs every 6 hours while app is active · auto-promotes staged memories")
        if st.button("Trigger sleep now", use_container_width=True):
            try:
                run = flyte.run(sleep_cycle)
                run_url = str(getattr(run, "url", ""))
                st.session_state.sleep_run = run
                st.session_state.sleep_run_id = str(getattr(run, "id", ""))
                st.session_state.sleep_run_url = run_url
                print(f"[sleep_cycle] started: {run_url}")
                st.toast("Sleep cycle started", icon="🌙")
                st.rerun()
            except Exception as e:
                st.error(f"Could not start sleep cycle: {e}")


def _render_memory_viewer(store: MemoryStore) -> None:
    import streamlit as st

    with st.expander("📋 Audit log (tail)", expanded=False):
        for ev in store.audit_tail(30):
            st.code(
                f"{ev.get('ts')} {ev.get('op')} {ev.get('path')} actor={ev.get('actor')}",
                language="text",
            )

    topic_index = load_topic_index(store)
    with st.expander(f"📚 Topic knowledge base ({len(topic_index)} topics)", expanded=False):
        if not topic_index:
            st.caption("No topics yet — seed a URL above.")
        for slug, entry in sorted(topic_index.items()):
            label = entry.get("label", slug)
            sources = entry.get("sources", [])
            last_updated = entry.get("last_updated", "")
            with st.container(border=True):
                st.markdown(f"**{slug}** — {label}")
                st.caption(f"Updated: {last_updated}  |  Sources: {len(sources)}")
                for src in sources[:3]:
                    st.caption(f"  {src}")
                topic_paths = store.list_paths(f"memory/{slug}")
                for p in topic_paths[:5]:
                    content = store.read_text(p)
                    size_kb = len(content.encode()) / 1024
                    st.caption(f"`{p}` ({size_kb:.1f} KB)")
                    st.code(content[:400] + ("…" if len(content) > 400 else ""), language="text")

    user_paths = store.list_paths("user")

    with st.expander(f"Promoted memories ({len(user_paths)})", expanded=False):
        for p in user_paths[:30]:
            st.markdown(f"**{p}**")
            st.code(store.read_text(p)[:2000])

    def _is_archived(proposal_id: str) -> bool:
        for decision in ("approved", "rejected", "vetoed", "error", "needs_review"):
            if store.exists(f"staging/archive/{decision}/{proposal_id}.json"):
                return True
        return False

    staged = list_staged_proposals(store)
    user_staged_all = [p for p in staged if p.target == "user"]
    user_staged_pending = [p for p in user_staged_all if not _is_archived(p.id)]

    with st.expander(
        f"Staging inbox — user/ (pending {len(user_staged_pending)} · processed {len(user_staged_all) - len(user_staged_pending)})",
        expanded=False,
    ):
        if not user_staged_pending:
            st.caption("No pending staged user/ proposals.")
        for prop in user_staged_pending[:10]:
            st.markdown(f"`{prop.path}`")
            st.caption(prop.reason or "(no reason)")
            st.code(prop.content[:400])


def _render_preferences(store: MemoryStore) -> None:
    import streamlit as st

    st.subheader("Preferences")
    prefs = _load_prefs(store)

    with st.form("prefs_form"):
        tone = st.selectbox(
            "Tone",
            ["concise", "normal", "detailed"],
            index=["concise", "normal", "detailed"].index(prefs.get("tone", "concise"))
            if prefs.get("tone") in ("concise", "normal", "detailed") else 0,
        )
        fmt = st.selectbox(
            "Format", ["markdown", "plain"],
            index=0 if prefs.get("format", "markdown") == "markdown" else 1,
        )
        name = st.text_input("Name (optional)", value=str(prefs.get("name", "")))
        save = st.form_submit_button("Save preferences")

    if save:
        new_prefs = dict(prefs)
        new_prefs["tone"] = tone
        new_prefs["format"] = fmt
        if name.strip():
            new_prefs["name"] = name.strip()
        else:
            new_prefs.pop("name", None)

        _save_prefs(store, new_prefs, actor="streamlit", reason="manual-pref")
        st.toast("Saved")
        st.rerun()

    with st.expander("Advanced: stage raw proposal", expanded=False):
        with st.form("proposal_form"):
            path = st.text_input("Path", value="user/notes.txt")
            content = st.text_area("Content", height=120)
            reason = st.text_input("Reason", value="user requested")
            author = st.text_input("Author", value="streamlit")
            submitted = st.form_submit_button("Stage proposal")

        if submitted and content.strip():
            _api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            _topic_slug = classify_proposal_topic(
                content=content,
                source_question="",
                topic_index=load_topic_index(store),
                api_key=_api_key,
            )
            prop = MemoryWriteProposal(
                target="user",
                path=path.strip(),
                content=content,
                author=author,
                reason=reason,
                topic_slug=_topic_slug,
            )
            stage_proposal(store, prop)
            asyncio.run(_upload_memstore())
            st.success("Staged — auto-promoted on next sleep cycle")
            st.rerun()


def _clear_all_memory(store: MemoryStore) -> None:
    import shutil

    # Read topic slugs before clearing so we can drop each remote per-topic cognee DB
    topic_slugs = list(load_topic_index(store).keys())

    for subdir in ("user", "memory", Path("staging") / "inbox"):
        p = LOCAL_MEMSTORE_ROOT / subdir
        if p.exists():
            shutil.rmtree(p)

    if LOCAL_COGNEE_ROOT.exists():
        shutil.rmtree(LOCAL_COGNEE_ROOT)
    LOCAL_COGNEE_ROOT.mkdir(parents=True, exist_ok=True)

    store.ensure_layout()
    prefs_obj = {"tone": "concise", "format": "markdown"}
    store.write_json(PREFS_JSON_PATH, prefs_obj, actor="clear", reason="reset", op="create")
    store.write_text(
        PREFS_TXT_PATH,
        "\n".join(f"{k}={v}" for k, v in sorted(prefs_obj.items())) + "\n",
        actor="clear", reason="reset", op="create",
    )

    asyncio.run(_upload_memstore())
    # Upload empty local cognee root to each known topic's remote path (clears remote state)
    for slug in topic_slugs:
        asyncio.run(Dir.from_local(str(LOCAL_COGNEE_ROOT), remote_destination=_topic_db_path(slug)))


def _render_danger_zone(store: MemoryStore) -> None:
    import streamlit as st

    with st.expander("⚠️ Danger Zone", expanded=False):
        st.caption("Permanently deletes all memories and resets the Cognee knowledge graph. Reference docs are kept.")
        confirm = st.checkbox("I understand this cannot be undone", key="confirm_clear")
        if st.button("Clear all memory", type="primary", disabled=not confirm, use_container_width=True):
            _clear_all_memory(store)
            st.session_state.memory_proposals = {}
            st.toast("All memory cleared", icon="🗑️")
            st.rerun()


def _render_knowledge_seeding() -> None:
    import streamlit as st

    st.subheader("🌐 Seed Knowledge from URL")
    st.caption("Scrape a URL — Claude classifies it into a topic cluster and makes it retrievable by semantic search.")

    ingest_run = st.session_state.get("ingest_run")
    ingest_url_val = st.session_state.get("ingest_run_url", "")

    if ingest_run or ingest_url_val:
        url_display = getattr(ingest_run, "url", "") if ingest_run else ingest_url_val
        if url_display:
            st.info("Ingest running")
            st.code(url_display, language="text")
        else:
            st.info("Ingesting…")

        col1, col2 = st.columns([1, 1])
        if col1.button("Refresh status", use_container_width=True, key="refresh_ingest", disabled=not bool(ingest_run)):
            try:
                done, phase, _ = _try_finish_run(ingest_run)
                if done:
                    st.session_state.ingest_run = None
                    st.session_state.ingest_run_url = ""
                    if phase == "SUCCEEDED":
                        asyncio.run(_download_shared_state())
                        index = load_topic_index(MemoryStore(LOCAL_MEMSTORE_ROOT))
                        topic_summary = ", ".join(f"`{s}`" for s in sorted(index)[:8]) if index else "none yet"
                        st.success(f"URL ingested — topics: {topic_summary}")
                    else:
                        st.error(f"Ingest ended: {phase}")
                    st.rerun()
                else:
                    st.toast(f"Ingest still running: {phase or 'RUNNING'}")
            except Exception as e:
                st.warning(f"Could not refresh: {e}")

        if col2.button("Clear", use_container_width=True, key="clear_ingest"):
            st.session_state.ingest_run = None
            st.session_state.ingest_run_url = ""
            st.rerun()

        if not ingest_run:
            st.caption("Run handle not available in this session; open the Union UI link to monitor status.")
    else:
        url_input = st.text_input(
            "Seed URL",
            placeholder="https://docs.union.ai/v2/union/user-guide/",
            key="ingest_url_input",
        )
        max_pages = st.slider("Max pages to crawl", min_value=1, max_value=50, value=10, key="ingest_max_pages")
        st.caption("Crawls all linked subpages within the same domain and path prefix.")
        if st.button("Ingest URL", use_container_width=True, disabled=not bool((url_input or "").strip())):
            url = (url_input or "").strip()
            try:
                run = flyte.run(ingest_url, url=url, max_pages=max_pages)
                run_url = str(getattr(run, "url", ""))
                st.session_state.ingest_run = run
                st.session_state.ingest_run_url = run_url
                print(f"[ingest_url] started for {url!r}: {run_url}")
                st.toast(f"Ingesting {url}", icon="🌐")
                st.rerun()
            except Exception as e:
                st.error(f"Could not start ingest: {e}")


def _render_sidebar(store: MemoryStore) -> None:
    import streamlit as st

    st.header("🗂️ Memory Store")

    _render_sleep_section()
    st.divider()
    _render_knowledge_seeding()
    st.divider()

    # Chat continuity (durable transcript + optional summary)
    session_id = st.session_state.get("chat_session_id", "")
    with st.expander("💬 Chat continuity", expanded=False):
        st.caption(f"Session: `{session_id}`")
        if session_id:
            st.caption(f"Transcript: `{_chat_transcript_path(session_id)}`")
            st.caption(f"Summary: `{_chat_summary_path(session_id)}`")
            current_summary = store.read_text(_chat_summary_path(session_id), default="").strip()
            st.text_area("Current summary", value=current_summary, height=120, disabled=True)

            summary_run = st.session_state.get("summary_run")
            summary_url = st.session_state.get("summary_run_url", "")

            if summary_run or summary_url:
                url = getattr(summary_run, "url", "") if summary_run else summary_url
                if url:
                    st.caption("Summary running")
                    st.code(url, language="text")

                if st.button(
                    "Refresh summary status",
                    use_container_width=True,
                    key="refresh_summary",
                    disabled=not bool(summary_run),
                ):
                    try:
                        done, phase, _ = _try_finish_run(summary_run)
                        if done:
                            st.session_state.summary_run = None
                            st.session_state.summary_run_id = ""
                            st.session_state.summary_run_url = ""
                            if phase == "SUCCEEDED":
                                asyncio.run(_download_shared_state())
                                st.toast("Summary updated")
                            else:
                                st.error(f"Summary run ended: {phase}")
                            st.rerun()
                        else:
                            st.toast(f"Summary still running: {phase}")
                    except Exception as e:
                        st.warning(f"Could not refresh: {e}")

                if st.button("Clear summary run", use_container_width=True, key="clear_summary_run"):
                    st.session_state.summary_run = None
                    st.session_state.summary_run_id = ""
                    st.session_state.summary_run_url = ""
                    st.rerun()

                if not summary_run:
                    st.caption("Run handle not available in this session; open the Union UI link to monitor status.")

            else:
                if st.button("Update summary (Flyte)", use_container_width=True):
                    try:
                        run = flyte.run(summarize_chat_session, session_id=session_id)
                        run_url = str(getattr(run, "url", ""))
                        st.session_state.summary_run = run
                        st.session_state.summary_run_id = str(getattr(run, "id", ""))
                        st.session_state.summary_run_url = run_url
                        print(f"[summarize_chat_session] started for session {session_id!r}: {run_url}")
                        st.toast("Summary task started")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not start summary task: {e}")

    _render_memory_viewer(store)
    st.divider()
    _render_preferences(store)

    if DEBUG:
        st.divider()
        with st.expander("Debug: last proposal detection", expanded=False):
            st.caption("Raw extractor output")
            st.code(st.session_state.get("last_proposal_raw", "")[:4000])
            st.caption("Parsed JSON (if any)")
            st.json(st.session_state.get("last_proposal_parsed", None))
            err = st.session_state.get("last_proposal_error", "")
            if err:
                st.caption(f"Validation error: {err}")

    st.divider()
    _render_danger_zone(store)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def _init_flyte() -> None:
    """Initialize flyte once at startup (cached so it doesn't re-run on every Streamlit rerun)."""
    import streamlit as st

    @st.cache_resource
    def _do_init():
        flyte.init_from_config()
        return True

    _do_init()


_SLEEP_INTERVAL_S = 6 * 3600  # 6 hours


def _start_sleep_scheduler() -> None:
    """Start a daemon thread that fires sleep_cycle every 6 hours via flyte.run().

    flyte.Trigger + flyte.Cron on a task is broken on the Union cluster: the cluster
    never writes inputs.pb before starting the container, so every triggered execution
    fails with READ_FAILED. flyte.run() works correctly because the SDK uploads
    inputs.pb via the dataproxy service before creating the execution.
    """
    import streamlit as st

    @st.cache_resource
    def _start_once():
        def _loop() -> None:
            while True:
                time.sleep(_SLEEP_INTERVAL_S)
                try:
                    run = flyte.run(sleep_cycle)
                    print(f"[scheduler] sleep_cycle triggered: {getattr(run, 'url', '')}")
                except Exception as e:
                    print(f"[scheduler] sleep_cycle failed to start: {e}")

        t = threading.Thread(target=_loop, daemon=True, name="sleep-scheduler")
        t.start()
        return True

    _start_once()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Cognee Memory Store", layout="wide")

    _init_flyte()
    _start_sleep_scheduler()

    # _init_session must run before the seeded check so session_state exists
    _init_session()

    if not st.session_state.get("_seeded"):
        _ensure_seeded()
        st.session_state["_seeded"] = True

    store = MemoryStore(LOCAL_MEMSTORE_ROOT)

    with st.sidebar:
        _render_sidebar(store)

    st.title("🧠 Cognee + Flyte Memory Store")
    st.caption(
        "Sleep/wake architecture: Flyte consolidates memories every 6 hours. "
        "After each answer, Claude suggests a memory to stage — accept, edit, or deny inline."
    )

    _drain_toasts()

    # Render chat history + any pending proposal cards
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            proposal = st.session_state.memory_proposals.get(i)
            if proposal:
                _render_proposal_card(i, proposal, store)

    _maybe_open_preference_dialog(store)

    if user_input := st.chat_input("Ask a question…"):
        import concurrent.futures

        current_prefs = _load_prefs(store)

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build system prompt
        prefs_text = _prefs_to_text(current_prefs).strip()

        # Retrieve context then run answer + proposal detection in parallel
        t0 = time.perf_counter()
        context = _retrieve_context(user_input)
        t_retrieve = time.perf_counter() - t0

        session_id = st.session_state.get("chat_session_id", "")
        chat_summary = store.read_text(_chat_summary_path(session_id), default="") if session_id else ""

        system = (
            "You are an assistant. Prefer correctness over verbosity.\n"
            "Treat the user's latest messages as authoritative for newly introduced facts.\n"
            "Treat [preferences] as requirements.\n"
            "- If preferences include name=<X>, address the user by that name in every response.\n"
            "- If preferences include tone/format, comply.\n"
            "- For other preference keys, interpret them as user directives and follow them as best you can.\n"
            "When [retrieved] is non-empty, treat it as the authoritative source for the topic and answer "
            "primarily from it. If it contradicts your prior knowledge (e.g. an older framework version), "
            "follow [retrieved]. Quote API names, decorators, and code samples exactly as they appear there. "
            "If [retrieved] does not contain enough to answer, say so explicitly rather than guessing.\n\n"
            f"[preferences]\n{prefs_text or '<<none>>'}\n\n"
            f"[chat_summary]\n{chat_summary.strip() or '<<none>>'}\n\n"
            f"[retrieved]\n{context or '<<no retrieved context>>'}\n"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            t1 = time.perf_counter()
            llm_messages = _build_llm_messages(st.session_state.messages, CHAT_CONTEXT_MESSAGES)
            answer_future = pool.submit(_call_llm, system, llm_messages, 30.0)
            proposal_future = pool.submit(_extract_proposal_from_message, user_input)
            answer = answer_future.result()
            proposal = proposal_future.result()
            t_answer = time.perf_counter() - t1

        with st.chat_message("assistant"):
            st.markdown(answer)
            if DEBUG:
                st.caption(
                    f"retrieve={t_retrieve:.2f}s answer+proposal={t_answer:.2f}s ctx_chars={len(context)}"
                )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        assistant_msg_idx = len(st.session_state.messages) - 1
        st.session_state.last_answer = answer

        # Persist transcript (durable continuity across restarts).
        session_id = st.session_state.get("chat_session_id", "")
        if session_id:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _append_transcript(
                store,
                session_id,
                [
                    {"ts": now, "role": "user", "content": user_input},
                    {"ts": now, "role": "assistant", "content": answer},
                ],
                actor="chat",
                reason="chat-transcript",
            )
            asyncio.run(_upload_memstore())

        if proposal:
            # Avoid spamming proposals that don't change anything.
            if proposal.get("type") == "preference" and isinstance(proposal.get("updates"), dict):
                updates = {k: v for k, v in proposal["updates"].items() if current_prefs.get(k) != v}
                if not updates:
                    proposal = None
                else:
                    proposal["updates"] = updates

        if proposal:
            proposal["status"] = "pending"
            proposal["source_question"] = user_input
            st.session_state.memory_proposals[assistant_msg_idx] = proposal

            if proposal.get("type") == "preference":
                st.session_state.pending_pref_dialog = assistant_msg_idx
                _queue_toast("Preference detected — please review", icon="⚙️")
            else:
                _queue_toast("Memory suggestion ready to review", icon="💡")

        st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _looks_like_repo_test_runner() -> bool:
    if os.environ.get("SELF_CHECK") == "true":
        return True
    cfg = os.environ.get("FLYTECTL_CONFIG", "")
    return cfg.endswith("/test/config.flyte.yaml") or cfg.endswith("\\test\\config.flyte.yaml")


if __name__ == "__main__":
    if _looks_like_repo_test_runner():
        from memory_store import _self_check
        _self_check()
        print("app self-check: ok")
        raise SystemExit(0)

    if "--server" in sys.argv:
        # Union container entrypoint — started by app_env with --server flag
        main()
    else:
        # Deploy: register sleep schedule + serve app on Union
        flyte.init_from_config()
        from workflow import env
        flyte.deploy(env)
        print("Sleep cycle schedule registered (every 6 hours).")
        app = flyte.serve(app_env)
        print(f"App URL: {app.url}")