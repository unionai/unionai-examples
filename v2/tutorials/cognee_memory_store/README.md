# Cognee + Flyte Memory Store

A full-stack tutorial demonstrating a **sleep/wake memory architecture** for AI agents using [Cognee](https://github.com/topoteretes/cognee) (knowledge graph) and [Flyte v2](https://docs.union.ai/v2/).

Inspired by Claude's Managed Agents memory store concepts:
- Many small focused memory files, addressed by path
- Access control via directory prefix (`memory/` is read-only)
- Immutable version history + append-only audit log
- Staged proposals: untrusted writes land in `staging/` before promotion
- Optimistic concurrency (expected SHA-256 preconditions)

## Architecture

```
User question
     │
     │  Wake cycle (per question — Flyte task)
     │  ├─ Download shared state
     │  ├─ Route question → topic slugs (Claude classifier)
     │  ├─ cognee.search(datasets=[slugs])  ← semantic retrieval
     │  ├─ Raw-file fallback (memory/<slug>/*.txt) if graph incomplete
     │  ├─ Inject user/sessions/<session>/memories/ context
     │  └─ Claude answer with [preferences] + [retrieved]
     │
     ▼
Chat answer  ──►  Claude proposes a memory to stage  ──►  Accept/Edit/Deny
                                                              │
                                                    staging/sessions/<session>/inbox/
                                                              │
                                         Sleep cycle (every 6h — Flyte task)
                                         ├─ Auto-promote staged proposals
                                         ├─ Cluster + consolidate user memories
                                         │  (flyte.map.aio — parallel pods)
                                         ├─ Rebuild Cognee graph per topic
                                         │  (flyte.map.aio — parallel pods)
                                         └─ Upload updated state
```

**Flyte features demonstrated:**
| Feature | Where used |
|---|---|
| `flyte.map.aio` | Parallel cluster consolidation + per-topic Cognee rebuild |
| `flyte.io.Dir` | Sync memstore + Cognee DBs across pods via object storage |
| `cache="auto"` | `consolidate_cluster` skips unchanged clusters on retry |
| `report=True` | Live HTML progress streamed to Union UI during sleep cycle |
| `flyte.group()` | Per-phase spans visible in the execution timeline |
| `retries=N` | Transient network / Cognee failures auto-retried |
| `flyte.app.AppEnvironment` | Streamlit app served on Union |

## Storage layout

```
memory/                         ← READ-ONLY (written by ingest_url only)
  _index.json                   ← topic index {slug: {label, sources}}
  <topic_slug>/
    <page>.txt                  ← scraped + stripped page content

user/
  preferences.json              ← shared across sessions
  preferences.txt               ← derived plain-text copy
  sessions/
    _registry.json              ← {session_name: {created_at_s, label}}
    <session>/
      memories/
        _topic_map.json         ← {memory_path: topic_slug}
        <name>.txt              ← promoted user memories
      chat/
        <chat_id>/
          transcript.jsonl
          summary.txt

staging/
  sessions/
    <session>/
      inbox/<id>.json           ← untrusted staged proposals
      archive/<decision>/       ← archived proposals with audit note

audit/log.jsonl                 ← append-only mutation log
meta/                           ← sha256 + timestamp per memory file
versions/                       ← immutable snapshots of every write
```

## Files

| File | Purpose |
|---|---|
| `memory_store.py` | Audited, versioned file-based memory store; access control, concurrency, versioning |
| `agent.py` | Proposal schema, staging, validation (anti-injection), promotion |
| `workflow.py` | All Flyte tasks: `init_memory_store`, `ingest_url`, `consolidate_cluster`, `rebuild_topic_dataset`, `sleep_cycle`, `wake_cycle`, `summarize_chat_session` |
| `app.py` | Streamlit UI: chat, session selector, URL ingestion, sleep cycle trigger, memory viewer |

## Prerequisites

1. **Union account** — [sign up at union.ai](https://union.ai)
2. **Anthropic API key** stored as a Union secret:
   ```bash
   union create secret internal-anthropic-api-key
   # paste your key when prompted
   ```
3. **`uv`** installed — [docs.astral.sh/uv](https://docs.astral.sh/uv)

## Quickstart (Union)

```bash
# Clone and enter the repo
git clone https://github.com/unionai/unionai-examples
cd unionai-examples

# Authenticate with Union
union login

# Deploy and serve (builds image, registers sleep schedule, serves app)
uv run v2/tutorials/cognee_memory_store/app.py
```

Open the printed app URL.

### Optional: configure image registry

By default, images are pushed to `ghcr.io/flyteorg`. Override with:

```bash
export AI_MEMORY_STORE_IMAGE_REGISTRY="ghcr.io/<your-org-or-username>"
```

## Using the app

### 1. Initialize
The app seeds a fresh memory store on first launch via the `init_memory_store` Flyte task.

### 2. Ingest reference knowledge
In the sidebar under **Seed Knowledge from URL**:
- Enter a URL (e.g. `https://docs.union.ai/v2/union/user-guide/`)
- Set **Max pages** (up to 50)
- Click **Ingest URL**

The `ingest_url` task crawls the site, classifies content into a topic, writes pages to `memory/<slug>/`, and builds a Cognee knowledge graph. Check the **Topic knowledge base** expander to see ingested files.

### 3. Ask questions
Type in the chat input. The app retrieves context from Cognee + raw memory files, then calls Claude. After each answer, Claude suggests a memory to stage — accept, edit, or deny inline.

### 4. Manage sessions
Use the **Session** selector in the sidebar to create isolated sessions. Each session has its own promoted memories, staging inbox, and chat history. Reference knowledge (`memory/`) is shared across all sessions.

### 5. Sleep cycle
Click **Trigger sleep now** (or wait — it fires automatically every 6 hours):
- Staged proposals are auto-promoted to `user/sessions/<session>/memories/`
- Related memories are clustered and consolidated by Claude
- Cognee knowledge graphs are rebuilt per topic
- Live HTML progress streams to the Union UI report panel

### 6. Preferences
Under **Preferences** in the sidebar, set tone, format, and your name. Claude will follow these in every answer. You can also say things like *"always answer in bullet points"* in chat — Claude detects the preference and offers an inline approval card.

## Debug mode

```bash
export AI_MEMORY_STORE_DEBUG=1
```

Enables per-message retrieval/answer timing and proposal detection details in the UI.

## Local dev (no Union required)

```bash
streamlit run v2/tutorials/cognee_memory_store/app.py -- --server
```

Staging, promotion, and memory viewer work locally. The `wake_cycle` and `sleep_cycle` Flyte tasks require remote object storage and will not run in pure local mode.

Run the built-in self-checks:

```bash
python v2/tutorials/cognee_memory_store/memory_store.py  # storage self-check
SELF_CHECK=true python v2/tutorials/cognee_memory_store/app.py  # app self-check
```

## What to look for

- **Audit trail** — expand "Audit log (tail)" to see every stage/promote/consolidate event with timestamps and actors
- **Versioning** — every promoted write creates an immutable snapshot under `versions/`
- **Staged proposals** — the "Staging inbox" expander shows pending proposals before they're promoted
- **Parallel consolidation** — watch the sleep cycle's `flyte.map.aio` fan-out in the Union UI execution timeline
- **Raw-file fallback** — when Cognee's graph is incomplete (entity extraction can hit LLM output limits), the app falls back to reading `memory/<slug>/*.txt` directly
