# Cognee + Flyte “Memory Store”

This tutorial shows how to build **excellent, durable memory** for an agent by combining:

- **Cognee** → semantic memory (RAG / KG / embeddings): *retrieve relevant context*
- **Flyte v2** → orchestration + observability: *promote, cognify, persist, time, cache*
- **A file-based “memory store”** (this tutorial) → **auditable, versioned, access-controlled** durable memory

It takes inspiration from Claude’s Managed Agents “memory stores” concepts:
- many small memories (files)
- multiple stores / mounts with different access rules
- immutable version history (audit trail)
- safe edits (optimistic concurrency)
- staging vs trusted memory (prompt-injection defense)

## Why this design
If you rely only on semantic search (embeddings), it’s hard to enforce:
- “don’t overwrite shared memory accidentally”
- “don’t let untrusted inputs write persistent instructions”
- “show me exactly what changed and when”

So we add a **curated memory layer** that’s transparent and governable.

## Architecture

```
User message
   |
   |  (1) Retrieve
   |      - read user/ preferences
   |      - run cognee.search(query)
   v
LLM answer  (fast path runs inside app container)
   |
   |  (2) Optional: stage proposals (untrusted)
   v
staging/inbox/*.json   (NOT trusted)
   |
   |  (3) Flyte task: review_and_promote
   |      - validate (anti-poisoning rules)
   |      - write into user/ or shared/
   |      - version snapshot + audit log
   |      - cognee.add(promoted memory)
   v
Trusted memory (user/, shared/) + audit/ + versions/
   |
   |  (4) Flyte task: cognify_if_needed (cached)
   v
Cognee KG / indices updated
```

## Memory store layout
Everything lives under a directory that is synced via `flyte.io.Dir` to object storage.

- `reference/` **read-only** (curated docs/conventions)
- `user/` **read-write** (per-user preferences, notes)
- `shared/` **promotion-only** (team/org conventions; requires explicit approval)
- `staging/inbox/` **untrusted proposals**
- `staging/archive/` archived proposals + decisions
- `versions/` immutable snapshots of every write
- `audit/log.jsonl` append-only log of memory mutations
- `meta/` current sha256 + last update info per memory

### Access modes (Claude-inspired)
- `reference/` is **read-only** (writes rejected)
- `shared/` is **write-behind-review** (direct writes rejected; only allowed via promotion)

### Optimistic concurrency (safe edits)
Writes can include an `expected_sha256` precondition. If the on-disk sha does not match,
we fail fast to avoid clobbering another writer.

## Files
- `memory_store.py` — the memory store implementation (audit/versioning/access/concurrency)
- `agent.py` — proposal schema + staging + validation + promotion helpers
- `workflow.py` — Flyte tasks (init, answer, promote, cognify)
- `app.py` — Streamlit app (chat + memory sidebar; triggers Flyte tasks)

## Run (Union / Flyte App)

### 1) Ensure Anthropic key exists
This example expects a Union secret named `internal-anthropic-api-key` injected as `ANTHROPIC_API_KEY`.

### 2) Configure GHCR
This tutorial always publishes images to **GHCR**.

Default registry is `ghcr.io/flyteorg`. If you want to use your own org/user, set:

```bash
export AI_MEMORY_STORE_IMAGE_REGISTRY="ghcr.io/<your-org-or-username>"
# Optional: Union secret key that contains registry credentials (for private registries)
export AI_MEMORY_STORE_IMAGE_REGISTRY_SECRET="<union-secret-key>"
```

### 3) Serve the app

```bash
# Default behavior prefers local image builds if Docker is available.
# To force remote builds:
#   export AI_MEMORY_STORE_IMAGE_BUILDER=remote
uv run v2/tutorials/cognee_memory_store/app.py
```

Open the printed URL.

### 3) Try the workflow
1. Ask a question.
2. Change preferences either:
   - manually in the **Preferences** panel, or
   - by saying something like “please be concise” / “use markdown” / “use my name Adil every time you answer”.
     You’ll get a **preference approval popup**.
3. Confirm by clicking **Save preference** in the popup (or use **Save preferences** in the sidebar).
4. (Optional) Use **Advanced: stage raw proposal** + **Run promotion task** to exercise staged→promoted memory.
5. Click **Run cognify task**.
6. Ask a similar question again and observe retrieval improving.

### Debug
Set:

```bash
export AI_MEMORY_STORE_DEBUG=1
export AI_MEMORY_STORE_MODEL=claude-haiku-4-5-20251001
```

The chat will print retrieval/answer timings when debug is enabled.

## Run (local dev)
Local mode does not require Flyte/Union credentials; persistence is local-only.

```bash
streamlit run v2/tutorials/cognee_memory_store/app.py -- --server
```

You can still stage proposals and inspect audit/versioning locally, but promotion/cognify
Flyte tasks require remote object storage.

## What to look for
- **Auditability:** expand “Audit log (tail)” and see every stage/promote/archive event.
- **Versioning:** each promoted write creates a `versions/<memory>/*` immutable snapshot.
- **Safety:** staged proposals are not trusted until validated and promoted.
- **Separation of concerns:** Cognee retrieves; the memory store governs what is persisted.

## Notes / extensions
- Add an LLM-based “memory gate” validator (still stage first; only promote after approval).
- Split into multiple stores (per-user store vs shared org store) the same way Claude supports
  multiple stores with different access rules.
- Add redaction tools over `audit/` + `versions/` for compliance workflows.
