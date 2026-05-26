# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.1.5",
#    "cognee==1.0.7",  # 1.0.3-1.0.6 had broken/inconsistent handler defaults; 1.0.9+ adds a subprocess worker that times out in this image
#    "pydantic>=2.11.0",
#    "litellm>=1.83.0",
#    "anthropic>=0.40.0",
#    "fastembed>=0.3.0",
# ]
# main = "main"
# ///

"""Cognee + Flyte memory stores — sleep/wake workflow.

Architecture
------------
Knowledge is organised into per-topic cognee datasets ("topic_<slug>").
URL ingestion classifies content into a topic and cognifies only that dataset.
At query time a cheap Claude classifier routes the question to 1-2 relevant
datasets for targeted retrieval — no reference material injected into every prompt.

Sleep cycle  (autonomous, every 6 h via flyte.Cron):
  1. Download latest state from shared object storage
  2. Auto-promote user/ staged proposals — the validator is the only gate
  3. Cluster related user/ memories by topic prefix
  4. Consolidate each cluster in parallel via flyte.map.aio (Claude merges them)
  5. Per-topic rebuild: empty_dataset → re-add → cognify (background) → memify (background)
  6. Upload updated state; stream live HTML report to Union UI

  Flyte features in play:
  - app.py background thread  →  calls flyte.run(sleep_cycle) every 6 h
  - flyte.map.aio               →  parallel cluster consolidation across pods
  - cache="auto"                →  consolidate_cluster is idempotent on retry
  - retries=2                   →  transient failures (network, cognify) auto-retried
  - report=True                 →  live HTML progress streamed to Union UI dashboard
  - flyte.group()               →  per-phase spans visible in execution timeline

Wake cycle  (on-demand, triggered per question):
  - Downloads latest consolidated state
  - Claude classifier routes question to relevant topic dataset(s)
  - Targeted cognee.search(datasets=[slugs]) for retrieved context
  - Assembles memory-augmented prompt (preferences + retrieved)
  - Calls Claude, returns answer + timing metrics

Deployment
----------
Register the sleep schedule (once per cluster):
    python workflow.py --deploy

Run the app:
    python app.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import flyte
from flyte.io import Dir

from memory_store import (
    MemoryStore,
    SESSION_REGISTRY_PATH,
    TOPIC_INDEX_PATH,
    TOPIC_MAP_PATH,
    load_topic_index,
    list_sessions,
    register_session,
    session_memories_prefix,
    session_topic_map_path,
    upsert_topic_index,
    read_topic_map,
    upsert_topic_map,
    _parse_json_object,
    _parse_json_array,
)

# Shared object-storage root for cross-run persistence.
# Override via env var to target a different cluster's bucket.
# Bare relative paths (e.g. "cognee-memory-store/memstore") are NOT supported by
# flyte.io.Dir — without a scheme `Dir.download()` treats the path as local and
# `Dir.from_local(remote_destination=...)` uploads to local pod disk.
SHARED_REMOTE_ROOT = os.environ.get(
    "COGNEE_MEMORY_STORE_REMOTE_ROOT",
    "s3://union-oc-production-persistent/cognee-memory-store",
)
SHARED_MEMSTORE_PATH = f"{SHARED_REMOTE_ROOT}/memstore"
SHARED_COGNEE_DB_PREFIX = f"{SHARED_REMOTE_ROOT}/cognee_db"
LOCAL_MEMSTORE_ROOT = Path("/tmp/memory_store")
LOCAL_COGNEE_ROOT = Path("/tmp/cognee_db")


def _topic_db_path(slug: str) -> str:
    """Remote object-storage path for a topic's isolated Cognee DB."""
    return f"{SHARED_COGNEE_DB_PREFIX}/{slug}"

GENERAL_TOPIC_SLUG = "topic_user_general"

DEFAULT_MODEL = os.environ.get("AI_MEMORY_STORE_MODEL", "claude-haiku-4-5-20251001")
# Cognee entity extraction needs a model with high output capacity (graph JSON can be large).
# Haiku has an 8192-token output ceiling; Sonnet handles denser knowledge graphs.
COGNEE_LLM_MODEL = os.environ.get("COGNEE_LLM_MODEL", "claude-sonnet-4-6")
THIS_DIR = Path(__file__).resolve().parent


_IMAGE = (
    flyte.Image.from_uv_script(__file__, name="cognee-memory-store", pre=True)
    .with_source_file(THIS_DIR / "memory_store.py", "/root")
    .with_source_file(THIS_DIR / "agent.py", "/root")
)

env = flyte.TaskEnvironment(
    name="cognee-memory-store",
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=_IMAGE,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)


# ---------------------------------------------------------------------------
# URL scraping helpers
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Minimal HTML → plain-text stripper using only stdlib."""

    _SKIP = {"script", "style", "head", "meta", "link", "noscript", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _fetch_url_text(url: str, max_bytes: int = 500_000) -> str:
    """Fetch a URL and return plain text.

    Strategy 1: Jina Reader (r.jina.ai) — handles JS-rendered pages and SPAs,
    returns clean markdown. No auth needed for public URLs.
    Strategy 2: Direct HTTP fetch with HTML stripping — fallback for sites where
    Jina is unavailable or times out.
    """
    # Strategy 1: Jina Reader
    try:
        jina_req = urllib.request.Request(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "text/plain", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(jina_req, timeout=30) as resp:
            jina_text = resp.read(max_bytes).decode("utf-8", errors="replace").strip()
        if len(jina_text) > 200:
            print(f"[ingest] fetch strategy: jina-reader ({len(jina_text):,} chars)")
            return re.sub(r'\n{3,}', '\n\n', jina_text)
    except Exception:
        pass

    print("[ingest] fetch strategy: direct-http (jina unavailable or too short)")
    # Strategy 2: Direct fetch + HTML stripping
    direct_req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(direct_req, timeout=30) as resp:
        raw = resp.read(max_bytes)
        content_type = resp.headers.get("Content-Type", "")

    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=")[-1].split(";")[0].strip() or "utf-8"
    text = raw.decode(charset, errors="replace")

    if "<html" in text.lower() or "<!doctype" in text.lower():
        extractor = _TextExtractor()
        extractor.feed(text)
        text = extractor.get_text()

    return re.sub(r'\n{3,}', '\n\n', text).strip()


class _LinkExtractor(HTMLParser):
    """Extract all href links from raw HTML using stdlib only."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() == "a":
            for name, val in attrs:
                if name == "href" and val:
                    self.links.append(val)


def _extract_links(html: str, base_url: str) -> list[str]:
    """Return deduplicated absolute URLs found in anchor tags."""
    from urllib.parse import urljoin

    extractor = _LinkExtractor()
    extractor.feed(html)
    seen: set[str] = set()
    result: list[str] = []
    for raw in extractor.links:
        absolute = urljoin(base_url, raw).split("#")[0].rstrip("/")
        if absolute and absolute not in seen:
            seen.add(absolute)
            result.append(absolute)
    return result


def _fetch_raw_html(url: str, max_bytes: int = 300_000) -> str:
    """Lightweight raw HTML fetch for link discovery (no Jina, no stripping)."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; CogneeBot/1.0)",
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read(max_bytes).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _strip_jina_header(text: str) -> str:
    """Strip Jina Reader preamble and page navigation chrome before actual content.

    Jina returns pages in two formats:
      1. With metadata block: "Title: ...\nURL Source: ...\nMarkdown Content:\n..."
      2. Without metadata: raw markdown starting directly with nav chrome

    Either way, navigation chrome (logo, nav links, banners, search bar) always
    appears before the first real section heading or code block. We scan forward
    to find where real content begins.

    A line is nav chrome if it has fewer than 3 prose words after stripping
    markdown links and images. Real content starts at the first heading
    (## or deeper, or # without "|"), code fence, or prose-dense line.
    """
    # Format 1: strip Jina metadata preamble when present
    marker = "\nMarkdown Content:\n"
    idx = text.find(marker)
    if idx >= 0:
        text = text[idx + len(marker):]

    def _is_nav_chrome(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        # Our own ingest-written comments
        if s.startswith("# Source:") or s.startswith("# (truncated"):
            return True
        # Jina's "# Page Title|Site Name" heading is always site chrome
        if s.startswith("# ") and "|" in s:
            return True
        # Any other heading or code fence marks real content — stop here
        if s.startswith("#") or s.startswith("```"):
            return False
        # Lines whose non-link text has < 3 prose words are nav chrome
        cleaned = re.sub(r"!?\[[^\]]*\]\([^)]*\)", "", s)  # remove links/images
        cleaned = re.sub(r"`[^`]*`", "", cleaned)           # remove inline code
        return len(re.findall(r"\b[a-zA-Z]{3,}\b", cleaned)) < 3

    lines = text.splitlines()
    i = 0
    # Never skip more than the first third of the file (safety bound)
    max_skip = min(len(lines), max(len(lines) // 3, 100))
    while i < max_skip and _is_nav_chrome(lines[i]):
        i += 1

    return "\n".join(lines[i:]).strip()


def _extract_links_from_markdown(markdown: str, base_url: str) -> list[str]:
    """Extract absolute URLs from Jina Reader's markdown output (inline links only)."""
    from urllib.parse import urljoin

    seen: set[str] = set()
    result: list[str] = []
    for raw in re.findall(r'\]\(([^)\s]+)\)', markdown):
        raw = raw.split("#")[0].rstrip("/")
        if not raw:
            continue
        if raw.startswith(("http://", "https://")):
            absolute = raw
        else:
            absolute = urljoin(base_url, raw).split("#")[0].rstrip("/")
        if absolute and absolute not in seen:
            seen.add(absolute)
            result.append(absolute)
    return result


def _crawl_site(seed_url: str, max_pages: int = 50) -> list[str]:
    """BFS crawl from seed_url, staying within the same domain and path prefix.

    Returns an ordered list of discovered page URLs (seed first).
    Scope is limited to URLs whose path starts with the seed's path prefix so that
    e.g. https://docs.union.ai/v2/union/ only crawls /v2/union/* pages.

    For JS-rendered sites (e.g. Mintlify, Docusaurus) raw HTML contains no <a>
    navigation tags. In those cases we fall back to Jina Reader and parse markdown
    links from the rendered output.
    """
    parsed_seed = urlparse(seed_url)
    base_domain = parsed_seed.netloc
    seed_path = parsed_seed.path
    path_prefix = seed_path[: seed_path.rfind("/") + 1] if "/" in seed_path else "/"

    visited: set[str] = set()
    queue: list[str] = [seed_url.rstrip("/")]
    discovered: list[str] = []

    while queue and len(discovered) < max_pages:
        url = queue.pop(0)
        clean = url.rstrip("/")
        if clean in visited:
            continue
        visited.add(clean)

        html = _fetch_raw_html(clean)
        if not html:
            continue

        discovered.append(clean)
        print(f"[crawl] discovered ({len(discovered)}/{max_pages}): {clean}")

        candidate_links = _extract_links(html, clean)

        # JS-rendered sites (Mintlify, Docusaurus, etc.) put navigation in React
        # bundles — raw HTML has header/footer <a> tags but none within the crawl
        # scope. Check in-scope count (not total) before deciding to fall back.
        in_scope_raw = [
            lnk for lnk in candidate_links
            if urlparse(lnk).netloc == base_domain
            and urlparse(lnk).path.startswith(path_prefix)
            and urlparse(lnk).scheme in ("http", "https")
            and lnk.rstrip("/") != clean  # exclude self
        ]
        if len(in_scope_raw) < 3:
            try:
                jina_req = urllib.request.Request(
                    f"https://r.jina.ai/{clean}",
                    headers={"Accept": "text/plain", "User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(jina_req, timeout=30) as resp:
                    jina_md = resp.read(300_000).decode("utf-8", errors="replace")
                candidate_links = _extract_links_from_markdown(jina_md, clean)
                print(f"[crawl] jina link fallback: {len(candidate_links)} link(s) on {clean}")
            except Exception as e:
                print(f"[crawl] jina link fallback failed: {e}")

        for link in candidate_links:
            p = urlparse(link)
            if (
                p.netloc == base_domain
                and p.path.startswith(path_prefix)
                and p.scheme in ("http", "https")
                and link.rstrip("/") not in visited
            ):
                queue.append(link)

    print(f"[crawl] total pages discovered: {len(discovered)}")
    return discovered


def _classify_topic(
    content_preview: str,
    url_path: str,
    existing_slugs: dict[str, str],
    api_key: str,
) -> dict:
    """Classify scraped content into a topic slug using Claude.

    Returns {"slug": "topic_x", "label": "Human Label", "is_new": bool}.
    Falls back to a slug derived from the URL path on parse failure.
    """
    from anthropic import Anthropic

    existing_lines = "\n".join(f"  {s}: {l}" for s, l in list(existing_slugs.items())[:20])
    existing_block = f"Existing topics:\n{existing_lines}" if existing_slugs else "No topics exist yet — create a new one."

    prompt = (
        f"{existing_block}\n\n"
        f"URL path: {url_path}\n\n"
        f"Content preview (first 2000 chars):\n{content_preview[:2000]}\n\n"
        "Return a JSON object with:\n"
        '  {"slug": "topic_<snake_case>", "label": "Human Readable Label", "is_new": true|false}\n'
        "Assign to an existing slug if the content clearly belongs there, otherwise create a new one.\n"
        "slug must start with topic_ and use only lowercase letters, digits, and underscores.\n"
        "Return JSON only, no explanation."
    )

    try:
        client = Anthropic(api_key=api_key, timeout=20.0)
        msg = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=120,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _parse_json_object(msg.content[0].text)
        if result and isinstance(result.get("slug"), str):
            raw_slug = result["slug"]
            slug = "topic_" + re.sub(r"[^a-z0-9]+", "_", raw_slug.lower().removeprefix("topic_")).strip("_")[:40]
            label = str(result.get("label", slug))
            is_new = slug not in existing_slugs
            return {"slug": slug, "label": label, "is_new": is_new}
    except Exception:
        pass

    # Fallback: derive slug from URL path
    slug = "topic_" + re.sub(r"[^a-z0-9]+", "_", url_path.lower()).strip("_")[:40]
    return {"slug": slug, "label": slug.replace("_", " ").title(), "is_new": slug not in existing_slugs}


def _route_query_to_topics(
    question: str,
    topic_index: dict,
    api_key: str,
) -> list[str]:
    """Return 0-2 dataset slugs most relevant to the question.

    Returns [] for general/cross-cutting questions — caller should then search
    across all datasets without scoping.
    """
    if not topic_index:
        return []

    from anthropic import Anthropic

    topic_lines = "\n".join(f"  {s}: {v.get('label', s)}" for s, v in list(topic_index.items())[:20])
    prompt = (
        f"Topics:\n{topic_lines}\n\n"
        f"Question: {question}\n\n"
        "Return a JSON array of 0-2 topic slugs most relevant to this question.\n"
        "Return [] if none match well or the question is general.\n"
        "Return JSON only, no explanation."
    )

    try:
        client = Anthropic(api_key=api_key, timeout=15.0)
        msg = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=80,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        slugs = _parse_json_array(msg.content[0].text)
        return [s for s in slugs if isinstance(s, str) and s in topic_index][:2]
    except Exception:
        return []


def _url_to_filename(url: str) -> str:
    """Convert a URL to a safe, readable filename stem."""
    parsed = urlparse(url)
    domain = re.sub(r'[^a-zA-Z0-9]', '_', parsed.netloc)
    path = re.sub(r'[^a-zA-Z0-9]', '_', parsed.path)
    name = re.sub(r'_+', '_', f"{domain}{path}").strip("_")[:80]
    return name or "scraped"


# ---------------------------------------------------------------------------
# Internal helpers  (not Flyte tasks — called within task bodies)
# ---------------------------------------------------------------------------

def _setup_cognee_env(local_cognee_root: Path = LOCAL_COGNEE_ROOT) -> None:
    """Set env vars that cognee reads at first import (before config is cached)."""
    local_cognee_root.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    os.environ["DATA_ROOT_DIRECTORY"] = str(local_cognee_root / "data")
    os.environ["SYSTEM_ROOT_DIRECTORY"] = str(local_cognee_root / "system")
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["LLM_MODEL"] = COGNEE_LLM_MODEL
    os.environ["LLM_API_KEY"] = api_key
    os.environ["EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["EMBEDDING_MODEL"] = "BAAI/bge-small-en-v1.5"
    os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")
    # Pin the graph DB provider/handler. Some cognee builds default to "ladybug"
    # which isn't in the supported_dataset_database_handlers registry, causing
    # `KeyError: 'ladybug'` during cognify. Kuzu is the embedded default that
    # actually ships in the wheel.
    os.environ["GRAPH_DATABASE_PROVIDER"] = "kuzu"
    os.environ["GRAPH_DATASET_DATABASE_HANDLER"] = "kuzu"
    # Flyte pods set LOG_LEVEL to a numeric string (e.g. "30") which cognee's
    # setup_logging() can't translate — it indexes a name→int dict with the raw
    # value and crashes with KeyError. Override with a level name cognee accepts.
    log_level_raw = os.environ.get("LOG_LEVEL", "INFO")
    if log_level_raw.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        os.environ["LOG_LEVEL"] = "INFO"


def _configure_cognee_runtime(cognee_module, local_cognee_root: Path) -> None:
    """Override cognee's cached config via its Python API after import.

    get_llm_config() and get_graph_config() are cached — env var changes after
    first import are invisible to them. This function pushes values directly into
    the live config objects, which is the only reliable way to reconfigure cognee
    when switching topic DBs within the same process (wake_cycle per-topic loop).

    llm_args passes max_tokens to work around a cognee bug: AnthropicAdapter accepts
    max_completion_tokens in its constructor but never forwards it to messages.create(),
    which requires max_tokens as a mandatory field. Without it every entity extraction
    call fails with "Missing required arguments".
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    cognee_module.config.set_llm_config({
        "llm_provider": "anthropic",
        "llm_model": COGNEE_LLM_MODEL,
        "llm_api_key": api_key,
        # 8192 is the largest Anthropic accepts for *non-streaming* requests on
        # Sonnet 4.6. Going higher trips "Streaming is required for operations
        # that may take longer than 10 minutes" from inside cognee, which uses
        # blocking calls. To keep outputs small, MAX_CHARS in ingest_url caps
        # per-page input so the generated entity-extraction JSON fits.
        "llm_args": {"max_tokens": 8192},
    })
    # system_root_directory() cascades to graph + vector + relational DB paths
    cognee_module.config.system_root_directory(str(local_cognee_root / "system"))
    cognee_module.config.set_embedding_config({
        "embedding_provider": "fastembed",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimensions": 384,
    })


async def _download_dir(d: Dir, local_path: Path) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    await d.download(local_path=str(local_path))


async def _upload_dir(local_path: Path, remote_destination: str) -> Dir:
    return await Dir.from_local(str(local_path), remote_destination=remote_destination)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_already_archived(store: MemoryStore, proposal_id: str, session: str = "default") -> bool:
    """Return True if a proposal has already been processed (promoted, rejected, or vetoed)."""
    for decision in ("approved", "rejected", "vetoed", "error", "needs_review"):
        if store.exists(f"staging/sessions/{session}/archive/{decision}/{proposal_id}.json"):
            return True
        # Backward compat: check old-style paths for default session
        if session == "default" and store.exists(f"staging/archive/{decision}/{proposal_id}.json"):
            return True
    return False


def _try_parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


async def _preserve_newest_preferences_before_upload() -> None:
    """Avoid clobbering preferences due to long-running sleep cycles.

    sleep_cycle downloads remote state at the start, mutates locally, and uploads
    at the end. If preferences were updated remotely during the run (e.g. a user
    clicked 'Save' in the app), the final upload can overwrite them.

    This function re-downloads the remote memstore right before upload and keeps
    the newer copy of user/preferences.json + user/preferences.txt.
    """
    with tempfile.TemporaryDirectory() as td:
        remote_root = Path(td) / "memstore_remote"
        await _download_dir(Dir(path=SHARED_MEMSTORE_PATH), remote_root)

        remote_store = MemoryStore(remote_root)
        local_store = MemoryStore(LOCAL_MEMSTORE_ROOT)

        pref_json = "user/preferences.json"
        pref_txt = "user/preferences.txt"

        rmeta = remote_store.get_meta(pref_json)
        lmeta = local_store.get_meta(pref_json)

        rts = _try_parse_iso(rmeta.updated_at) if rmeta else None
        lts = _try_parse_iso(lmeta.updated_at) if lmeta else None

        remote_newer = False
        if rts and lts:
            remote_newer = rts > lts
        elif rts and not lts:
            remote_newer = True

        if not remote_newer:
            return

        src_json = remote_root / pref_json
        src_txt = remote_root / pref_txt
        dst_json = LOCAL_MEMSTORE_ROOT / pref_json
        dst_txt = LOCAL_MEMSTORE_ROOT / pref_txt

        if src_json.exists():
            dst_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_json, dst_json)
        if src_txt.exists():
            dst_txt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_txt, dst_txt)


def _cluster_user_memories(store: MemoryStore, session: str = "default") -> list[dict]:
    """Group a session's promoted memories by topic prefix for parallel consolidation.

    Groups files whose stems share a common base (everything before the first '_').
    Skips JSON files (structured data). Only returns groups with 2+ members.

    Example: notes_flyte.txt + notes_tasks.txt → cluster "notes".
    """
    prefix = session_memories_prefix(session)
    paths = [p for p in store.list_paths(prefix) if not p.endswith(".json")]
    groups: dict[str, list[dict]] = {}
    for path in paths:
        stem = Path(path).stem
        label = stem.split("_")[0]
        content = store.read_text(path, default="")
        if content.strip():
            groups.setdefault(label, []).append({"path": path, "content": content})

    return [
        {"label": label, "memories": mems}
        for label, mems in groups.items()
        if len(mems) >= 2
    ]


# ---------------------------------------------------------------------------
# Init task
# ---------------------------------------------------------------------------

@env.task
async def init_memory_store() -> Dir:
    """Seed memory store with default preferences and empty topic index, upload to shared storage."""
    with flyte.group("init:setup"):
        LOCAL_MEMSTORE_ROOT.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(LOCAL_MEMSTORE_ROOT)
        store.ensure_layout()

        # Seed empty topic index — write directly since memory/ is read-only in the store
        index_path = LOCAL_MEMSTORE_ROOT / Path(TOPIC_INDEX_PATH)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if not index_path.exists():
            index_path.write_text("{}", encoding="utf-8")

        prefs_obj = {"tone": "concise", "format": "markdown"}
        store.write_json(
            "user/preferences.json", prefs_obj,
            actor="init_memory_store", reason="seed", op="create",
        )
        store.write_text(
            "user/preferences.txt",
            "\n".join(f"{k}={v}" for k, v in sorted(prefs_obj.items())) + "\n",
            actor="init_memory_store", reason="seed", op="create",
        )
        register_session(store, "default", label="Default Session")

    with flyte.group("init:upload"):
        memstore_dir = await _upload_dir(LOCAL_MEMSTORE_ROOT, SHARED_MEMSTORE_PATH)
        # No cognee_db upload — per-topic DBs are created on first ingest

    return memstore_dir


# ---------------------------------------------------------------------------
# URL ingestion task — on-demand, triggered per URL
# ---------------------------------------------------------------------------

@env.task(retries=1, timeout=timedelta(minutes=30))
async def ingest_url(url: str, max_pages: int = 10) -> tuple[Dir, Dir]:
    """Crawl a URL and all linked subpages, classify into a topic cluster, index in cognee.

    Pipeline:
      1. Crawl: BFS from seed URL, staying within the same domain + path prefix,
         up to max_pages pages. Link discovery uses raw HTML; content fetching
         uses Jina Reader (handles JS-rendered SPAs) with direct-HTTP fallback.
      2. Classify: Claude assigns or creates a topic slug from the seed page content.
      3. Write: each page → memory/topic_<slug>/<filename>.txt; update _index.json.
      4. Index: cognee.add() per page, then cognify(background) + memify(background).
      5. Upload: pod stays alive through upload, giving background cognee tasks time
         to process.

    Flyte features:
      retries=1        handles transient network / cognee failures
      timeout=30 min   allows crawling up to 50 pages via Jina Reader
    """
    with flyte.group("ingest:download"):
        await _download_dir(Dir(path=SHARED_MEMSTORE_PATH), LOCAL_MEMSTORE_ROOT)
        # Per-topic cognee DB downloaded after classification — slug is unknown until then

    LOCAL_MEMSTORE_ROOT.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(LOCAL_MEMSTORE_ROOT)
    store.ensure_layout()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    MAX_CHARS = 60_000  # captures full Jina-rendered docs page; cognify's chunk_size=512 splits it internally so each LLM call stays under the 8192-token non-streaming output ceiling

    with flyte.group("ingest:crawl"):
        print(f"[ingest] Crawling from seed: {url} (max_pages={max_pages})")
        all_urls = _crawl_site(url, max_pages=max_pages)
        if not all_urls:
            raise ValueError(f"No pages reachable from {url!r}")
        print(f"[ingest] Crawl complete — {len(all_urls)} page(s) to ingest")

    with flyte.group("ingest:classify"):
        # Classify topic using the seed page content
        print(f"[ingest] Fetching seed page for classification: {all_urls[0]}")
        seed_text = _fetch_url_text(all_urls[0])
        topic_index = load_topic_index(store)
        existing_slugs = {s: e.get("label", s) for s, e in topic_index.items()}
        parsed_seed = urlparse(url)
        url_path = parsed_seed.netloc + parsed_seed.path
        classification = _classify_topic(seed_text[:2000], url_path, existing_slugs, api_key)
        slug = classification["slug"]
        label = classification["label"]
        print(f"[ingest] classified → {slug!r} ({'new' if classification['is_new'] else 'existing'}): {label}")
        upsert_topic_index(store, slug, label=label, source_url=url, actor="ingest_url")

    # Initialize Cognee with a fresh per-topic dir. We intentionally do NOT
    # re-download the prior cognee_db from remote — when the cognee library
    # version differs from what wrote those files (e.g. Kuzu↔Ladybug schema
    # rename), downloads cause cryptic "version_code" / KeyError crashes during
    # cognify. Each ingest re-builds the graph from the source documents.
    local_cognee = Path(f"/tmp/cognee_db_{slug}")
    if local_cognee.exists():
        shutil.rmtree(local_cognee, ignore_errors=True)
    local_cognee.mkdir(parents=True, exist_ok=True)
    _setup_cognee_env(local_cognee)
    import cognee
    _configure_cognee_runtime(cognee, local_cognee)

    with flyte.group("ingest:fetch_and_write"):
        topic_dir = LOCAL_MEMSTORE_ROOT / "memory" / slug
        topic_dir.mkdir(parents=True, exist_ok=True)
        total_chars = 0

        for page_url in all_urls:
            try:
                text = _fetch_url_text(page_url) if page_url != all_urls[0] else seed_text
                if not text.strip():
                    print(f"[ingest] skip (empty): {page_url}")
                    continue

                text = _strip_jina_header(text)
                truncated = len(text) > MAX_CHARS
                text = text[:MAX_CHARS]
                content = (
                    f"# Source: {page_url}\n"
                    + (f"# (truncated to {MAX_CHARS} chars)\n" if truncated else "")
                    + f"\n{text}\n"
                )
                filename = _url_to_filename(page_url)
                file_path = topic_dir / f"{filename}.txt"
                file_path.write_text(content, encoding="utf-8")
                total_chars += len(content)
                print(f"[ingest] written memory/{slug}/{filename}.txt ({len(content):,} chars)")

                await cognee.add(f"[REFERENCE]\n{content}", dataset_name=slug)
            except Exception as e:
                print(f"[ingest] error on {page_url}: {type(e).__name__}: {e}")

        print(f"[ingest] total written: {total_chars:,} chars across {len(all_urls)} page(s)")

    with flyte.group("ingest:cognee"):
        # chunk_size caps tokens per chunk. cognee's default (~max_chunk_tokens
        # of the embedding model) is large enough that the entity-extraction
        # JSON for a dense docs page overflows the 8192-token non-streaming
        # output ceiling. 512 keeps each call's output well under that.
        print(f"[ingest] cognee.cognify(datasets=[{slug!r}], chunk_size=512) ...")
        await cognee.cognify(datasets=[slug], chunk_size=512)
        print(f"[ingest] cognify complete for {slug!r}")

    with flyte.group("ingest:upload"):
        print("[ingest] Uploading (pod stays alive for background cognee tasks) ...")
        memstore_dir = await _upload_dir(LOCAL_MEMSTORE_ROOT, SHARED_MEMSTORE_PATH)
        cognee_dir = await _upload_dir(local_cognee, _topic_db_path(slug))
        print("[ingest] Upload complete.")

    return memstore_dir, cognee_dir


# ---------------------------------------------------------------------------
# Consolidation subtask — called via flyte.map.aio inside sleep_cycle
# ---------------------------------------------------------------------------

@env.task(
    cache="auto",
    retries=1,
    timeout=timedelta(minutes=5),
)
async def consolidate_cluster(cluster_json: str) -> str:
    """Merge a cluster of related memories into one coherent summary using Claude.

    Accepts JSON: {"label": str, "memories": [{"path": str, "content": str}]}
    Returns JSON: {"path": str, "content": str, "merged_from": [str]}

    cache="auto": if this cluster's content hasn't changed since the last sleep
    cycle (e.g. after a crash/retry), the cached result is returned — no
    redundant Claude call, no redundant Flyte pod spin-up.
    """
    from anthropic import Anthropic

    cluster = json.loads(cluster_json)
    memories: list[dict] = cluster["memories"]

    if len(memories) == 1:
        m = memories[0]
        return json.dumps({"path": m["path"], "content": m["content"], "merged_from": []})

    label = cluster["label"]
    combined = "\n\n".join(f"--- {m['path']} ---\n{m['content']}" for m in memories)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        m = memories[0]
        return json.dumps({"path": m["path"], "content": m["content"], "merged_from": []})

    client = Anthropic(api_key=api_key, timeout=60.0)
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1200,
        system=(
            "You consolidate related memory entries into a single coherent summary. "
            "Preserve all distinct facts. Remove duplicates and contradictions (keep newest). "
            "Be concise but complete. Return only the consolidated text, no preamble."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Consolidate these {len(memories)} related memories about '{label}':\n\n"
                f"{combined}"
            ),
        }],
        temperature=0,
    )
    content = msg.content[0].text
    canonical_path = memories[0]["path"]
    merged_from = [m["path"] for m in memories[1:]]
    return json.dumps({"path": canonical_path, "content": content, "merged_from": merged_from})


# ---------------------------------------------------------------------------
# Per-topic rebuild — fanned out via flyte.map.aio inside sleep_cycle
# ---------------------------------------------------------------------------

@env.task(retries=1, timeout=timedelta(minutes=15))
async def rebuild_topic_dataset(rebuild_json: str) -> str:
    """Rebuild Cognee knowledge graph for one topic in an isolated pod.

    Accepts JSON: {"slug": str}

    Downloads the latest memstore (read-only) and the per-topic cognee DB,
    clears stale nodes, re-adds all content with source tags, then fires
    cognify + memify in the background before uploading the updated DB.

    Returns JSON: {"slug": str, "ref_docs": int, "user_docs": int}

    Using a separate pod per topic means all topics rebuild in parallel
    (concurrency=3 in the calling flyte.map.aio), and each pod only touches
    its own isolated cognee DB — no shared-DB write conflicts.
    """
    data = json.loads(rebuild_json)
    slug = data["slug"]
    local_cognee = Path(f"/tmp/cognee_db_{slug}")

    with flyte.group(f"rebuild:{slug}:download"):
        await _download_dir(Dir(path=SHARED_MEMSTORE_PATH), LOCAL_MEMSTORE_ROOT)
        await _download_dir(Dir(path=_topic_db_path(slug)), local_cognee)

    _setup_cognee_env(local_cognee)
    import cognee
    _configure_cognee_runtime(cognee, local_cognee)

    store = MemoryStore(LOCAL_MEMSTORE_ROOT)
    ref_docs = 0

    with flyte.group(f"rebuild:{slug}:index"):
        datasets_list = await cognee.datasets.list_datasets()
        ds_by_name = {ds.name: ds for ds in (datasets_list or [])}
        if slug in ds_by_name:
            await cognee.datasets.empty_dataset(ds_by_name[slug].id)

        # Reference content — authoritative ground truth from ingested URLs
        topic_dir = LOCAL_MEMSTORE_ROOT / "memory" / slug
        if topic_dir.exists():
            for fpath in sorted(topic_dir.glob("*.txt")):
                fc = fpath.read_text(encoding="utf-8")
                if fc.strip():
                    await cognee.add(f"[REFERENCE]\n{fc}", dataset_name=slug)
                    ref_docs += 1

        print(f"[rebuild] {slug}: {ref_docs} reference docs")

    with flyte.group(f"rebuild:{slug}:cognify"):
        await cognee.cognify(datasets=[slug], chunk_size=512)
        print(f"[rebuild] cognify complete for {slug!r}")

    with flyte.group(f"rebuild:{slug}:upload"):
        await _upload_dir(local_cognee, _topic_db_path(slug))

    return json.dumps({"slug": slug, "ref_docs": ref_docs})


# ---------------------------------------------------------------------------
# Sleep cycle — autonomous, scheduled every 6 hours
# ---------------------------------------------------------------------------

@env.task(
    retries=2,
    timeout=timedelta(minutes=45),
    report=True,
)
async def sleep_cycle() -> dict:
    """Autonomous memory consolidation — fired every 6 hours by the app-level scheduler.

    This task is the heart of the sleep/wake architecture. It runs with no human
    interaction; the app.py background thread calls flyte.run(sleep_cycle) every 6 hours.
    (flyte.Trigger + flyte.Cron on tasks is not used: the Union cluster does not write
    inputs.pb for triggered task executions, causing READ_FAILED on every trigger fire.)

    Pipeline (each phase visible as a flyte.group span in the Union UI timeline):
      1. Download latest state from shared object storage
      2. Auto-promote user/ staged proposals (validator is the only gate)
      3. Cluster related user/ memories by topic prefix
      4. Consolidate each cluster in parallel via flyte.map.aio
         → each cluster runs as an isolated Flyte pod
         → Claude merges related memories into coherent summaries
         → cache="auto" on consolidate_cluster skips unchanged clusters
      5. cognee.cognify() — rebuild the full knowledge graph
      6. Upload updated state to shared object storage
      7. Stream final HTML summary to Union UI report panel

    Flyte features:
      flyte.map.aio   parallel pods, one per memory cluster
      retries=2       transient failures auto-retried
      report=True     live HTML progress in Union UI
      flyte.group()   per-phase spans in execution timeline
    """
    from agent import (
        archive_proposal,
        classify_proposal_topic,
        list_staged_proposals,
        promote_proposal,
        validate_proposal,
    )

    ts = _utc_now()
    summary: dict = {
        "trigger_time": ts,
        "promoted": 0,
        "rejected": 0,
        "clusters_found": 0,
        "clusters_consolidated": 0,
        "memories_merged": 0,
        "topics_rebuilt": 0,
        "cognify_ran": False,
        "cognify_s": 0.0,
        "errors": [],
        "phase": "starting",
    }

    # Phase 1: Download latest state
    with flyte.group("sleep:download"):
        summary["phase"] = "downloading"
        await _emit_report(summary)
        await _download_dir(Dir(path=SHARED_MEMSTORE_PATH), LOCAL_MEMSTORE_ROOT)
        # No cognee_db download — each rebuild_topic_dataset pod handles its own topic DB

    store = MemoryStore(LOCAL_MEMSTORE_ROOT)
    store.ensure_layout()

    # Load topic index and api_key — used throughout the cycle
    topic_index = load_topic_index(store)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    changed_topics: set[str] = set()

    # Discover sessions — register "default" if none exist (first-time / migration)
    sessions = list_sessions(store)
    if not sessions:
        register_session(store, "default", label="Default Session")
        sessions = ["default"]

    # Phase 2: Auto-promote staged proposals for all sessions
    with flyte.group("sleep:promote"):
        summary["phase"] = "promoting"
        await _emit_report(summary)

        for session in sessions:
            staged = list_staged_proposals(store, limit=100, session=session)
            staged = [p for p in staged if not _is_already_archived(store, p.id, session)]
            user_proposals = [p for p in staged if p.target == "user"]

            for proposal in user_proposals:
                decision = validate_proposal(store, proposal)
                if decision.ok:
                    try:
                        promote_proposal(
                            store, proposal,
                            actor="sleep_cycle",
                            promotion_reason="auto-promoted by scheduled sleep cycle",
                        )
                        archive_proposal(
                            store, proposal,
                            actor="sleep_cycle", decision="approved",
                            note="auto-promoted by sleep_cycle",
                        )
                        summary["promoted"] += 1
                        # Track topic for rebuild (reference datasets may need refreshing)
                        slug = proposal.topic_slug
                        if not slug:
                            slug = classify_proposal_topic(
                                proposal.content, proposal.source_question, topic_index, api_key
                            )
                        if slug and slug in topic_index:
                            changed_topics.add(slug)
                        # Write session-scoped topic map entry
                        upsert_topic_map(
                            store, decision.normalized_path, slug,
                            topic_map_path=session_topic_map_path(session),
                        )
                    except Exception as e:
                        summary["errors"].append(f"promote:{session}:{proposal.id[:8]}:{type(e).__name__}")
                else:
                    archive_proposal(
                        store, proposal,
                        actor="sleep_cycle", decision="rejected",
                        note=decision.reason,
                    )
                    summary["rejected"] += 1

    # Phase 3: Cluster + consolidate memories across all sessions in parallel
    with flyte.group("sleep:consolidate"):
        summary["phase"] = "consolidating"
        await _emit_report(summary)

        all_clusters = []
        for session in sessions:
            session_clusters = _cluster_user_memories(store, session)
            # Embed session in cluster dict so we know which session after consolidation
            for c in session_clusters:
                c["session"] = session
            all_clusters.extend(session_clusters)

        clusters = all_clusters
        summary["clusters_found"] = len(clusters)

        if clusters:
            cluster_jsons = [json.dumps(c) for c in clusters]
            consolidated: list[str] = []

            # flyte.map.aio fans out consolidate_cluster as parallel Flyte pods.
            # concurrency=3 caps simultaneous pods to avoid overwhelming the cluster.
            async for result in flyte.map.aio(
                consolidate_cluster,
                cluster_jsons,
                concurrency=3,
                return_exceptions=True,
            ):
                if isinstance(result, Exception):
                    summary["errors"].append(f"consolidate:{type(result).__name__}:{result}")
                else:
                    consolidated.append(result)

            for result_json in consolidated:
                try:
                    result = json.loads(result_json)
                    merged_from = result.get("merged_from", [])
                    if merged_from:
                        store.write_text(
                            result["path"],
                            result["content"],
                            actor="sleep_cycle",
                            reason=f"consolidated {len(merged_from) + 1} memories",
                            op="consolidate",
                        )
                        summary["clusters_consolidated"] += 1
                        summary["memories_merged"] += len(merged_from)
                        # Resolve topic via the session-scoped topic map
                        result_session = result.get("session", "default")
                        topic_map = read_topic_map(store, topic_map_path=session_topic_map_path(result_session))
                        slug = topic_map.get(result["path"])
                        if slug and slug in topic_index:
                            changed_topics.add(slug)
                except Exception as e:
                    summary["errors"].append(f"write_consolidated:{type(e).__name__}")

    # Phase 4: Early upload — push promoted + consolidated memstore to shared storage
    # so that rebuild_topic_dataset pods download the latest state (not the pre-sleep snapshot).
    with flyte.group("sleep:early_upload"):
        summary["phase"] = "uploading_memstore"
        await _emit_report(summary)
        await _preserve_newest_preferences_before_upload()
        await _upload_dir(LOCAL_MEMSTORE_ROOT, SHARED_MEMSTORE_PATH)

    # Phase 5: Fan out per-topic Cognee rebuild as parallel Flyte pods.
    # Each rebuild_topic_dataset pod downloads the fresh memstore + its own topic DB,
    # tags content as [REFERENCE] or [USER_MEMORY], runs cognify+memify, and
    # uploads its updated topic DB. Pods are isolated — no shared-DB write conflicts.
    with flyte.group("sleep:cognify"):
        summary["phase"] = "cognifying"
        await _emit_report(summary)
        t0 = time.perf_counter()

        rebuild_results: list[str] = []
        if changed_topics:
            rebuild_jsons = [json.dumps({"slug": slug}) for slug in sorted(changed_topics)]

            async for result in flyte.map.aio(
                rebuild_topic_dataset,
                rebuild_jsons,
                concurrency=3,
                return_exceptions=True,
            ):
                if isinstance(result, Exception):
                    summary["errors"].append(f"rebuild:{type(result).__name__}:{result}")
                else:
                    rebuild_results.append(result)

        summary["topics_rebuilt"] = len(rebuild_results)
        summary["cognify_ran"] = bool(rebuild_results)
        summary["cognify_s"] = round(time.perf_counter() - t0, 2)

    # Phase 6: Final memstore upload — catches any preference changes that arrived
    # while rebuild pods were running (preference-race guard).
    with flyte.group("sleep:upload"):
        summary["phase"] = "uploading"
        await _emit_report(summary)

        await _preserve_newest_preferences_before_upload()
        await _upload_dir(LOCAL_MEMSTORE_ROOT, SHARED_MEMSTORE_PATH)
        # cognee_db not uploaded here — each rebuild_topic_dataset pod handled its own

    summary["phase"] = "complete"
    await flyte.report.replace.aio(_build_sleep_report(summary), do_flush=True)
    await flyte.report.flush.aio()

    return summary


# ---------------------------------------------------------------------------
# Chat summary — on-demand
# ---------------------------------------------------------------------------

@env.task(retries=1, timeout=timedelta(minutes=3))
async def summarize_chat_session(
    session_id: str,
    session: str = "default",
    model: str = DEFAULT_MODEL,
    max_lines: int = 200,
) -> str:
    """Summarize a chat session transcript into a short running summary.

    Reads:
      user/sessions/<session>/chat/<session_id>/transcript.jsonl
    Writes:
      user/sessions/<session>/chat/<session_id>/summary.txt

    This enables durable conversation continuity without stuffing the entire
    transcript into every prompt.
    """
    with flyte.group("summary:download"):
        await _download_dir(Dir(path=SHARED_MEMSTORE_PATH), LOCAL_MEMSTORE_ROOT)

    store = MemoryStore(LOCAL_MEMSTORE_ROOT)
    store.ensure_layout()

    transcript_path = f"user/sessions/{session}/chat/{session_id}/transcript.jsonl"
    summary_path = f"user/sessions/{session}/chat/{session_id}/summary.txt"

    transcript = store.read_text(transcript_path, default="").strip()
    if not transcript:
        return ""

    # Build a compact plain-text view for the summarizer.
    lines = transcript.splitlines()[-max_lines:]
    rendered: list[str] = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            role = str(obj.get("role", ""))
            content = str(obj.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                rendered.append(f"{role}: {content}")
        except Exception:
            continue

    excerpt = "\n".join(rendered).strip()
    if not excerpt:
        return ""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""

    from anthropic import Anthropic

    prev = store.read_text(summary_path, default="").strip()

    client = Anthropic(api_key=api_key, timeout=60.0)
    msg = client.messages.create(
        model=model,
        max_tokens=500,
        system=(
            "You maintain a running summary of a chat between a user and an assistant. "
            "Update the summary to reflect any new facts, decisions, and open questions. "
            "Keep it short and concrete. Return only the summary text."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Current summary (may be empty):\n{prev or '<<none>>'}\n\n"
                f"Recent transcript excerpt:\n{excerpt}\n"
            ),
        }],
        temperature=0,
    )
    summary = msg.content[0].text.strip()

    expected = store.current_sha(summary_path) or None
    store.write_text(
        summary_path,
        (summary + "\n") if summary else "",
        actor="summarize_chat_session",
        reason="chat-summary",
        expected_sha=expected,
        op="summarize",
        extra_audit={"chat_session_id": session_id, "session": session},
    )

    with flyte.group("summary:upload"):
        await _upload_dir(LOCAL_MEMSTORE_ROOT, SHARED_MEMSTORE_PATH)

    return summary


# ---------------------------------------------------------------------------
# Wake cycle — on-demand per question
# ---------------------------------------------------------------------------

@env.task(retries=1, timeout=timedelta(minutes=2))
async def wake_cycle(
    question: str,
    session: str = "default",
    model: str = DEFAULT_MODEL,
    search_timeout_s: float = 60.0,
    answer_timeout_s: float = 30.0,
) -> tuple[str, dict]:
    """Answer a question using the latest consolidated memory + Cognee retrieval.

    Downloads the current shared state, runs a Cognee semantic search, assembles
    a memory-augmented system prompt (preferences + reference docs + retrieved
    context), and calls Claude for the answer.

    retries=1 handles transient Cognee or Anthropic API failures.
    Each call always reads the latest state — sleep cycle consolidations are
    immediately visible to the next wake call.
    """
    with flyte.group("wake:download"):
        await _download_dir(Dir(path=SHARED_MEMSTORE_PATH), LOCAL_MEMSTORE_ROOT)
        # Per-topic cognee DBs downloaded after routing — only fetch what's needed

    store = MemoryStore(LOCAL_MEMSTORE_ROOT)

    with flyte.group("wake:retrieve"):
        prefs_obj = store.read_json("user/preferences.json", default=None)
        prefs = (
            "\n".join(f"{k}={v}" for k, v in sorted(prefs_obj.items()))
            if isinstance(prefs_obj, dict) and prefs_obj
            else store.read_text("user/preferences.txt", default="")
        )

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        topic_index = load_topic_index(store)
        target_slugs = _route_query_to_topics(question, topic_index, api_key)
        if not target_slugs:
            target_slugs = list(topic_index.keys())
        print(f"[wake] routing → {target_slugs if target_slugs else 'no topics available'}")

        # Inject raw session user memories as additional context
        session_mem_dir = LOCAL_MEMSTORE_ROOT / "user" / "sessions" / session / "memories"
        user_memory_parts: list[str] = []
        if session_mem_dir.exists():
            for fpath in sorted(session_mem_dir.glob("*.txt")):
                if fpath.name.startswith("_"):
                    continue
                try:
                    uc = fpath.read_text(encoding="utf-8")
                    if uc.strip():
                        user_memory_parts.append(f"[USER_MEMORY]\n{uc[:5000]}")
                except Exception:
                    pass
        print(f"[wake] session {session!r}: {len(user_memory_parts)} user memory file(s)")

        t0 = time.perf_counter()
        all_results: list = []
        for slug in target_slugs:
            local_cognee = Path(f"/tmp/cognee_db_{slug}")
            try:
                await _download_dir(Dir(path=_topic_db_path(slug)), local_cognee)
            except Exception as e:
                print(f"[wake] skip {slug!r}: DB not found ({type(e).__name__})")
                continue
            _setup_cognee_env(local_cognee)
            import cognee
            _configure_cognee_runtime(cognee, local_cognee)
            try:
                results = await asyncio.wait_for(
                    cognee.search(
                        query_text=question,
                        datasets=[slug],
                    ),
                    timeout=search_timeout_s,
                )
                all_results.extend(results or [])
                print(f"[wake] {slug!r}: {len(results or [])} result(s)")
            except asyncio.TimeoutError:
                print(f"[wake] {slug!r}: search timed out after {search_timeout_s}s")
            except Exception as e:
                print(f"[wake] {slug!r}: search error: {type(e).__name__}: {e}")

        def _extract_result_text(r) -> str:
            # SearchResult.search_result holds the actual content
            sr = getattr(r, "search_result", None)
            if sr is not None:
                if isinstance(sr, str):
                    return sr.strip()
                if isinstance(sr, dict):
                    return str(sr).strip()
                return str(sr).strip()
            # Fallback: check other common attrs then raw repr
            for attr in ("text", "content", "payload", "value"):
                val = getattr(r, attr, None)
                if isinstance(val, str) and val.strip():
                    return val
            s = str(r)
            return "" if (s.startswith("<") and s.endswith(">")) else s

        context_parts = [_extract_result_text(r) for r in all_results[:5]]
        cognee_ctx = "\n\n".join(p for p in context_parts if p)
        user_mem_ctx = "\n\n".join(user_memory_parts[:10])
        context = "\n\n".join(p for p in (cognee_ctx, user_mem_ctx) if p)
        print(f"[wake] context: {len(context)} chars from {len(all_results)} result(s)")
        if context:
            print(f"[wake] context preview: {context[:300]!r}")
        retrieve_s = time.perf_counter() - t0

    with flyte.group("wake:answer"):
        from anthropic import Anthropic, APITimeoutError

        if not api_key:
            return "[error] ANTHROPIC_API_KEY not set", {}

        system = (
            "You are an assistant with access to two types of retrieved memory:\n"
            "- [REFERENCE]: authoritative ground truth from ingested documents. Treat as fact.\n"
            "- [USER_MEMORY]: user-specific overrides. These take precedence over [REFERENCE] "
            "content for how this user wants things done.\n"
            "If context is empty, answer from general knowledge and say so.\n"
            "Treat the user's latest messages as authoritative for newly introduced facts.\n"
            "Treat [preferences] as requirements.\n"
            "- If preferences include name=<X>, address the user by that name in every response.\n"
            "- If preferences include tone/format, comply.\n"
            "- For other preference keys, interpret them as user directives and follow them as best you can.\n\n"
            f"[preferences]\n{prefs or '(none)'}\n\n"
            f"[retrieved]\n{context or '<<no retrieved context>>'}\n"
        )
        client = Anthropic(api_key=api_key, timeout=answer_timeout_s)
        t0 = time.perf_counter()
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=900,
                system=system,
                messages=[{"role": "user", "content": question}],
                temperature=0,
            )
            answer = msg.content[0].text
        except APITimeoutError:
            answer = f"[error] timed out after {answer_timeout_s}s"
        answer_s = time.perf_counter() - t0

    return answer, {
        "retrieve_s": round(retrieve_s, 2),
        "answer_s": round(answer_s, 2),
        "ctx_chars": len(context),
    }


# ---------------------------------------------------------------------------
# HTML report for sleep_cycle (streamed live to Union UI)
# ---------------------------------------------------------------------------

async def _emit_report(summary: dict) -> None:
    await flyte.report.replace.aio(_build_sleep_report(summary), do_flush=True)


def _build_sleep_report(summary: dict) -> str:
    phase = summary.get("phase", "starting")
    phases = ["downloading", "promoting", "consolidating", "uploading_memstore", "cognifying", "uploading", "complete"]

    steps_html = ""
    for step in phases:
        done = phases.index(step) < phases.index(phase) or phase == "complete"
        current = step == phase and phase != "complete"
        if done:
            icon, color = "✅", "#2ecc71"
        elif current:
            icon, color = "⏳", "#f39c12"
        else:
            icon, color = "○", "#aaa"
        steps_html += f'<div style="color:{color};padding:5px 0;font-size:15px">{icon} {step.capitalize()}</div>'

    errors_html = ""
    if summary.get("errors"):
        items = "".join(f"<li><code>{e}</code></li>" for e in summary["errors"])
        errors_html = f"<h3 style='color:#e74c3c'>Errors ({len(summary['errors'])})</h3><ul>{items}</ul>"

    cognify_val = (
        f'<span style="color:#2ecc71">{summary.get("cognify_s", 0):.1f}s</span>'
        if summary.get("cognify_ran") else "—"
    )

    stats = [
        (summary["promoted"], "proposals auto-promoted"),
        (summary["rejected"], "proposals rejected"),
        (summary.get("clusters_found", 0), "memory clusters found"),
        (summary["clusters_consolidated"], "clusters consolidated"),
        (summary["memories_merged"], "memories merged"),
        (summary.get("topics_rebuilt", 0), "topic datasets rebuilt"),
        (cognify_val, "cognify+memify fired"),
    ]
    stats_html = "".join(
        f'<div class="stat"><div class="stat-value">{v}</div><div class="stat-label">{l}</div></div>'
        for v, l in stats
    )

    status = "✅ Complete" if phase == "complete" else "⏳ Running…"
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 740px; margin: 40px auto; padding: 0 24px; color: #222; }}
  h1 {{ color: #1a1a2e; }}
  .steps {{ background: #f8f9fa; border-radius: 8px; padding: 16px 22px; margin: 18px 0; }}
  .stat {{ display: inline-block; background: #e8f4f8; border-radius: 8px; padding: 14px 20px; margin: 6px; text-align: center; min-width: 130px; }}
  .stat-value {{ font-size: 28px; font-weight: bold; color: #0f3460; }}
  .stat-label {{ font-size: 12px; color: #555; margin-top: 4px; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 13px; }}
  hr {{ border: none; border-top: 1px solid #eee; margin-top: 40px; }}
</style>
</head>
<body>
<h1>🌙 Sleep Cycle — {status}</h1>
<p style="color:#555">Triggered: {summary.get("trigger_time", "—")}</p>
<div class="steps">{steps_html}</div>
<div style="margin:20px 0">{stats_html}</div>
{errors_html}
<hr>
<p style="color:#aaa;font-size:12px">
  Cognee Memory Store · Flyte sleep/wake architecture<br>
  Schedule: every 6 hours, managed by app.py background thread<br>
  Deploy: <code>python app.py</code>
</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if os.environ.get("SELF_CHECK") == "true":
        from memory_store import _self_check
        _self_check()
        print("workflow self-check: ok")
        return

    print(
        "Primary entrypoint: python app.py\n"
        "To register the 6-hour sleep schedule on Union: python workflow.py --deploy\n"
    )


if __name__ == "__main__":
    import sys

    try:
        flyte.init_from_config()
    except Exception:
        pass

    if "--deploy" in sys.argv:
        flyte.deploy(env)
        print("Sleep cycle trigger registered — fires every 6 hours.")
    else:
        main()