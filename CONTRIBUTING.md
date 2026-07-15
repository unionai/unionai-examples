# Contributing to `unionai-examples`

Thanks for adding to the Union/Flyte examples collection. This repository holds the
**runnable, tested code** that powers the examples and tutorials on
[docs.union.ai](https://docs.union.ai). This guide covers what belongs here, how the
code is structured and tested, how it gets embedded into the docs, and the line
between an *example* and *product documentation*.

> **Scope of this guide: Flyte 2.x (`v2/`).** Everything below describes the modern
> `v2/` examples and the `uv` + Flyte 2 testing workflow. The `v1/` tree is legacy
> (Flyte 1.x / `flytekit` / `union run`) and is not covered here — don't add new
> content to it.

## Table of contents

- [What belongs in this repo (the examples-vs-docs boundary)](#what-belongs-in-this-repo)
- [Repository structure](#repository-structure)
- [Two kinds of contribution: user-guide examples vs. tutorials](#two-kinds-of-contribution)
- [Authoring conventions](#authoring-conventions)
  - [Flyte 2 SDK](#flyte-2-sdk)
  - [PEP 723 inline metadata (required)](#pep-723-inline-metadata-required)
  - [Docs-fragment markers](#docs-fragment-markers)
- [How examples are embedded into the docs](#how-examples-are-embedded-into-the-docs)
- [Testing your example](#testing-your-example)
- [Adding a new example — step by step](#adding-a-new-example)
- [Templates](#templates)
- [Pull request expectations](#pull-request-expectations)

---

## What belongs in this repo

This repo is the home of **code the reader runs**. Its canonical artifact is a
`.py` file (plus the minimum scaffolding to run it), not prose. The conceptual
explanation lives on [docs.union.ai](https://docs.union.ai) (the `unionai-docs`
repo) and *pulls code fragments from here* (see
[embedding](#how-examples-are-embedded-into-the-docs)).

**Keep examples and product documentation on their own sides of the line:**

| Belongs here (`unionai-examples`) | Belongs in product docs (`unionai-docs`) |
| --- | --- |
| Runnable, tested `.py` code | Conceptual explanation of a feature |
| A short README that says **what the example does and how to run it** | API/CLI reference — every parameter, every option |
| An end-to-end scenario a reader can execute | "How Flyte works" narrative, architecture, lifecycle |
| Inline comments explaining *this code* | Cross-feature how-to guides and tutorials' surrounding prose |

**The failure mode this guide exists to prevent:** a tutorial README that grows
into feature documentation — exhaustively documenting `TaskEnvironment`, every
caching mode, or the full CLI surface. When you feel a README turning into
reference material, **stop and move that content to the product docs**, then link
to it. The example demonstrates; the docs explain.

Rules of thumb:

- **If you're documenting an API or a concept in general terms → product docs.**
  Add it to `unionai-docs`, not to a README here.
- **If you're showing one concrete, runnable way to do a thing → here.**
- A tutorial README should read like a lab guide ("do this, run that, here's what
  you get"), not a spec.

---

## Repository structure

```
v2/                         # Flyte 2.x examples (this is where new work goes)
  user-guide/               # Small, focused examples embedded in User Guide pages
    task-configuration/
    task-programming/
    getting-started/
    ...
  integrations/             # Integration examples
    connectors/
    flyte-plugins/
  tutorials/                # End-to-end, standalone tutorials (often multi-file)
    financial_research_agent/
    trading_agents/
    ...
v1/                         # Legacy Flyte 1.x examples — do not extend
_blogs/                     # Code featured in blog posts (temporary)
test/                       # The Flyte 2 test harness (test_runner.py, configs)
templates/                  # Copy-me starting points for new contributions
```

Mirror the docs layout: a `v2/user-guide/task-configuration/...` example maps to
the corresponding User Guide page, and a `v2/tutorials/<name>/` directory maps to a
tutorial. Put new work under the directory that matches where it will appear in the
docs.

---

## Two kinds of contribution

### 1. User-guide example (a focused snippet)

A single `.py` file under `v2/user-guide/...` that demonstrates **one feature or
pattern** (retries, caching, secrets, resources, …). It is embedded — usually in
[named fragments](#docs-fragment-markers) — into a User Guide page. Keep it small,
self-contained, and heavily commented. See
[`v2/user-guide/task-configuration/retries-and-timeouts/retries.py`](v2/user-guide/task-configuration/retries-and-timeouts/retries.py)
for the reference shape.

### 2. Tutorial (an end-to-end scenario)

A directory under `v2/tutorials/<name>/` with an entrypoint (commonly `main.py`), a
`README.md`, and any supporting modules/assets. Tutorials tell a complete,
runnable story. See
[`v2/tutorials/financial_research_agent/`](v2/tutorials/financial_research_agent/)
for the reference shape (README with *what it does / requirements / secrets / run
it*, plus the runnable code).

Both are held to the same authoring and testing conventions below.

---

## Authoring conventions

### Flyte 2 SDK

Examples use the Flyte 2.x `flyte` SDK (not the legacy `flytekit`/`union` SDK).
The core shape, as seen throughout `v2/`:

```python
import flyte

# Group task configuration in a TaskEnvironment.
env = flyte.TaskEnvironment(name="my_env", resources=flyte.Resources(cpu=1, memory="250Mi"))

# Tasks are decorated with @env.task. Type annotations are required.
@env.task
def main(x: int = 5) -> int:
    return x * 2

# A runnable example ends with a __main__ guard that initializes and runs.
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, x=10)
    print(run.name)
    print(run.url)
    run.wait()
```

Conventions:

- **`import flyte`; configure via `flyte.TaskEnvironment(name=...)`; decorate tasks
  with `@env.task`.** Type annotations on task signatures are required.
- **A runnable example must have a `__main__` guard *and* a `flyte.init...` call.**
  The test harness only collects a file as a test if it contains **both**
  `if __name__ == "__main__":` **and** a `flyte.init` call (`flyte.init_from_config()`
  in cloud examples). Files without both are treated as support modules, not tests.
- Prefer `flyte.init_from_config()` in the `__main__` block so the harness can run
  the example against the configured backend.
- Use SDK constructs as the real examples do — e.g. `flyte.map`,
  `flyte.RetryStrategy` / `flyte.Backoff`, `flyte.errors.NonRecoverableError`,
  `flyte.Image.from_uv_script(__file__, ...)`. Ground new code against existing
  `v2/` examples rather than inventing patterns.

### PEP 723 inline metadata (required)

Every runnable example carries a [PEP 723](https://peps.python.org/pep-0723/) inline
script-metadata block at the top. The harness parses it to build an **isolated venv
per script** and to construct the run command:

```python
# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "numpy",
# ]
# main = "main"
# params = "x_list=[1,2,3,4,5,6,7,8,9,10]"
# ///
```

| Field | Meaning |
| --- | --- |
| `requires-python` | Python version the example targets (examples currently use `==3.13`). |
| `dependencies` | PEP 723 dependency list. **Must include `flyte`** (pinned as the other examples do, e.g. `flyte>=2.0.0b52`), plus anything the script imports. In local mode the harness runs `uv pip install --requirement <script>` from this list. |
| `main` | The name of the entrypoint task the harness runs (defaults to `main` if omitted). Set it to your entrypoint task's function name. |
| `params` | Space-separated `key=value` defaults passed to the run as `--key=value` (e.g. `params = "x_list=[1,2,3]"`). Leave as `""` if the entrypoint needs no arguments. |

### Docs-fragment markers

To embed only *part* of a file into a docs page, wrap the region in matching
fragment markers (Python comments, so they don't affect execution):

```python
# {{docs-fragment retry-count}}
@env.task(retries=3)
async def call_service() -> str:
    return await fetch_from_flaky_upstream()
# {{/docs-fragment retry-count}}
```

The docs page references the fragment by name to pull in exactly that block. Name
fragments for the concept they show (`import-and-env`, `retry-count`, `run`, …).
Keep the file runnable as a whole; fragments are a *view* over it, not a
replacement for a coherent script.

---

## How examples are embedded into the docs

Docs pages in `unionai-docs` don't copy-paste code — they reference files in this
repo, so the published snippet is always the tested source. A page either embeds a
whole file or a single [named fragment](#docs-fragment-markers). Because of this:

- **The file path and fragment names are an API.** Renaming a file or a fragment,
  or removing a `# {{docs-fragment ...}}` block, can break a docs page. If you must
  rename or move, search the `unionai-docs` repo for the old path/fragment name and
  update the references (or flag it in your PR).
- **Keep examples runnable end-to-end even when only a fragment is shown.** A reader
  who clicks through to the full file must be able to run it.

---

## Testing your example

The harness lives in `test/` and is driven by the `Makefile`. It uses **`uv`** for
all dependency management. Install `uv` first
(`curl -LsSf https://astral.sh/uv/install.sh | sh`).

**One-time setup:**

```bash
make setup-venv                 # create a uv venv at $HOME/.venv (Python 3.12)
source $HOME/.venv/bin/activate
make update-flyte               # install the latest flyte into the venv
```

**Run the tests** (scans the `v2/` tree by default):

```bash
# Local execution — isolated venv per script, `flyte run --local`. Best for dev.
make test-local FILE=v2/user-guide/getting-started/hello.py

# Preview — list what would run, execute nothing. Good for checking discovery.
make test-preview FILE=v2/user-guide/getting-started/hello.py

# Cloud execution — runs on the Union backend (needs credentials, see below).
make test FILE=v2/user-guide/getting-started/hello.py

# Filter instead of a single file:
make test-local FILTER=task-configuration
```

- **`test-local`** creates an isolated venv per script, installs its PEP 723
  dependencies with `uv pip install --requirement <script>`, and runs
  `flyte run --local <script> <main> --<param>=<value>`. No backend needed —
  **use this to validate your example before opening a PR.**
- **`test`** (cloud) uses `uv run` and executes on Union's backend. It needs a valid
  Flyte config (`test/config.flyte.yaml`, currently the Union-internal canary) and
  the `FLYTE_CLIENT_SECRET` environment variable. Examples that need extra API keys
  use Flyte secrets — see `test/create_secrets.sh`.
- **`test-preview`** just exercises discovery. If your new file doesn't appear,
  it's missing the `__main__` guard or a `flyte.init` call (see
  [conventions](#flyte-2-sdk)).

Reports land in `test/reports/` (`test_report.html`, `test_report.json`); per-script
logs in `test/logs/`. `make clean` clears them.

**CI:** `.github/workflows/test-examples.yml` runs the same `make` targets on
Python 3.13 via `uv`. It is currently **manual-trigger only** (`workflow_dispatch`):
Actions tab → *Test Examples* → *Run workflow*, choosing a mode and optional
filter/file. Because CI is manual, **run `make test-local` on your example locally
before you open the PR.**

---

## Adding a new example

1. **Decide the type and location.** A focused snippet → `v2/user-guide/<area>/...`;
   an end-to-end scenario → `v2/tutorials/<name>/`. Match the docs layout.
2. **Start from a [template](#templates).** Copy it and rename.
3. **Write the code** against the Flyte 2 SDK conventions above. Keep it minimal and
   commented; add `# {{docs-fragment ...}}` markers around any regions a docs page
   will embed.
4. **Fill in the PEP 723 header** — pin `flyte` and all imports in `dependencies`,
   set `main` to your entrypoint task, set `params` to sensible defaults (or `""`).
5. **For a tutorial**, add a `README.md` (what it does / requirements / secrets /
   how to run) using the tutorial template. Keep it a lab guide, not reference docs.
6. **Test locally:** `make test-local FILE=<your file>` (and `make test-preview` to
   confirm discovery). Fix until green.
7. **Open a PR** (see below).

---

## Templates

Copy-me starting points live in [`templates/`](templates/):

- [`templates/user-guide-example.py`](templates/user-guide-example.py) — a single
  focused, embeddable example with the PEP 723 header and docs-fragment markers.
- [`templates/tutorial/`](templates/tutorial/) — an end-to-end tutorial skeleton
  (`main.py` + `README.md`).

`templates/` is excluded from the test sweep, so the skeletons don't run as tests —
copy them into `v2/` before filling them in.

---

## Pull request expectations

Before requesting review, make sure:

- [ ] The example is under `v2/` in the directory that matches its docs location
      (not `v1/`).
- [ ] It uses the Flyte 2 `flyte` SDK and has a `__main__` guard + `flyte.init...`
      call (so the harness discovers it).
- [ ] It has a complete PEP 723 header (`dependencies` pins `flyte` and every
      import; `main`/`params` set correctly).
- [ ] `make test-local FILE=<your file>` passes locally (and `make test-preview`
      shows it discovered).
- [ ] Any embedded regions are wrapped in `# {{docs-fragment ...}}` markers; if you
      renamed/moved a file or fragment referenced by the docs, you've updated (or
      flagged) the `unionai-docs` references.
- [ ] A tutorial has a `README.md` that stays on the *example* side of the
      [examples-vs-docs boundary](#what-belongs-in-this-repo) — it demonstrates and
      links to product docs rather than duplicating them.

A PR template with this checklist is applied automatically
([`.github/pull_request_template.md`](.github/pull_request_template.md)).
