# Templates

Copy-me starting points for new `unionai-examples` contributions. See the
repository [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full authoring and
testing conventions.

This directory is **excluded from the test sweep** (see `test/config.json` and
`test/test_runner.py`), so these skeletons are not run as tests. Copy a template
into the appropriate `v2/` location and rename it before filling it in.

| Template | Use for |
| --- | --- |
| [`user-guide-example.py`](user-guide-example.py) | A single focused, embeddable example (one feature/pattern) under `v2/user-guide/...`. |
| [`tutorial/`](tutorial/) | An end-to-end, standalone tutorial under `v2/tutorials/<name>/` (`main.py` + `README.md`). |

## Quick start

```bash
# A user-guide example:
cp templates/user-guide-example.py v2/user-guide/<area>/<feature>.py

# A tutorial:
cp -r templates/tutorial v2/tutorials/<name>

# Then edit, and validate locally:
make test-local FILE=v2/user-guide/<area>/<feature>.py
```
