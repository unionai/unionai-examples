<!--
Thanks for contributing to unionai-examples! See CONTRIBUTING.md for the full
authoring and testing conventions. Scope is Flyte 2.x (v2/) — v1/ is legacy.
-->

## What this changes

<!-- Briefly describe the example/tutorial added or updated, and the docs page it
     supports (if any). -->

## Checklist

- [ ] The example is under `v2/` in the directory matching its docs location (not `v1/`).
- [ ] It uses the Flyte 2 `flyte` SDK and has a `__main__` guard + a `flyte.init...` call (so the harness discovers it).
- [ ] Complete PEP 723 header: `dependencies` pins `flyte` and every import; `main`/`params` are correct.
- [ ] `make test-local FILE=<path>` passes locally, and `make test-preview` shows it discovered.
- [ ] Regions embedded by the docs are wrapped in `# {{docs-fragment ...}}` markers; any renamed/moved file or fragment referenced by the docs has been updated or flagged.
- [ ] For a tutorial: `README.md` stays on the example side of the examples-vs-product-docs boundary (demonstrates and links out, rather than duplicating product docs).
