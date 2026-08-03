# Contributing to piku

Thanks for contributing. Piku is a Rust workspace; the binary, runtime,
provider API, and tools are separate crates with a one-way dependency graph.
Read [the architecture note](docs/design.md) before changing a public boundary.

## Before you start

Open an issue before a new public API, feature, data model, or large refactor so
the scope and placement can be reviewed. Small fixes and documentation updates
do not need one.

Use the stable Rust toolchain and install `just`. Build from the repository root:

```bash
cargo build --workspace
```

## Make changes

- Read the target, its callers, nearby tests, and existing conventions first.
- Keep changes scoped to one concern and preserve unrelated work in a dirty
  tree.
- Prefer property-level tests at the changed boundary. Model review and live
  LLM output do not replace deterministic assertions.
- Use direct, lowercase prose. Avoid marketing language and em dashes.
- Follow the repository's scope-style commit subjects, such as
  `runtime: preserve tool results during compaction`.

## Verify

Run the canonical gate before requesting review:

```bash
just check
```

The gate is defined in `scripts/ci.sh` and shared with CI. It runs formatting,
shell self-tests, Clippy with warnings denied, deterministic workspace tests,
isolated PTY smoke tests, and a release build.

Live LLM suites are intentionally separate. Run a relevant `just live*`,
`just agentic-user*`, or `just playground*` recipe when the change affects a
provider or agent interaction. Include the provider/model and deterministic
acceptance evidence in the review; treat model-only findings as hypotheses.

## Pull requests

- Keep the PR focused on one concern.
- Explain the behavior change and the evidence that verifies it.
- Link the related issue when one exists.
- Wait for CI to pass before requesting merge.

## License

Piku is licensed under the MIT License. By contributing, you agree that your
contribution is licensed under the same terms.
