# piku-fixture project context

This is the piku dogfood fixture — a small intentionally-imperfect Rust library.

## Known issues to fix

- `src/stats.rs`: `mean()` returns `NaN` on an empty slice — add an empty check
- `src/parser.rs`: `split_csv()` drops empty fields — remove the `.filter()`
- `src/utils.rs`: `format_output` has no doc comment — add one

## Build command

```bash
cargo build
cargo test
```
