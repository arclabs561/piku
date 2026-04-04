# piku-fixture

A small Rust library used as the stable dogfood target for piku integration tests.

## What it does

- `stats`: sum, min, max, mean over integer slices
- `parser`: key=value and CSV parsing, CLI flag extraction
- `utils`: string formatting utilities

## process_batch

The `process_batch(values: &[i32]) -> String` function in `lib.rs` should:

1. Compute the sum, min, and max of `values`
2. Return a formatted string: `"sum={sum} min={min} max={max}"`
3. Return `"empty"` if `values` is empty

This function is currently stubbed with `unimplemented!()`.

## Known issues

- `stats::mean` panics on an empty slice
- `parser::split_csv` drops empty fields between consecutive commas
- `utils::format_output` is missing a doc comment
- `process_batch` is not implemented
