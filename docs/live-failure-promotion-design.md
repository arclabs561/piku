# Design: live failure promotion

status: accepted
decisions: ADR-0007
decided: 2026-06-20

## Problem

Live dogfood is useful only if its failures improve the normal development
loop. Without a promotion rule, failures can stay as one-off logs, provider
complaints, or memory. That does not make piku better.

## Chosen approach

Keep a small promotion ledger in `docs/live-failure-promotions.md`. Every
repeated live failure gets classified and assigned one next action:

- add a deterministic scripted-provider test
- add a parser or trace-schema test
- quarantine a provider/model row with a note
- open a product behavior design decision
- mark it as infrastructure-only

The live workflow is still not a release gate. The promotion ledger is the
bridge from live evidence to PR-blocking coverage.

## Non-goals

- Do not block normal PR CI on live provider behavior.
- Do not promote a single flaky result without repetition or a clear root cause.
- Do not parse model prose to classify failures.
- Do not add a new issue tracker process.

## Decision gates

- If the same failure class appears twice for the same model row, add a
  promotion row.
- If the same failure appears across two provider families, prefer a
  deterministic piku test over provider quarantine.
- If a provider row fails because of credentials, rate limits, or upstream
  outage, classify it as infrastructure-only.
