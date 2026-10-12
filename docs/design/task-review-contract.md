# Design: reviewing child task completion

Status: draft workflow; enforced acceptance remains an open contract

## Problem

A child reaching `Done` establishes that execution ended. It does not establish
that the parent's task requirements passed. Piku already records tool effects,
command verification, parent/child identities, and human run conclusions; the
missing contract is how a task's required checks and reviewed revision bind to
an accepting actor.

Independent record writers must first allocate sequences safely. A human
`conclude` call and a still-open runtime recorder must not corrupt the evidence
they are reviewing. Unknown task joins must also report missing state promptly
rather than wait for a task that cannot complete.

## Options considered

1. **Use existing human review.** Inspect the child record and workspace against
   explicit task requirements, then record `Accepted`, `NeedsWork`, or
   `Abandoned` with the existing `conclude` command. This is useful now, but
   review notes describe evidence; they do not enforce check sufficiency,
   actor identity, or revision freshness.
2. **Declare a machine-enforced task contract.** Bind named required checks,
   pass/fail/indeterminate outcomes, workspace content identity, reviewed run
   prefix, and accepting actor in typed records. This can enforce acceptance
   preconditions, but changes task/run schemas and needs a focused contract and
   compatibility review.

## Chosen approach

Advance option 1 as the initial workflow and preserve option 2 as the next
enforcement decision. Reuse `ChildRunRef`, `SubagentEvidence`, tool verification,
and `UserDisposition`; a second registry or conclusion mechanism is unnecessary.

This is an implementation sequence, not a claim that a human `Accepted` record
already proves verified task success. Runtime execution, observed verification,
and human acceptance remain separate dimensions in the audit.

## Review workflow

1. Write the actual required outcome and check commands before reviewing.
2. Follow the typed parent/child pointer to the exact child session and run;
   inspect full evidence rather than the bounded completion preview.
3. Run the required checks on the retained workspace. Preserve each command's
   exit and measured property; unavailable or incomplete verification remains
   indeterminate. A successful unrelated command does not satisfy a requirement.
4. Record the reviewed run prefix and workspace content snapshot in the review
   note. A Git HEAD alone omits dirty and untracked work. State any snapshot
   coverage limits, including uncontained shell effects.
5. Use the existing human conclusion only after examining that evidence:

   ```sh
   piku inspect CHILD_SESSION_ID --json
   piku conclude CHILD_SESSION_ID --status needs-work --note 'Required check failed; retain workspace for repair.'
   ```

   Choose `accepted` only when the human review supports it. Record the reason,
   required-check results, and evidence references. These notes are descriptive
   until typed preconditions are implemented.
6. Treat accepting a result and merging or deleting its worktree as separate
   actions. Record the task, path, branch, owner, reason, and intended disposition;
   a human conclusion does not authorize cleanup.

## Enforced-contract requirements

Before option 2 is implemented, define:

- task identity and immutable contract digest;
- stable required-check IDs and their exact measured properties;
- explicit passed, failed, unavailable, and indeterminate outcomes;
- workspace content identity including declared dirty/untracked coverage;
- the run prefix preceding acceptance, avoiding a self-referential hash;
- accepting actor, authority, and accept/reject/retain disposition;
- schema migration and rejection of absent or stale evidence.

Check descriptions and command counts alone are insufficient. Restart recovery,
notification acknowledgement, and child cancellation need separate lifecycle
tests; none can be inferred from an acceptance note.

## Decision gates

- Concurrent runtime and CLI conclusions must retain valid unique sequences and
  lossless artifact references, or fail honestly without a success receipt.
- Unknown joins fail promptly; canceled waiters cannot accumulate through
  repeated timeouts, and live/terminal joins retain their behavior.
- An enforced contract must reject missing, failed, indeterminate, stale, or
  unauthorized acceptance evidence before it is called verified completion.
- If descriptive review repeatedly fails to identify the tested revision or
  required checks, prioritize option 2 rather than adding conventions to prose.

## Why not automatic acceptance from a final answer?

The agent's answer, its terminal stop reason, successful tool count, and a human
disposition each measure different things. Automatically equating them would
invent evidence for task requirements that were never checked.

## Implementation references

- `crates/piku-runtime/src/run_record.rs`: effects, verification and run disposition.
- `crates/piku-runtime/src/task.rs`: task identity and durable child evidence.
- `crates/piku/src/main.rs`: existing `conclude` writer.
- [Fork run evidence](fork-run-evidence-design.md): typed links and bounded readers.
