# Design: Repository authority

Status: implemented; canonical checks and isolated admission probes verified.

## Overview

Piku keeps provider connections, project tool approval and executable replacement
under separate operator control. Opening a repository supplies task context and
restrictions; granting one capability does not grant the others. This design
defines the admission rules used by the [configuration](../design.md) and
[developer self-update](../self-update.md) paths.

## Problem

Opening a checkout previously let its dotenv choose provider endpoints, its
settings extend tool grants, and a newer local build replace the running Piku.
These are separate operator powers. Repository content must not acquire them
merely because Piku starts in that directory.

## Chosen approach

Provider connections belong to the operator's launching environment, global
configuration and explicit command options. Piku stops importing provider
variables from repository or ancestor `.env` files. Project settings cannot
override provider/model selection or provider connection blocks. Actor labels
and instructions remain project task data; actor connection choices remain
operator-controlled. Existing parsed provider blocks are not made executable
configuration by this change.

Project deny rules apply immediately. Project allow rules require an explicit
review and approval tied to the canonical repository root and a digest of the
permission policy. `piku permissions review` displays that root, policy and
digest; `piku permissions approve --root ROOT --digest DIGEST` approves only
that reviewed revision. Changing permission rules requires renewed review;
editing ordinary source does not. Global grants and project-hook approval keep
their existing, separate scopes. A project can lower the effective turn budget,
but cannot increase the operator's limit. Scoped shell grants must not silently
authorize additional commands through shell control syntax.

Developer self-update is disabled by default. `--self-update-root` explicitly
names an absolute Cargo target directory; only its `release/piku` candidate is
eligible. Capture a handle to that approved directory before hooks run; open the
release directory and candidate relative to it, rejecting symlinks. Copy the
opened candidate into a private staging file before replacement, so a pathname
swap cannot choose a different source. The same policy governs startup, idle
polling and tool-result updates. Read-only mode keeps updates disabled.

## Tradeoffs and alternatives

Projects that relied on dotenv provider credentials must export them explicitly
in their launch environment. Project routing moves to operator settings or
command options. Tool-rule approval adds a review step after policy changes;
default-disabled self-update needs a launch flag during self-hosting.

Global-only tool rules would be simpler but would discard useful reviewed
project capabilities. A blanket repository grant would combine credentials,
tool execution and binary replacement, and let later checkout changes inherit
unrelated authority. Per-update manual installation remains an alternative to
developer mode.

## Scope boundaries

- Approval is not OS containment: approved builds and tests execute repository
  code with the operator's permissions. Sandbox enforcement remains separate.
- Local-web client identity and PTY access retain their separate acceptance gate.
- Hook approval remains independent; this change does not redesign hook trust.
- Task acceptance, recording policy and verified child completion retain their
  existing contracts.

## Acceptance and decision gates

Synthetic loopback endpoints prove repository settings cannot redirect an
operator credential, while explicit operator routing still works. Real CLI
review/approval tests prove root binding, rule-change invalidation, persistent
denial and benign approved capabilities. Isolated executable copies prove
unapproved candidates cannot replace Piku, while opted-in updates preserve
session recovery. Run the canonical `just check` before delivery.

If scoped shell patterns cannot distinguish their reviewed capability from a
compound command, decline that automatic grant rather than infer approval.
No approval store or build-root flag is evidence of sandbox enforcement.
