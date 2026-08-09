# Design: spatial workspace canvas

Status: implemented checkpoint

## Problem

The prototype conflates the user-owned workspace with a model-generated HTML
page. The result is a nested “workspace” inside the actual object board, unclear
targeting, and two incompatible meanings for canvas changes.

## Chosen approach

The workspace and canvas are one host-owned spatial board. Chat, terminal,
file, note, task, and page-preview objects live on it. Workspace changes are
typed host operations over those objects. Page-source changes target one
selected sandboxed page-preview object and cannot alter host chrome, authority
labels, or other workspace objects. Existing saved HTML migrates into an
optional page-preview object rather than occupying the canvas itself.

Object identity and geometry persist on the server per surface; browser storage
is not an authority. Human dragging and resizing update the same object record
that model-proposed workspace operations use. Creating a terminal through
a workspace operation creates an inert terminal object; only a human click may
start its PTY.

The current PTY is deliberately labeled as an unrestricted host shell, not a
sandbox. Every terminal object starts inert and presents that warning before a
human starts it. A future sandbox must remove at least one part of the dangerous
combination of untrusted repository content, private host access, and general
network access; changing the current directory alone is not confinement.

Generated pages run in an opaque-origin iframe with scripts but without forms,
popups, downloads, parent access, or network-capable CSP directives. The host
injects `default-src 'none'`, `connect-src 'none'`, and `form-action 'none'`.

The visual system is modern restrained brutalism using GitHub dark neutrals and
one blue signal color. Trusted chrome uses only three type sizes: label, body,
and title. Borders communicate ownership and focus; color communicates state.

### Visual-system choice

GitHub's brand palette is neutral-first with one hero color, but Piku is a
product surface rather than a marketing surface. It therefore follows Primer's
semantic product roles (`canvas`, `foreground`, `border`, `accent`, `success`,
and `danger`) instead of copying the GitHub brand green everywhere. The dark
base is `#0d1117`; blue marks focus and selected authority. Green and red are
reserved for outcomes.

Google Design is a useful reference for accessibility, onboarding, motion, and
explaining evolving AI capabilities. Its more expressive typography, shape,
and motion language is not the default here: an agent harness benefits more
from GitHub's compact technical density and quieter state hierarchy. Motion may
explain a transition, but never substitute for durable progress or provenance.

References:

- https://brand.github.com/foundations/color
- https://primer.style/product/primitives/color/
- https://design.google/

## Non-goals

- Generated HTML does not rewrite trusted host UI or security controls.
- Workspace operations do not execute commands or type into terminals.
- This checkpoint does not build a general reactive graph or arbitrary plugin
  schema.
- Binary file rendering remains viewer-specific rather than dumping arbitrary
  bytes into a text object.

## Decision gates

- Reject the design if a generated page can create, start, observe, or control a
  terminal.
- Reject a claim that the PTY is sandboxed until filesystem, environment,
  process, credential, and network limits are enforced and tested.
- Reject an operation if its target ID is absent, its object kind is unknown, or
  its geometry/content exceeds host bounds.
- Revisit the schema when a second artifact type cannot be represented without
  page-specific fields on every object.

## Why not one generated webpage?

It makes terminals, provenance, evidence, and approvals awkward overlays and
gives presentation output conceptual control over the workspace.

## Why not the current hybrid?

It creates nested canvases and makes ordinary language such as “make it red”
ambiguous about whether it targets the board or a child page.

---
Decided: 2026-08-08 | Session: 019fe1d4
