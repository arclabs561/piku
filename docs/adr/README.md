# Architecture Decision Records

| ADR | Status | Decision |
| --- | --- | --- |
| [0001](0001-runtime-owned-provider-resolution.md) | Accepted | Provider resolution belongs in `piku-runtime`; protocol clients stay in `piku-api`. |
| [0002](0002-explicit-embed-backend-config.md) | Accepted | Embedding backend selection is explicit; Ollama defaults stay in `piku-api`. |
| [0003](0003-split-embed-memory-modules.md) | Accepted | Split embedding memory internals behind the existing public facade. |
| [0004](0004-deterministic-agent-loop-coverage.md) | Accepted | PR-blocking agent-loop coverage uses scripted providers; live suites stay opt-in. |
| [0005](0005-live-llm-matrix.md) | Accepted | Live LLM coverage runs outside PR CI across a small model matrix. |
| [0006](0006-live-dogfood-ledger.md) | Accepted | Live dogfood writes comparable JSONL run records when requested. |
