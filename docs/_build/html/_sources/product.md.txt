# Product Overview

**gunn** (群) is a multi-agent simulation core that provides a controlled interface for agent–environment interaction, supporting both single and multi-agent settings.

## Core Purpose

The system enables multiple AI agents to interact in a shared environment with:
- Partial observation driven by configurable policies
- Concurrent execution managed by a deterministic orchestrator
- Interruption-aware scheduling with cancel tokens and staleness checks
- An event-driven architecture with a replayable log for auditability

## Current Capabilities (implementation status)

- Deterministic event log with hash-chain integrity and a replay CLI for debugging/replay workflows.
- Async orchestrator that handles agent registration, intent submission, deduplication, quota/backpressure enforcement, and weighted round-robin scheduling before emitting effects.
- Observation pipeline that generates RFC6902 JSON Patch deltas per agent, supports latency models, and tracks per-agent view sequences.
- Telemetry utilities providing structured logging with PII redaction plus Prometheus-ready metrics/timers.
- Storage layer with SQLite-backed or in-memory deduplication to guarantee idempotent intent processing.

## Roadmap Highlights (see `tasks.md`)

- Implement richer `EffectValidator` logic (Task 6) covering permissions, quotas, and conflict resolution.
- Ship RL-style and message-oriented facades (Tasks 13–14) to expose the core through ergonomic APIs.
- Build production adapters (Tasks 17–19) for web, LLM streaming, and Unity integrations.
- Extend observability, performance monitoring, and memory management (Tasks 15–16).
- Harden security, multitenancy, and deployment tooling (Tasks 21–22) ahead of public release.

## Target Use Cases

Current builds are best suited for prototyping deterministic multi-agent worlds, experimenting with observation policies, and validating orchestration workflows. Full external adapter support, streaming UX, and end-to-end demos are in progress and tracked via the roadmap above.
