# Requirements Document

## Introduction

This feature implements a multi-agent simulation core system that enables multiple AI agents to interact in a shared environment with partial observation, concurrent execution, and intelligent interruption/regeneration capabilities. The system uses an event-driven architecture with a unified core that supports both RL-style and message-oriented interaction patterns.

This system provides a deterministic, event-driven core with per-agent views (partial observability), a two-phase intent→effect protocol, and interruption-aware scheduling. It supports both RL-style and message-oriented facades over the same log-structured state, enabling replayable, auditable simulations and real-time integrations with game engines and web APIs.

## Requirements

### Requirement 0: Glossary & Identity

**User Story:** As a developer, I want clear definitions of core types and identifiers, so that I can implement and test the system consistently.

#### Acceptance Criteria

1. WHEN the system initializes THEN it SHALL define canonical types: WorldState, View, ObservationDelta, Intent, Effect, Event, global_seq, view_seq, req_id, agent_id
2. WHEN events are processed THEN global_seq SHALL be a strictly monotonically increasing integer assigned by the core
3. WHEN agents receive observations THEN each agent SHALL maintain view_seq (last applied global sequence visible to that agent)
4. WHEN intents are submitted THEN req_id SHALL be unique per intent and traceable across logs and telemetry
5. WHEN ObservationDelta is generated THEN patch format SHALL use RFC6902 JSON Patch specification

### Requirement 1: Event-Driven Core Architecture

**User Story:** As a developer, I want a unified event-driven core system, so that I can build multi-agent simulations with consistent state management and observation handling.

#### Acceptance Criteria

1. WHEN the system initializes THEN it SHALL create a WorldState as the single source of truth
2. WHEN an event occurs THEN the system SHALL record it in a global sequential log with a global_seq identifier
3. WHEN an agent requests observation THEN the system SHALL generate a View based on the agent's observation policy
4. WHEN multiple events arrive within the same scheduling quantum THEN the system SHALL process them using deterministic ordering rule: (sim_time, priority, source, uuid) → global_seq
5. IF an agent submits an Intent THEN the system SHALL validate it through EffectValidator before creating an Effect

### Requirement 2: Partial Observation System

**User Story:** As a simulation designer, I want agents to have limited visibility of the world state, so that I can create realistic scenarios where agents don't have perfect information.

#### Acceptance Criteria

1. WHEN an agent requests observation THEN the system SHALL apply the agent's ObservationPolicy to filter WorldState
2. WHEN the world state changes THEN the system SHALL generate ObservationDelta patches using RFC6902 JSON Patch format only for affected agents
3. WHEN an ObservationPolicy includes distance constraints THEN agents SHALL only observe entities within their specified range
4. WHEN an ObservationPolicy includes relationship constraints THEN agents SHALL only observe entities they have relationships with
5. IF an agent's view_seq is outdated THEN the system SHALL provide incremental updates via ObservationDelta patches

### Requirement 3: Concurrent Agent Execution

**User Story:** As a simulation runner, I want multiple agents to act simultaneously, so that the simulation feels natural and responsive.

#### Acceptance Criteria

1. WHEN multiple agents generate intents concurrently THEN the system SHALL process them without blocking each other
2. WHEN agents submit intents THEN each SHALL receive a unique req_id for tracking
3. WHEN the system processes intents THEN it SHALL use a two-phase commit (Intent → Effect) pattern with validation for staleness, validator rules, and quota
4. WHEN intent conflicts occur THEN the system SHALL resolve them according to priority and fairness policies
5. IF an agent's generation takes too long THEN the system SHALL enforce deadline_ms and token_budget limits

### Requirement 4: Intelligent Interruption and Regeneration

**User Story:** As a simulation designer, I want agents to intelligently interrupt and regenerate responses when new relevant information arrives, so that conversations feel natural and responsive.

#### Acceptance Criteria

1. WHEN an agent begins generating a response THEN the system SHALL issue a cancel_token with the current context_digest
2. WHEN new events occur during generation THEN the system SHALL evaluate staleness using: latest_view_seq > context_seq + staleness_threshold (default: 0, configurable per agent policy)
3. WHEN context becomes stale THEN the system SHALL cancel the current generation and trigger regeneration
4. WHEN cancellation occurs THEN the system SHALL provide the reason (stale_due_to) to the agent
5. IF interrupt_on_new_info policy is "always" THEN any new information SHALL trigger interruption
6. IF interrupt_on_new_info policy is "only_conflict" THEN only conflicting information SHALL trigger interruption
7. WHEN debounce_ms is configured THEN the system SHALL suppress rapid successive interruptions within the time window

### Requirement 5: Dual API Facades

**User Story:** As a developer, I want to interact with the system using either RL-style or message-oriented patterns, so that I can choose the most appropriate interface for my use case.

#### Acceptance Criteria

1. WHEN using RL facade THEN developers SHALL call env.observe(agent_id) to get observations
2. WHEN using RL facade THEN developers SHALL call env.step(agent_id, intent) and receive (Effect, ObservationDelta) tuple
3. WHEN using message facade THEN developers SHALL call env.emit() to broadcast events through ObservationPolicy filtering
4. WHEN using message facade THEN agents SHALL receive messages according to their observation policies
5. IF either facade is used THEN both SHALL operate on the same underlying event system

### Requirement 6: Streaming and Real-time Processing

**User Story:** As a simulation runner, I want agents to process information in real-time with streaming capabilities, so that the simulation responds immediately to changes.

#### Acceptance Criteria

1. WHEN LLM generates responses THEN it SHALL stream tokens incrementally
2. WHEN streaming generation is active THEN the system SHALL monitor for cancellation signals at token boundaries
3. WHEN cancel_token is triggered THEN the system SHALL immediately halt token generation within 100ms
4. WHEN ObservationDelta is generated THEN core SHALL add no intentional batching with per-message scheduling quantum ≤ 20ms
5. IF network latency simulation is enabled THEN ObservationDelta SHALL be delayed according to the latency model

### Requirement 7: State Consistency and Audit Trail

**User Story:** As a simulation analyst, I want complete audit trails and state consistency, so that I can replay, debug, and analyze simulation runs.

#### Acceptance Criteria

1. WHEN events occur THEN the system SHALL maintain a complete sequential log with global_seq numbers
2. WHEN agents receive observations THEN their view_seq SHALL be updated to reflect the latest synchronized state
3. WHEN simulation runs complete THEN the system SHALL provide replay capabilities from the event log
4. WHEN state queries are made THEN the system SHALL provide consistent snapshots based on global_seq
5. IF corruption is detected THEN the system SHALL validate state consistency using log hash/CRC and sequence gap detection

### Requirement 8: External System Integration

**User Story:** As a game developer, I want to integrate the simulation core with Unity, Unreal, or web APIs, so that I can create rich interactive experiences.

#### Acceptance Criteria

1. WHEN Unity adapter is used THEN it SHALL convert game events to Effects and Intents to game commands
2. WHEN Unreal adapter is used THEN it SHALL handle gRPC/WebSocket communication for event synchronization
3. WHEN web API adapter is used THEN it SHALL expose REST/WebSocket endpoints for external system integration
4. WHEN game engine provides TimeTick THEN it SHALL be converted to Effect events for time synchronization
5. IF physics collisions occur in the game engine THEN they SHALL be reflected as Effects in the simulation core

### Requirement 9: Determinism & Timebase

**User Story:** As a simulation developer, I want deterministic and reproducible simulations, so that I can debug issues and ensure consistent behavior.

#### Acceptance Criteria

1. WHEN events are ordered THEN the core SHALL provide deterministic ordering rule: (sim_time, priority, source, uuid) → global_seq
2. WHEN timebase is configured THEN the system SHALL support sim_time (engine ticks) and wall_time (real clock), using sim_time primarily if available
3. WHEN ordering ties occur THEN they SHALL be resolved by fixed tie-breaker: priority > source > UUID lexicographic
4. WHEN replays are performed THEN using the same event log SHALL reproduce identical global_seq and effects
5. IF multiple scheduling quanta overlap THEN the system SHALL maintain strict ordering within each quantum

### Requirement 10: Error Handling, Idempotency, Backpressure

**User Story:** As a system operator, I want robust error handling and backpressure management, so that the system remains stable under load.

#### Acceptance Criteria

1. WHEN retries occur THEN the system SHALL define idempotent ingestion where duplicate req_id MUST NOT create duplicate effects
2. WHEN queues exceed thresholds THEN the system SHALL apply backpressure policies configurable per agent class: defer (default), shed (drop oldest), drop (drop newest)
3. WHEN errors occur THEN the system SHALL surface structured error codes: STALE_CONTEXT, INTENT_CONFLICT, TIMEOUT, QUOTA_EXCEEDED
4. WHEN partial failures happen THEN the system SHALL remain consistent with the event log and report recoverable actions
5. IF cascading failures are detected THEN the system SHALL implement circuit breaker patterns per agent or adapter

### Requirement 11: Performance & SLOs

**User Story:** As a performance engineer, I want defined performance targets, so that I can validate system performance and identify regressions.

#### Acceptance Criteria

1. WHEN processing observations THEN median end-to-end ObservationDelta delivery latency (core in-proc) SHALL be ≤ 20ms under nominal load (≤10 concurrent agents, ≤50 events/sec)
2. WHEN cancelling generation THEN cancel-to-halt latency for LLM streaming SHALL be ≤ 100ms at token boundaries
3. WHEN processing intents THEN the system SHALL process ≥ 100 intents/sec per agent under benchmark scenario
4. WHEN handling multiple agents THEN all operations SHALL be non-blocking per agent (no head-of-line blocking across agents)
5. IF load exceeds capacity THEN the system SHALL gracefully degrade with defined backpressure policies

### Requirement 12: Security & Multitenancy

**User Story:** As a security engineer, I want secure multi-tenant operation, so that different simulations and users are properly isolated.

#### Acceptance Criteria

1. WHEN external adapters connect THEN they SHALL authenticate using mTLS or token-based auth and authorize per agent_id/world_id
2. WHEN multiple tenants use the system THEN it SHALL support tenancy isolation with no cross-tenant observation unless explicitly allowed by policy
3. WHEN logging occurs THEN sensitive payload fields SHALL be redacted by default (email, phone, address, auth tokens, user_data fields containing PII)
4. WHEN API access is requested THEN the system SHALL enforce rate limiting per tenant and agent
5. IF security violations are detected THEN the system SHALL log security events and optionally quarantine violating agents

### Requirement 13: Schema & Versioning

**User Story:** As an API maintainer, I want schema evolution support, so that I can update the system without breaking existing integrations.

#### Acceptance Criteria

1. WHEN API messages are sent THEN Intent/Effect/ObservationDelta SHALL include a schema_version field using semantic versioning (e.g., "1.2.0")
2. WHEN schema changes occur THEN minor changes SHALL be backward compatible within the same major version
3. WHEN integration is needed THEN the system SHALL expose JSON Schema for messages and generate OpenAPI for Web adapter
4. WHEN version mismatches occur THEN the system SHALL provide clear error messages and migration guidance
5. IF deprecated features are used THEN the system SHALL log deprecation warnings with migration timelines

### Requirement 14: Observability

**User Story:** As a system operator, I want comprehensive observability, so that I can monitor, debug, and optimize the system.

#### Acceptance Criteria

1. WHEN events are processed THEN the system SHALL emit structured logs with global_seq, view_seq, agent_id, req_id, latency_ms
2. WHEN metrics are collected THEN the system SHALL expose Prometheus metrics for: queue depths, cancel rate, stale-context rate, conflicts, per-agent throughput
3. WHEN distributed operations occur THEN the system SHALL support OpenTelemetry tracing across adapters and core
4. WHEN performance issues arise THEN the system SHALL provide detailed timing breakdowns for intent processing pipeline
5. IF anomalies are detected THEN the system SHALL generate alerts for SLO violations and error rate spikes