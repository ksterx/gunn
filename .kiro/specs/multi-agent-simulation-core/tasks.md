# Implementation Plan

- [x] 0. Set up project foundation and tooling
  - Configure pyproject.toml with Python 3.13, pydantic v2, orjson, jsonpatch, structlog, prometheus-client, opentelemetry, fastapi, websockets, sqlite
  - Set up ruff/mypy/pytest-asyncio/pre-commit configuration for code quality
  - Create CI pipeline with linting, type checking, and testing
  - Add docs/errors.md as single source of truth for error codes and status mappings
  - Write basic project documentation and development setup guide
  - _Requirements: 13.1, 13.2, 13.3_

- [x] 1. Set up core data models and types
  - Create Pydantic models for WorldState, View, and EventLogEntry with proper defaults
  - Define TypedDict types for Intent, Effect, EffectDraft, and ObservationDelta with schema versioning
  - Implement CancelToken class with reason tracking and wait functionality
  - Write unit tests for all data model validation and serialization
  - _Requirements: 0.1, 0.2, 0.3, 0.4, 0.5_

- [x] 2. Implement TimedQueue for latency simulation
  - Create TimedQueue class with heap-based priority scheduling
  - Implement put_at() method for timed delivery with proper locking
  - Implement get() method with lock-free sleep to avoid blocking
  - Add comprehensive unit tests for concurrent operations and timing accuracy with ±5ms tolerance
  - _Requirements: 6.4, 6.5, 4.7_

- [x] 3. Create hash chain utilities for log integrity
  - Implement canonical_json() function using orjson with sorted keys
  - Create chain_checksum() function with SHA-256 hash chaining
  - Add integrity validation methods for detecting corruption
  - Write unit tests for hash consistency and corruption detection
  - _Requirements: 7.1, 7.5_

- [x] 4. Build EventLog with append-only storage
  - Implement EventLog class with thread-safe append operations
  - Add hash chain checksum calculation in append() method with str | None req_id
  - Create get_entries_since() method for replay and catch-up
  - Implement validate_integrity() method for full chain verification
  - Write unit tests for concurrent appends and integrity validation
  - _Requirements: 1.2, 7.1, 7.3, 7.5_

- [x] 4.1. Add basic observability infrastructure
  - Initialize structlog for structured logging with global_seq, view_seq, agent_id, req_id
  - Add PII redaction processor for email, phone, tokens in structured logs
  - Use monotonic clock (loop.time()) for internal timing, wall-clock only for log display
  - Add latency measurement for append(), submit_intent(), broadcast_event() operations
  - Create basic metrics collection for operation counts and timing
  - Write unit tests for logging accuracy, PII redaction, and performance impact
  - _Requirements: 14.1, 14.4, 12.3_

- [x] 4.2. Create replay CLI utility with determinism
  - Build simple CLI tool for replaying event logs from specified ranges
  - Add replay --from 0 --to latest functionality for debugging
  - Record world_seed in EventLog for deterministic replay with fixed random sequences
  - Ensure single random source (random/np.random) with seed fixation for tests
  - Implement log validation and integrity checking in CLI
  - Write integration tests for replay consistency and determinism validation
  - _Requirements: 7.3, 7.4, 9.1, 9.4_

- [x] 5. Implement ObservationPolicy for partial observation
  - Create ObservationPolicy interface with distance and relationship filtering
  - Implement filter_world_state() method for generating agent views
  - Add should_observe_event() method for event visibility determination
  - Create calculate_observation_delta() method using RFC6902 JSON Patch with stable paths
  - Add max_patch_ops threshold with fallback to full snapshot for large changes
  - Write unit tests for filtering accuracy, patch generation, path stability, and large patch fallback
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 5.1. Define API contracts early
  - Create openapi.yaml with observe/intent/emit endpoints using Pydantic schema generation
  - Define minimal protobuf schema for Unity adapter communication
  - Add golden schema files and contract tests to prevent breaking changes in CI
  - Add explicit "breaking change flag" requirement for major schema version changes
  - Reference docs/errors.md for unified error code mapping to HTTP/gRPC status codes
  - Write documentation for external integration patterns
  - _Requirements: 13.1, 13.2, 13.3_

- [ ] 6. Create EffectValidator for intent validation
  - Implement DefaultEffectValidator with basic validation logic
  - Add validate_intent() method checking world state constraints
  - Create validation for quota limits, cooldowns, and permissions
  - Write unit tests for various validation scenarios and edge cases
  - _Requirements: 1.5, 3.4, 10.3_

- [x] 7. Implement core Orchestrator functionality
  - Create Orchestrator class with world_id and dependency injection
  - Add register_agent() method with observation policy assignment
  - Implement deterministic ordering using (sim_time, priority, source_id, uuid) tuple
  - Create _next_seq() and _current_sim_time() helper methods with sim_time authority handling
  - Write unit tests for agent registration and basic orchestration
  - _Requirements: 1.1, 1.3, 1.4, 9.1, 9.3_

- [x] 8. Build AgentHandle for per-agent interface
  - Implement AgentHandle class with view_seq tracking as thin proxy to Orchestrator
  - Create next_observation() method using Orchestrator's TimedQueue
  - Add submit_intent() method with proper error handling
  - Implement cancel() method for intent cancellation
  - Write unit tests for agent isolation and non-blocking operations
  - _Requirements: 3.1, 3.2, 4.4, 6.1, 6.2_

- [ ] 9. Add two-phase commit for intent processing
  - Implement submit_intent() method with idempotency checking using tuple keys and SQLite persistence
  - Add dedup_ttl configuration with cleanup job for expired entries (N minutes or N thousand entries)
  - Add TTL warmup guard for relaxed deduplication immediately after restart
  - Add staleness detection using latest_view_seq > context_seq + threshold
  - Create intent validation pipeline: quota → backpressure → priority → fairness (Weighted Round Robin) → validator → commit
  - Implement effect creation from validated intents with UUID generation and priority completion
  - Write unit tests for two-phase commit integrity, conflict resolution, and TTL cleanup
  - _Requirements: 3.3, 4.2, 4.3, 10.1, 10.2_

- [ ] 9.1. Create dummy LLM for cancellation testing
  - Build mock LLM adapter that yields every 20-30ms for responsive cancellation
  - Add configurable generation time and token count for testing
  - Implement proper cancel token integration for 100ms SLO validation
  - Write unit tests for cancellation timing accuracy within ±5ms tolerance
  - _Requirements: 6.2, 6.3, 11.2_

- [ ] 10. Implement observation distribution system
  - Create broadcast_event() method converting EffectDraft to complete Effect with priority completion
  - Add config.default_priority completion for EffectDraft when priority is unspecified
  - Add observation delta generation for affected agents with max_patch_ops fallback
  - Implement timed delivery using per-agent TimedQueues with latency models
  - Create efficient filtering to only notify relevant agents
  - Store world_id in EventLogEntry.source_metadata for multi-tenant auditing
  - Write unit tests for observation consistency, delivery timing, and priority completion
  - _Requirements: 2.2, 2.5, 6.4, 6.5_

- [ ] 11. Add cancellation and staleness detection
  - Implement issue_cancel_token() method with tuple key tracking
  - Create cancel_if_stale() method with configurable staleness threshold
  - Add automatic cancellation when context becomes outdated
  - Implement debounce logic to prevent rapid successive interruptions
  - Write unit tests for cancellation timing and staleness accuracy
  - _Requirements: 4.1, 4.2, 4.3, 4.7_

- [ ] 12. Create error handling and recovery system
  - Implement structured error types: StaleContextError, IntentConflictError, QuotaExceededError
  - Add ErrorRecoveryPolicy with configurable recovery strategies
  - Create CircuitBreaker class for fault tolerance with failure thresholds
  - Implement backpressure policies: defer (default), shed oldest, drop newest
  - Add queue_depth_high_watermark metrics and logging for backpressure monitoring
  - Reference docs/errors.md for unified error code mapping implementation
  - Write unit tests for error scenarios, recovery behavior, and backpressure triggers
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 13. Build RL-style facade interface
  - Create RLFacade class wrapping Orchestrator functionality
  - Implement observe(agent_id) method returning current observations
  - Add step(agent_id, intent) method returning (Effect, ObservationDelta) tuple
  - Create proper error handling and timeout management
  - Write unit tests for RL-style interaction patterns
  - _Requirements: 5.1, 5.2, 5.5_

- [ ] 14. Build message-oriented facade interface
  - Create MessageFacade class for event-driven interactions
  - Implement emit() method for broadcasting events through observation policies
  - Add message subscription and filtering based on agent policies
  - Create async message delivery with proper error handling
  - Write unit tests for message routing and delivery
  - _Requirements: 5.3, 5.4, 5.5_

- [ ] 15. Add performance monitoring and metrics
  - Implement structured logging with global_seq, view_seq, agent_id, req_id, latency_ms
  - Create Prometheus metrics for queue depths, cancel rates, conflicts, throughput
  - Add OpenTelemetry tracing support across components
  - Implement memory usage tracking and reporting
  - Add bandwidth/CPU measurement for max_patch_ops fallback scenarios
  - Write unit tests for metrics accuracy and performance impact
  - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 16. Create memory management and compaction
  - Implement MemoryManager class with configurable limits
  - Add WorldState snapshot creation every N events for faster replay
  - Create log compaction while preserving replay capability
  - Implement view cache eviction with LRU policy
  - Write unit tests for memory limits and compaction correctness
  - _Requirements: 7.3, 11.4_

- [ ] 17. Build Web adapter for external integration
  - Create FastAPI-based Web adapter with REST endpoints for intent submission
  - Add WebSocket support for real-time observation streaming
  - Implement authentication and authorization per agent_id/world_id
  - Create proper error handling and rate limiting
  - Write integration tests for Web API functionality
  - _Requirements: 8.3, 12.1, 12.4_

- [ ] 18. Implement LLM adapter with streaming support
  - Create LLM adapter with streaming token generation
  - Add cancel token integration for 100ms cancellation SLO
  - Implement token yield every 20-30ms for responsive interruption
  - Create proper error handling for generation failures
  - Write integration tests for streaming and cancellation behavior
  - _Requirements: 6.1, 6.2, 6.3, 11.2_

- [ ] 19. Create Unity adapter for game engine integration
  - Build Unity adapter using WebSocket communication
  - Implement TimeTick event conversion to Effects
  - Add Move intent conversion to Unity game commands
  - Create physics collision event handling
  - Write integration tests with Unity simulation
  - _Requirements: 8.1, 8.4, 8.5_

- [ ] 20. Add comprehensive integration testing
  - Create multi-agent conversation test with interruption scenarios
  - Implement performance tests validating SLOs: 20ms delivery, 100ms cancellation, 100 intents/sec
  - Add replay consistency tests ensuring identical behavior from same event log
  - Create determinism golden trace tests: same log → identical results in CI
  - Create fault tolerance tests with network partitions and adapter failures
  - Add CI performance job for SLO regression detection (measurement-only)
  - Write end-to-end tests covering complete user workflows
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 7.4, 9.4_

- [ ] 21. Implement security and multitenancy features
  - Add tenant isolation with proper access controls
  - Implement PII redaction in logs for sensitive fields
  - Create rate limiting per tenant and agent
  - Add security event logging and alerting
  - Implement agent quarantine functionality for security violations (configurable flag)
  - Write security tests for isolation, access control, and quarantine behavior
  - _Requirements: 12.1, 12.2, 12.3, 12.5_

- [ ] 22. Create configuration and deployment utilities
  - Implement configuration validation and defaults
  - Add environment-specific configuration loading
  - Add GUNN_FEATURES environment variable for feature flags (latency, backpressure, etc.)
  - Create deployment scripts and Docker containers
  - Add health checks and readiness probes
  - Export feature flag status in metrics for operational visibility
  - Write deployment tests and documentation
  - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [ ] 23. Build example applications and demos
  - Create A/B/C conversation demo with visible interruption and regeneration
  - Implement 2D spatial simulation with movement and distance-based observation
  - Add Unity integration demo showing real-time agent interaction
  - Create performance benchmark scenarios for load testing
  - Write comprehensive documentation and tutorials
  - _Requirements: All requirements demonstrated in working examples_
