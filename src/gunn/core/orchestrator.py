"""Core orchestrator for multi-agent simulation.

This module provides the central Orchestrator class that coordinates all system
operations including event ingestion, view generation, intent validation,
and observation distribution with deterministic ordering.
"""

import asyncio
import uuid
from typing import Any, Literal, Protocol

from gunn.core.event_log import EventLog
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.messages import WorldState
from gunn.schemas.types import CancelToken, Effect, EffectDraft, Intent
from gunn.storage.dedup_store import DedupStore
from gunn.utils.errors import StaleContextError
from gunn.utils.telemetry import MonotonicClock, get_logger
from gunn.utils.timing import TimedQueue


class EffectValidator(Protocol):
    """Protocol for validating intents before creating effects."""

    def validate_intent(self, intent: Intent, world_state: WorldState) -> bool:
        """Validate if intent can be executed.

        Args:
            intent: Intent to validate
            world_state: Current world state

        Returns:
            True if intent is valid and can be executed
        """
        ...


class DefaultEffectValidator:
    """Default implementation of EffectValidator with basic validation."""

    def validate_intent(self, intent: Intent, world_state: WorldState) -> bool:
        """Basic validation - always returns True for now.

        Args:
            intent: Intent to validate
            world_state: Current world state

        Returns:
            True (basic implementation allows all intents)
        """
        # TODO: Implement real validation logic in task 6
        return True


class AgentHandle:
    """Per-agent interface for observation and intent submission.

    Provides isolated interface for each agent with view sequence tracking
    and non-blocking operations.
    """

    def __init__(self, agent_id: str, orchestrator: "Orchestrator"):
        """Initialize agent handle.

        Args:
            agent_id: Unique identifier for this agent
            orchestrator: Reference to the orchestrator
        """
        self.agent_id = agent_id
        self.orchestrator = orchestrator
        self.view_seq: int = 0

    async def next_observation(self) -> Any:
        """Get next observation delta from orchestrator's timed queue.

        Returns:
            Next observation delta when available

        Raises:
            RuntimeError: If agent is not registered or queue is closed
        """
        if self.agent_id not in self.orchestrator._per_agent_queues:
            raise RuntimeError(f"Agent {self.agent_id} is not registered")

        timed_queue = self.orchestrator._per_agent_queues[self.agent_id]
        delta = await timed_queue.get()

        # Update view sequence from delta
        if hasattr(delta, "view_seq"):
            self.view_seq = delta.view_seq
        elif isinstance(delta, dict) and "view_seq" in delta:
            self.view_seq = delta["view_seq"]

        return delta

    async def submit_intent(self, intent: Intent) -> str:
        """Submit intent for validation and processing.

        Args:
            intent: Intent to submit

        Returns:
            Request ID for tracking the intent
        """
        return await self.orchestrator.submit_intent(intent)

    async def cancel(self, req_id: str) -> None:
        """Cancel pending intent by req_id.

        Args:
            req_id: Request ID to cancel
        """
        key = (self.orchestrator.world_id, self.agent_id, req_id)
        if key in self.orchestrator._cancel_tokens:
            self.orchestrator._cancel_tokens[key].cancel("user_requested")

    def get_view_seq(self) -> int:
        """Get current view sequence number.

        Returns:
            Current view sequence number for this agent
        """
        return self.view_seq


class OrchestratorConfig:
    """Configuration for the Orchestrator."""

    def __init__(
        self,
        max_agents: int = 100,
        staleness_threshold: int = 0,
        debounce_ms: float = 100.0,
        deadline_ms: float = 5000.0,
        token_budget: int = 1000,
        backpressure_policy: str = "defer",
        default_priority: int = 0,
        dedup_ttl_minutes: int = 60,
        max_dedup_entries: int = 10000,
        cleanup_interval_minutes: int = 10,
        warmup_ttl_minutes: int = 5,
        queue_depth_high_watermark: int = 1000,
    ):
        """Initialize orchestrator configuration.

        Args:
            max_agents: Maximum number of agents
            staleness_threshold: Threshold for staleness detection
            debounce_ms: Debounce time for interruptions
            deadline_ms: Deadline for intent processing
            token_budget: Token budget for generation
            backpressure_policy: Policy for handling backpressure
            default_priority: Default priority when not specified
            dedup_ttl_minutes: TTL for deduplication entries
            max_dedup_entries: Maximum deduplication entries
            cleanup_interval_minutes: Cleanup interval for dedup store
            warmup_ttl_minutes: Warmup TTL for relaxed deduplication
            queue_depth_high_watermark: High watermark for queue depth monitoring
        """
        self.max_agents = max_agents
        self.staleness_threshold = staleness_threshold
        self.debounce_ms = debounce_ms
        self.deadline_ms = deadline_ms
        self.token_budget = token_budget
        self.backpressure_policy = backpressure_policy
        self.default_priority = default_priority
        self.dedup_ttl_minutes = dedup_ttl_minutes
        self.max_dedup_entries = max_dedup_entries
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.warmup_ttl_minutes = warmup_ttl_minutes
        self.queue_depth_high_watermark = queue_depth_high_watermark


class Orchestrator:
    """Central coordinator for multi-agent simulation.

    Coordinates all system operations including event ingestion and ordering,
    view generation and delta distribution, intent validation and effect creation,
    and cancellation token management.

    Requirements addressed:
    - 1.1: Create WorldState as single source of truth
    - 1.3: Generate View based on agent's observation policy
    - 1.4: Process events using deterministic ordering
    - 9.1: Deterministic ordering using (sim_time, priority, source_id, uuid) tuple
    - 9.3: Fixed tie-breaker: priority > source > UUID lexicographic
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        world_id: str = "default",
        effect_validator: EffectValidator | None = None,
        dedup_store: DedupStore | None = None,
    ):
        """Initialize orchestrator with configuration and dependencies.

        Args:
            config: Orchestrator configuration
            world_id: Identifier for this world instance
            effect_validator: Optional custom effect validator
            dedup_store: Optional custom deduplication store
        """
        self.world_id = world_id
        self.config = config
        self.event_log: EventLog = EventLog(world_id)
        self.world_state: WorldState = WorldState()
        self.observation_policies: dict[str, ObservationPolicy] = {}
        self.effect_validator: EffectValidator = (
            effect_validator or DefaultEffectValidator()
        )
        self.agent_handles: dict[str, AgentHandle] = {}

        # Deduplication store
        self.dedup_store: DedupStore = dedup_store or DedupStore(
            dedup_ttl_minutes=config.dedup_ttl_minutes,
            max_entries=config.max_dedup_entries,
            cleanup_interval_minutes=config.cleanup_interval_minutes,
            warmup_ttl_minutes=config.warmup_ttl_minutes,
        )

        # Internal state
        self._global_seq: int = 0
        self._cancel_tokens: dict[
            tuple[str, str, str], CancelToken
        ] = {}  # (world_id, agent_id, req_id)
        self._per_agent_queues: dict[str, TimedQueue] = {}  # Timed delivery queues
        self._sim_time_authority: str = "none"  # Which adapter controls sim_time
        self._initialized: bool = False

        # Processing pipeline locks for fairness
        self._processing_lock = asyncio.Lock()
        self._agent_processing_order: list[str] = []  # Round-robin order

        # Logging
        self._logger = get_logger("gunn.orchestrator", world_id=world_id)

        self._logger.info(
            "Orchestrator initialized",
            world_id=world_id,
            max_agents=config.max_agents,
            sim_time_authority=self._sim_time_authority,
            dedup_ttl_minutes=config.dedup_ttl_minutes,
            staleness_threshold=config.staleness_threshold,
        )

    async def initialize(self) -> None:
        """Initialize orchestrator and dependencies."""
        if self._initialized:
            return

        await self.dedup_store.initialize()
        self._initialized = True

        self._logger.info("Orchestrator initialization complete")

    def set_sim_time_authority(
        self, authority: Literal["unity", "unreal", "none"]
    ) -> None:
        """Set which adapter controls sim_time.

        Args:
            authority: Authority for sim_time ("unity", "unreal", or "none")
        """
        self._sim_time_authority = authority
        self._logger.info("Sim time authority changed", authority=authority)

    async def register_agent(
        self, agent_id: str, policy: ObservationPolicy
    ) -> AgentHandle:
        """Register a new agent with observation policy.

        Args:
            agent_id: Unique identifier for the agent
            policy: Observation policy for this agent

        Returns:
            AgentHandle for the registered agent

        Raises:
            ValueError: If agent_id is empty or already registered
            RuntimeError: If maximum agents exceeded
        """
        if not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")

        if agent_id in self.agent_handles:
            raise ValueError(f"Agent {agent_id} is already registered")

        if len(self.agent_handles) >= self.config.max_agents:
            raise RuntimeError(f"Maximum agents ({self.config.max_agents}) exceeded")

        # Create agent handle
        handle = AgentHandle(agent_id, self)

        # Store policy and handle
        self.observation_policies[agent_id] = policy
        self.agent_handles[agent_id] = handle

        # Create timed queue for this agent
        self._per_agent_queues[agent_id] = TimedQueue()

        self._logger.info(
            "Agent registered",
            agent_id=agent_id,
            total_agents=len(self.agent_handles),
            policy_type=type(policy).__name__,
        )

        return handle

    async def broadcast_event(self, draft: EffectDraft) -> None:
        """Create complete effect from draft and distribute observations.

        Args:
            draft: Effect draft to process and broadcast

        Raises:
            ValueError: If draft is missing required fields
        """
        # Validate draft
        if not draft.get("kind"):
            raise ValueError("EffectDraft must have 'kind' field")
        if not draft.get("source_id"):
            raise ValueError("EffectDraft must have 'source_id' field")
        if not draft.get("schema_version"):
            raise ValueError("EffectDraft must have 'schema_version' field")

        # Create complete effect with orchestrator-managed fields
        effect: Effect = {
            "uuid": uuid.uuid4().hex,
            "kind": draft["kind"],
            "payload": draft["payload"],
            "source_id": draft["source_id"],
            "schema_version": draft["schema_version"],
            "sim_time": self._current_sim_time(),
            "global_seq": self._next_seq(),
        }

        # Append to event log
        await self.event_log.append(effect)

        # TODO: Update world state and distribute observations
        # This will be implemented in later tasks

        self._logger.info(
            "Event broadcast",
            effect_kind=effect["kind"],
            effect_uuid=effect["uuid"],
            global_seq=effect["global_seq"],
            sim_time=effect["sim_time"],
            source_id=effect["source_id"],
        )

    async def submit_intent(self, intent: Intent) -> str:
        """Two-phase commit: idempotency → staleness → quota → backpressure → priority → fairness → validator → commit.

        Args:
            intent: Intent to process

        Returns:
            Request ID for tracking

        Raises:
            ValueError: If intent is invalid
            StaleContextError: If intent context is stale
            QuotaExceededError: If quota limits exceeded
            ValidationError: If intent validation fails
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")

        # Validate intent structure
        if not intent.get("req_id"):
            raise ValueError("Intent must have 'req_id' field")
        if not intent.get("agent_id"):
            raise ValueError("Intent must have 'agent_id' field")
        if not intent.get("kind"):
            raise ValueError("Intent must have 'kind' field")

        req_id = intent["req_id"]
        agent_id = intent["agent_id"]
        context_seq = intent.get("context_seq", 0)

        self._logger.debug(
            "Processing intent",
            req_id=req_id,
            agent_id=agent_id,
            intent_kind=intent["kind"],
            context_seq=context_seq,
        )

        # Phase 1: Idempotency check using persistent store
        existing_seq = await self.dedup_store.check_and_record(
            self.world_id, agent_id, req_id, self._global_seq + 1
        )

        if existing_seq is not None:
            self._logger.info(
                "Intent already processed (idempotent)",
                req_id=req_id,
                agent_id=agent_id,
                existing_seq=existing_seq,
            )
            return req_id

        # Phase 2: Staleness detection
        await self._check_staleness(agent_id, req_id, context_seq)

        # Phase 3: Processing pipeline with fairness
        async with self._processing_lock:
            # Quota check
            await self._check_quota(agent_id, intent)

            # Backpressure check
            await self._check_backpressure(agent_id)

            # Priority and fairness handling
            await self._handle_priority_and_fairness(agent_id, intent)

            # Validation
            if not self.effect_validator.validate_intent(intent, self.world_state):
                from gunn.utils.errors import ValidationError

                raise ValidationError(intent, ["Effect validator rejected intent"])

            # Phase 4: Commit - create and broadcast effect
            effect_draft: EffectDraft = {
                "kind": intent["kind"],
                "payload": intent["payload"],
                "source_id": agent_id,
                "schema_version": intent.get("schema_version", "1.0.0"),
            }

            # Apply priority completion if not specified
            if "priority" not in effect_draft["payload"]:
                effect_draft["payload"]["priority"] = intent.get(
                    "priority", self.config.default_priority
                )

            # Broadcast the effect (this will assign the global_seq)
            await self.broadcast_event(effect_draft)

            self._logger.info(
                "Intent processed successfully",
                req_id=req_id,
                agent_id=agent_id,
                intent_kind=intent["kind"],
                global_seq=self._global_seq,
                context_seq=context_seq,
            )

            return req_id

    async def _check_staleness(
        self, agent_id: str, req_id: str, context_seq: int
    ) -> None:
        """Check if intent context is stale.

        Args:
            agent_id: Agent identifier
            req_id: Request identifier
            context_seq: Context sequence from intent

        Raises:
            StaleContextError: If context is stale
        """
        agent_handle = self.agent_handles.get(agent_id)
        if not agent_handle:
            # Agent not registered, can't check staleness
            return

        latest_seq = self._global_seq
        staleness = latest_seq - context_seq

        if staleness > self.config.staleness_threshold:
            self._logger.info(
                "Intent rejected due to staleness",
                agent_id=agent_id,
                req_id=req_id,
                context_seq=context_seq,
                latest_seq=latest_seq,
                staleness=staleness,
                threshold=self.config.staleness_threshold,
            )
            raise StaleContextError(req_id, context_seq, latest_seq)

    async def _check_quota(self, agent_id: str, intent: Intent) -> None:
        """Check quota limits for agent.

        Args:
            agent_id: Agent identifier
            intent: Intent to check

        Raises:
            QuotaExceededError: If quota limits exceeded
        """
        # TODO: Implement actual quota checking
        # For now, this is a placeholder that always passes
        pass

    async def _check_backpressure(self, agent_id: str) -> None:
        """Check backpressure conditions.

        Args:
            agent_id: Agent identifier

        Raises:
            BackpressureError: If backpressure limits exceeded
        """
        # Check queue depth for this agent
        if agent_id in self._per_agent_queues:
            queue = self._per_agent_queues[agent_id]
            queue_depth = queue.qsize() if hasattr(queue, "qsize") else 0

            if queue_depth > self.config.queue_depth_high_watermark:
                from gunn.utils.errors import BackpressureError

                raise BackpressureError(
                    agent_id=agent_id,
                    queue_type="observation",
                    current_depth=queue_depth,
                    threshold=self.config.queue_depth_high_watermark,
                    policy=self.config.backpressure_policy,
                )

    async def _handle_priority_and_fairness(
        self, agent_id: str, intent: Intent
    ) -> None:
        """Handle priority and fairness using Weighted Round Robin.

        Args:
            agent_id: Agent identifier
            intent: Intent being processed
        """
        # Update round-robin order
        if agent_id not in self._agent_processing_order:
            self._agent_processing_order.append(agent_id)
        else:
            # Move to end for round-robin fairness
            self._agent_processing_order.remove(agent_id)
            self._agent_processing_order.append(agent_id)

        # TODO: Implement actual priority weighting and fairness delays
        # For now, this ensures round-robin ordering
        pass

    def issue_cancel_token(self, agent_id: str, req_id: str) -> CancelToken:
        """Issue cancellation token for generation tracking.

        Args:
            agent_id: Agent identifier
            req_id: Request identifier

        Returns:
            CancelToken for tracking cancellation
        """
        token = CancelToken(req_id, agent_id)
        key = (self.world_id, agent_id, req_id)
        self._cancel_tokens[key] = token

        self._logger.debug(
            "Cancel token issued",
            agent_id=agent_id,
            req_id=req_id,
        )

        return token

    async def cancel_if_stale(
        self, agent_id: str, req_id: str, new_view_seq: int
    ) -> bool:
        """Check staleness and cancel if needed.

        Args:
            agent_id: Agent identifier
            req_id: Request identifier
            new_view_seq: New view sequence number

        Returns:
            True if cancellation occurred
        """
        key = (self.world_id, agent_id, req_id)
        if key not in self._cancel_tokens:
            return False

        token = self._cancel_tokens[key]
        if token.cancelled:
            return True

        # Get agent's current view sequence
        agent_handle = self.agent_handles.get(agent_id)
        if not agent_handle:
            return False

        # Check staleness threshold
        staleness = new_view_seq - agent_handle.view_seq
        if staleness > self.config.staleness_threshold:
            token.cancel(f"stale_due_to_seq_gap_{staleness}")
            self._logger.info(
                "Intent cancelled due to staleness",
                agent_id=agent_id,
                req_id=req_id,
                staleness=staleness,
                threshold=self.config.staleness_threshold,
            )
            return True

        return False

    def _next_seq(self) -> int:
        """Get next global sequence number.

        Returns:
            Next monotonically increasing sequence number
        """
        self._global_seq += 1
        return self._global_seq

    def _current_sim_time(self) -> float:
        """Get current simulation time based on authority.

        Returns:
            Current simulation time in seconds
        """
        if self._sim_time_authority == "none":
            # Use monotonic clock when no external authority
            return MonotonicClock.now()
        else:
            # TODO: Get time from external authority (Unity/Unreal)
            # For now, fall back to monotonic clock
            return MonotonicClock.now()

    def get_agent_count(self) -> int:
        """Get number of registered agents.

        Returns:
            Number of currently registered agents
        """
        return len(self.agent_handles)

    def get_latest_seq(self) -> int:
        """Get latest global sequence number.

        Returns:
            Latest global sequence number
        """
        return self._global_seq

    def get_world_state(self) -> WorldState:
        """Get current world state.

        Returns:
            Current world state (read-only)
        """
        return self.world_state

    async def shutdown(self) -> None:
        """Shutdown orchestrator and clean up resources."""
        if not self._initialized:
            return

        # Close all agent queues
        for queue in self._per_agent_queues.values():
            await queue.close()

        # Close deduplication store
        await self.dedup_store.close()

        # Clear state
        self.agent_handles.clear()
        self.observation_policies.clear()
        self._per_agent_queues.clear()
        self._cancel_tokens.clear()
        self._agent_processing_order.clear()
        self._initialized = False

        self._logger.info("Orchestrator shutdown complete")
