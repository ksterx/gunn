"""Core orchestrator for multi-agent simulation.

This module provides the central Orchestrator class that coordinates all system
operations including event ingestion, view generation, intent validation,
and observation distribution with deterministic ordering.
"""

import asyncio
import time
import uuid
from typing import Any, Literal, Protocol

from gunn.core.event_log import EventLog
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.messages import WorldState
from gunn.schemas.types import (
    CancelToken,
    Effect,
    EffectDraft,
    Intent,
    ObservationDelta,
)
from gunn.storage.dedup_store import DedupStore, InMemoryDedupStore
from gunn.utils.backpressure import backpressure_manager
from gunn.utils.errors import (
    BackpressureError,
    QuotaExceededError,
    StaleContextError,
    ValidationError,
)
from gunn.utils.memory import MemoryConfig, MemoryManager
from gunn.utils.scheduling import WeightedRoundRobinScheduler
from gunn.utils.telemetry import (
    MonotonicClock,
    async_performance_timer,
    bandwidth_monitor,
    get_logger,
    get_tracer,
    record_backpressure_event,
    record_conflict,
    record_intent_throughput,
    record_observation_delivery_latency,
    record_queue_depth,
    record_queue_high_watermark,
    system_monitor,
    update_active_agents_count,
    update_global_seq,
    update_view_seq,
)
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
    """Default implementation of EffectValidator with comprehensive validation.

    Provides validation for quota limits, cooldowns, permissions, and world state
    constraints to ensure intents can be safely executed.

    Requirements addressed:
    - 1.5: Intent validation through EffectValidator before creating an Effect
    - 3.4: Validation for quota limits, cooldowns, and permissions
    - 10.3: Structured error codes for validation failures
    """

    def __init__(
        self,
        max_intents_per_minute: int = 60,
        max_tokens_per_minute: int = 10000,
        default_cooldown_seconds: float = 1.0,
        max_payload_size_bytes: int = 10000,
        allowed_intent_kinds: set[str] | None = None,
    ):
        """Initialize validator with configuration.

        Args:
            max_intents_per_minute: Maximum intents per agent per minute
            max_tokens_per_minute: Maximum tokens per agent per minute
            default_cooldown_seconds: Default cooldown between intents
            max_payload_size_bytes: Maximum payload size in bytes
            allowed_intent_kinds: Set of allowed intent kinds (None = allow all)
        """
        self.max_intents_per_minute = max_intents_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute

        # Validation caching for performance optimization
        self._permission_cache: dict[str, set[str]] = {}
        self._validation_cache: dict[
            str, tuple[bool, float]
        ] = {}  # (is_valid, timestamp)
        self._cache_ttl = 1.0  # Cache TTL in seconds
        self.default_cooldown_seconds = default_cooldown_seconds
        self.max_payload_size_bytes = max_payload_size_bytes
        self.allowed_intent_kinds = allowed_intent_kinds or {
            "Speak",
            "Move",
            "Interact",
            "Custom",
        }

        # Tracking state for validation
        self._agent_intent_history: dict[str, list[float]] = {}
        self._agent_token_usage: dict[str, list[tuple[float, int]]] = {}
        self._agent_last_intent_time: dict[str, float] = {}
        self._agent_permissions: dict[str, set[str]] = {}
        self._intent_kind_cooldowns: dict[str, float] = {
            "Speak": 0.5,
            "Move": 0.1,
            "Interact": 2.0,
            "Custom": 1.0,
        }

    def set_agent_permissions(self, agent_id: str, permissions: set[str]) -> None:
        """Set permissions for an agent.

        Args:
            agent_id: Agent identifier
            permissions: Set of permission strings
        """
        self._agent_permissions[agent_id] = permissions.copy()
        # Invalidate cache for this agent
        self._permission_cache.pop(agent_id, None)
        # Invalidate validation cache entries for this agent
        keys_to_remove = [
            k for k in self._validation_cache.keys() if k.startswith(f"{agent_id}:")
        ]
        for key in keys_to_remove:
            self._validation_cache.pop(key, None)

    def add_agent_permission(self, agent_id: str, permission: str) -> None:
        """Add a permission for an agent.

        Args:
            agent_id: Agent identifier
            permission: Permission string to add
        """
        if agent_id not in self._agent_permissions:
            self._agent_permissions[agent_id] = set()
        self._agent_permissions[agent_id].add(permission)

    def remove_agent_permission(self, agent_id: str, permission: str) -> None:
        """Remove a permission for an agent.

        Args:
            agent_id: Agent identifier
            permission: Permission string to remove
        """
        if agent_id in self._agent_permissions:
            self._agent_permissions[agent_id].discard(permission)

    def set_intent_kind_cooldown(
        self, intent_kind: str, cooldown_seconds: float
    ) -> None:
        """Set cooldown for a specific intent kind.

        Args:
            intent_kind: Intent kind to set cooldown for
            cooldown_seconds: Cooldown duration in seconds
        """
        self._intent_kind_cooldowns[intent_kind] = cooldown_seconds

    def validate_intent(self, intent: Intent, world_state: WorldState) -> bool:
        """Validate if intent can be executed.

        Performs comprehensive validation including:
        - Basic structure validation
        - Permission checks
        - Quota limit validation
        - Cooldown enforcement
        - World state constraint checks
        - Payload size limits

        Args:
            intent: Intent to validate
            world_state: Current world state

        Returns:
            True if intent is valid and can be executed

        Raises:
            ValidationError: If validation fails with detailed reasons
        """
        validation_failures: list[str] = []
        current_time = time.time()
        _agent_id = intent.get("agent_id", "")
        _intent_kind = intent.get("kind", "")

        try:
            # 1. Basic structure validation
            validation_failures.extend(self._validate_structure(intent))

            # 2. Permission validation
            validation_failures.extend(self._validate_permissions(intent))

            # 3. Intent kind validation
            validation_failures.extend(self._validate_intent_kind(intent))

            # 4. Payload size validation
            validation_failures.extend(self._validate_payload_size(intent))

            # 5. Quota validation
            validation_failures.extend(
                self._validate_quota_limits(intent, current_time)
            )

            # 6. Cooldown validation
            validation_failures.extend(self._validate_cooldowns(intent, current_time))

            # 7. World state constraint validation
            validation_failures.extend(
                self._validate_world_state_constraints(intent, world_state)
            )

            # If any validation failed, raise ValidationError
            if validation_failures:
                from gunn.utils.errors import ValidationError

                raise ValidationError(intent, validation_failures)

            # Record successful validation for quota/cooldown tracking
            self._record_intent_validation(intent, current_time)

            return True

        except Exception as e:
            # Re-raise ValidationError as-is, wrap others
            if isinstance(e, ValidationError):
                raise

            # Wrap unexpected errors
            from gunn.utils.errors import ValidationError

            validation_failures.append(f"validation_error: {e!s}")
            raise ValidationError(intent, validation_failures) from e

    def _validate_structure(self, intent: Intent) -> list[str]:
        """Validate basic intent structure.

        Args:
            intent: Intent to validate

        Returns:
            List of validation failure messages
        """
        failures = []

        required_fields = ["agent_id", "kind", "req_id", "schema_version"]
        for field in required_fields:
            if not intent.get(field):
                failures.append(f"missing_required_field: {field}")

        # Validate agent_id format
        agent_id = intent.get("agent_id", "")
        if agent_id and (
            len(agent_id) > 100
            or not agent_id.replace("_", "").replace("-", "").isalnum()
        ):
            failures.append("invalid_agent_id_format")

        # Validate req_id format
        req_id = intent.get("req_id", "")
        if req_id and len(req_id) > 200:
            failures.append("req_id_too_long")

        # Validate priority range
        priority = intent.get("priority", 0)
        if not isinstance(priority, int) or priority < -100 or priority > 100:
            failures.append("invalid_priority_range")

        return failures

    def _validate_permissions(self, intent: Intent) -> list[str]:
        """Validate agent permissions for this intent.

        Args:
            intent: Intent to validate

        Returns:
            List of validation failure messages
        """
        failures = []
        agent_id = intent.get("agent_id", "")
        intent_kind = intent.get("kind", "")

        # Get agent permissions
        agent_permissions = self._agent_permissions.get(agent_id, set())

        # Check basic submit_intent permission
        if "submit_intent" not in agent_permissions:
            failures.append("missing_permission: submit_intent")

        # Check intent-kind-specific permissions
        kind_permission_map = {
            "Speak": "intent:speak",
            "Move": "intent:move",
            "Interact": "intent:interact",
            "Custom": "intent:custom",
        }

        required_permission = kind_permission_map.get(intent_kind)
        if required_permission and required_permission not in agent_permissions:
            failures.append(f"missing_permission: {required_permission}")

        return failures

    def _validate_intent_kind(self, intent: Intent) -> list[str]:
        """Validate intent kind is allowed.

        Args:
            intent: Intent to validate

        Returns:
            List of validation failure messages
        """
        failures = []
        intent_kind = intent.get("kind", "")

        if intent_kind not in self.allowed_intent_kinds:
            allowed_kinds = ", ".join(sorted(self.allowed_intent_kinds))
            failures.append(
                f"invalid_intent_kind: {intent_kind} (allowed: {allowed_kinds})"
            )

        return failures

    def _validate_payload_size(self, intent: Intent) -> list[str]:
        """Validate payload size limits.

        Args:
            intent: Intent to validate

        Returns:
            List of validation failure messages
        """
        failures = []
        payload = intent.get("payload", {})

        # Estimate payload size (rough JSON serialization size)
        import json

        try:
            payload_size = len(json.dumps(payload, separators=(",", ":")))
            if payload_size > self.max_payload_size_bytes:
                failures.append(
                    f"payload_too_large: {payload_size} > {self.max_payload_size_bytes} bytes"
                )
        except (TypeError, ValueError):
            failures.append("payload_not_serializable")

        return failures

    def _validate_quota_limits(self, intent: Intent, current_time: float) -> list[str]:
        """Validate quota limits for the agent.

        Args:
            intent: Intent to validate
            current_time: Current timestamp

        Returns:
            List of validation failure messages
        """
        failures = []
        agent_id = intent.get("agent_id", "")

        # Initialize tracking if needed
        if agent_id not in self._agent_intent_history:
            self._agent_intent_history[agent_id] = []
        if agent_id not in self._agent_token_usage:
            self._agent_token_usage[agent_id] = []

        # Clean up old entries (older than 1 minute)
        cutoff_time = current_time - 60.0
        self._agent_intent_history[agent_id] = [
            t for t in self._agent_intent_history[agent_id] if t > cutoff_time
        ]
        self._agent_token_usage[agent_id] = [
            (t, tokens)
            for t, tokens in self._agent_token_usage[agent_id]
            if t > cutoff_time
        ]

        # Check intent quota
        intent_count = len(self._agent_intent_history[agent_id])
        if intent_count >= self.max_intents_per_minute:
            failures.append(
                f"intent_quota_exceeded: {intent_count} >= {self.max_intents_per_minute}"
            )

        # Check token quota (estimate tokens from payload)
        payload = intent.get("payload", {})
        estimated_tokens = self._estimate_token_usage(payload)
        current_token_usage = sum(
            tokens for _, tokens in self._agent_token_usage[agent_id]
        )

        if current_token_usage + estimated_tokens > self.max_tokens_per_minute:
            failures.append(
                f"token_quota_exceeded: {current_token_usage + estimated_tokens} > {self.max_tokens_per_minute}"
            )

        return failures

    def _validate_cooldowns(self, intent: Intent, current_time: float) -> list[str]:
        """Validate cooldown constraints.

        Args:
            intent: Intent to validate
            current_time: Current timestamp

        Returns:
            List of validation failure messages
        """
        failures = []
        agent_id = intent.get("agent_id", "")
        intent_kind = intent.get("kind", "")

        # Check last intent time for this agent
        last_intent_time = self._agent_last_intent_time.get(agent_id, 0.0)

        # Get cooldown for this intent kind
        cooldown = self._intent_kind_cooldowns.get(
            intent_kind, self.default_cooldown_seconds
        )

        time_since_last = current_time - last_intent_time
        if time_since_last < cooldown:
            remaining_cooldown = cooldown - time_since_last
            failures.append(
                f"cooldown_active: {remaining_cooldown:.2f}s remaining for {intent_kind}"
            )

        return failures

    def _validate_world_state_constraints(
        self, intent: Intent, world_state: WorldState
    ) -> list[str]:
        """Validate world state constraints for the intent.

        Args:
            intent: Intent to validate
            world_state: Current world state

        Returns:
            List of validation failure messages
        """
        failures = []
        agent_id = intent.get("agent_id", "")
        intent_kind = intent.get("kind", "")
        payload = intent.get("payload", {})

        # Check if agent exists in world state
        if agent_id not in world_state.entities:
            failures.append(f"agent_not_in_world: {agent_id}")
            return failures  # Can't validate further without agent entity

        _agent_entity = world_state.entities[agent_id]

        # Validate based on intent kind
        if intent_kind == "Move":
            failures.extend(
                self._validate_move_constraints(agent_id, payload, world_state)
            )
        elif intent_kind == "Interact":
            failures.extend(
                self._validate_interact_constraints(agent_id, payload, world_state)
            )
        elif intent_kind == "Speak":
            failures.extend(
                self._validate_speak_constraints(agent_id, payload, world_state)
            )

        return failures

    def _validate_move_constraints(
        self, agent_id: str, payload: dict[str, Any], world_state: WorldState
    ) -> list[str]:
        """Validate movement constraints.

        Args:
            agent_id: Agent attempting to move
            payload: Move intent payload
            world_state: Current world state

        Returns:
            List of validation failure messages
        """
        failures = []

        # Check if agent has position
        if agent_id not in world_state.spatial_index:
            failures.append("agent_has_no_position")
            return failures

        current_pos = world_state.spatial_index[agent_id]

        # Validate target position format (support both 2D and 3D)
        target = payload.get("to")
        if not target or not isinstance(target, (list, tuple)) or len(target) < 2:
            failures.append("invalid_target_position_format")
            return failures

        try:
            # Convert to 3D if needed (2D -> add z=0)
            if len(target) == 2:
                target_pos = (float(target[0]), float(target[1]), 0.0)
            elif len(target) >= 3:
                target_pos = (float(target[0]), float(target[1]), float(target[2]))
            else:
                failures.append("invalid_target_position_format")
                return failures
        except (ValueError, TypeError):
            failures.append("invalid_target_position_values")
            return failures

        # Check movement distance (simple constraint)
        distance = (
            sum((a - b) ** 2 for a, b in zip(current_pos, target_pos, strict=False))
            ** 0.5
        )
        max_move_distance = 100.0  # Configurable constraint

        if distance > max_move_distance:
            failures.append(
                f"move_distance_too_large: {distance:.2f} > {max_move_distance}"
            )

        # Check for collision with other entities (simplified)
        for entity_id, entity_pos in world_state.spatial_index.items():
            if entity_id == agent_id:
                continue

            entity_distance = (
                sum((a - b) ** 2 for a, b in zip(target_pos, entity_pos, strict=False))
                ** 0.5
            )
            min_distance = 1.0  # Minimum distance between entities

            if entity_distance < min_distance:
                failures.append(f"target_position_occupied: too close to {entity_id}")
                break

        return failures

    def _validate_interact_constraints(
        self, agent_id: str, payload: dict[str, Any], world_state: WorldState
    ) -> list[str]:
        """Validate interaction constraints.

        Args:
            agent_id: Agent attempting to interact
            payload: Interact intent payload
            world_state: Current world state

        Returns:
            List of validation failure messages
        """
        failures = []

        target_id = payload.get("target")
        if not target_id:
            failures.append("missing_interaction_target")
            return failures

        # Check if target exists
        if target_id not in world_state.entities:
            failures.append(f"interaction_target_not_found: {target_id}")
            return failures

        # Check if agent and target are close enough (if both have positions)
        if (
            agent_id in world_state.spatial_index
            and target_id in world_state.spatial_index
        ):
            agent_pos = world_state.spatial_index[agent_id]
            target_pos = world_state.spatial_index[target_id]

            distance = (
                sum((a - b) ** 2 for a, b in zip(agent_pos, target_pos, strict=False))
                ** 0.5
            )
            max_interact_distance = 5.0  # Configurable constraint

            if distance > max_interact_distance:
                failures.append(
                    f"interaction_target_too_far: {distance:.2f} > {max_interact_distance}"
                )

        # Check interaction type constraints
        interaction_type = payload.get("type", "")
        if interaction_type and interaction_type not in [
            "examine",
            "use",
            "talk",
            "trade",
        ]:
            failures.append(f"invalid_interaction_type: {interaction_type}")

        return failures

    def _validate_speak_constraints(
        self, agent_id: str, payload: dict[str, Any], world_state: WorldState
    ) -> list[str]:
        """Validate speaking constraints.

        Args:
            agent_id: Agent attempting to speak
            payload: Speak intent payload
            world_state: Current world state

        Returns:
            List of validation failure messages
        """
        failures = []

        message = payload.get("message")
        if message is None or not isinstance(message, str) or not message.strip():
            failures.append("missing_or_invalid_message")
            return failures

        # Check message length
        max_message_length = 1000  # Configurable constraint
        if len(message) > max_message_length:
            failures.append(f"message_too_long: {len(message)} > {max_message_length}")

        # Check for prohibited content (simplified)
        prohibited_words = ["spam", "abuse"]  # Configurable list
        message_lower = message.lower()
        for word in prohibited_words:
            if word in message_lower:
                failures.append(f"prohibited_content: {word}")

        return failures

    def _estimate_token_usage(self, payload: dict[str, Any]) -> int:
        """Estimate token usage for a payload.

        Args:
            payload: Intent payload

        Returns:
            Estimated token count
        """
        # Simple estimation: 1 token per 4 characters
        import json

        try:
            payload_str = json.dumps(payload, separators=(",", ":"))
            return max(1, len(payload_str) // 4)
        except (TypeError, ValueError):
            return 10  # Default estimate for non-serializable payloads

    def _record_intent_validation(self, intent: Intent, current_time: float) -> None:
        """Record successful intent validation for tracking.

        Args:
            intent: Validated intent
            current_time: Current timestamp
        """
        agent_id = intent.get("agent_id", "")
        payload = intent.get("payload", {})

        # Record intent timestamp
        if agent_id not in self._agent_intent_history:
            self._agent_intent_history[agent_id] = []
        self._agent_intent_history[agent_id].append(current_time)

        # Record token usage
        estimated_tokens = self._estimate_token_usage(payload)
        if agent_id not in self._agent_token_usage:
            self._agent_token_usage[agent_id] = []
        self._agent_token_usage[agent_id].append((current_time, estimated_tokens))

        # Update last intent time
        self._agent_last_intent_time[agent_id] = current_time

    def get_agent_quota_status(self, agent_id: str) -> dict[str, Any]:
        """Get current quota status for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary with quota status information
        """
        current_time = time.time()
        cutoff_time = current_time - 60.0

        # Clean up old entries
        intent_history = self._agent_intent_history.get(agent_id, [])
        intent_history = [t for t in intent_history if t > cutoff_time]

        token_usage = self._agent_token_usage.get(agent_id, [])
        token_usage = [(t, tokens) for t, tokens in token_usage if t > cutoff_time]

        current_tokens = sum(tokens for _, tokens in token_usage)
        last_intent_time = self._agent_last_intent_time.get(agent_id, 0.0)

        return {
            "agent_id": agent_id,
            "intents_used": len(intent_history),
            "intents_limit": self.max_intents_per_minute,
            "intents_remaining": max(
                0, self.max_intents_per_minute - len(intent_history)
            ),
            "tokens_used": current_tokens,
            "tokens_limit": self.max_tokens_per_minute,
            "tokens_remaining": max(0, self.max_tokens_per_minute - current_tokens),
            "last_intent_time": last_intent_time,
            "time_since_last_intent": current_time - last_intent_time,
            "permissions": list(self._agent_permissions.get(agent_id, set())),
        }


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

        async with async_performance_timer(
            "next_observation",
            agent_id=self.agent_id,
            view_seq=self.view_seq,
            logger=self.orchestrator._logger,
            tracer_name="gunn.agent_handle",
        ):
            timed_queue = self.orchestrator._per_agent_queues[self.agent_id]
            delta = await timed_queue.get()

            # Update view sequence from delta
            if hasattr(delta, "view_seq"):
                self.view_seq = delta.view_seq
            elif isinstance(delta, dict) and "view_seq" in delta:
                self.view_seq = delta["view_seq"]

            # Update view sequence metric
            update_view_seq(self.agent_id, self.view_seq)

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
        dedup_cleanup_interval_minutes: int = 10,
        dedup_warmup_minutes: int = 5,
        max_queue_depth: int = 100,
        quota_intents_per_minute: int = 60,
        quota_tokens_per_minute: int = 10000,
        use_in_memory_dedup: bool = False,
        processing_idle_shutdown_ms: float = 250.0,
        # Memory management settings
        max_log_entries: int = 10000,
        view_cache_size: int = 1000,
        compaction_threshold: int = 5000,
        snapshot_interval: int = 1000,
        max_snapshots: int = 10,
        memory_check_interval_seconds: float = 60.0,
        auto_compaction_enabled: bool = True,
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
            dedup_cleanup_interval_minutes: Cleanup interval for dedup store
            dedup_warmup_minutes: Warmup period for relaxed deduplication
            max_queue_depth: Maximum queue depth per agent
            quota_intents_per_minute: Intent quota per agent per minute
            quota_tokens_per_minute: Token quota per agent per minute
            use_in_memory_dedup: Use in-memory dedup store for testing
            processing_idle_shutdown_ms: Idle duration before background intent loop auto-stops (<=0 disables)
            max_log_entries: Maximum number of log entries before compaction
            view_cache_size: Maximum size of view cache
            compaction_threshold: Threshold for triggering log compaction
            snapshot_interval: Create snapshot every N events
            max_snapshots: Maximum number of snapshots to keep
            memory_check_interval_seconds: Interval between memory checks
            auto_compaction_enabled: Enable automatic log compaction
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
        self.dedup_cleanup_interval_minutes = dedup_cleanup_interval_minutes
        self.dedup_warmup_minutes = dedup_warmup_minutes
        self.max_queue_depth = max_queue_depth
        self.quota_intents_per_minute = quota_intents_per_minute
        self.quota_tokens_per_minute = quota_tokens_per_minute
        self.use_in_memory_dedup = use_in_memory_dedup
        self.processing_idle_shutdown_ms = processing_idle_shutdown_ms
        # Memory management settings
        self.max_log_entries = max_log_entries
        self.view_cache_size = view_cache_size
        self.compaction_threshold = compaction_threshold
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.memory_check_interval_seconds = memory_check_interval_seconds
        self.auto_compaction_enabled = auto_compaction_enabled


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
    ):
        """Initialize orchestrator with configuration and dependencies.

        Args:
            config: Orchestrator configuration
            world_id: Identifier for this world instance
            effect_validator: Optional custom effect validator
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

        # Initialization tracking for lazy startup
        self._initialized: bool = False
        self._initialize_lock: asyncio.Lock | None = None

        # Two-phase commit infrastructure
        if config.use_in_memory_dedup:
            self._dedup_store: DedupStore | InMemoryDedupStore = InMemoryDedupStore(
                dedup_ttl_minutes=config.dedup_ttl_minutes,
                max_entries=config.max_dedup_entries,
                warmup_duration_minutes=config.dedup_warmup_minutes,
            )
        else:
            self._dedup_store = DedupStore(
                db_path=f"{world_id}_dedup.db",
                dedup_ttl_minutes=config.dedup_ttl_minutes,
                max_entries=config.max_dedup_entries,
                cleanup_interval_minutes=config.dedup_cleanup_interval_minutes,
                warmup_duration_minutes=config.dedup_warmup_minutes,
            )

        # Intent processing pipeline
        self._scheduler = WeightedRoundRobinScheduler(
            max_queue_depth=config.max_queue_depth,
            priority_levels=3,  # 0=high, 1=normal, 2=low
        )

        # Backpressure management
        self._backpressure_manager = backpressure_manager
        self._agent_backpressure_policies: dict[
            str, str
        ] = {}  # agent_id -> policy_name

        # Quota tracking
        self._quota_tracker: dict[
            str, dict[str, list[float]]
        ] = {}  # agent_id -> {intents: timestamps, tokens: timestamps}

        # Internal state
        self._global_seq: int = 0
        self._cancel_tokens: dict[
            tuple[str, str, str], CancelToken
        ] = {}  # (world_id, agent_id, req_id)
        self._per_agent_queues: dict[str, TimedQueue] = {}  # Timed delivery queues
        self._sim_time_authority: str = "none"  # Which adapter controls sim_time
        self._processing_task: asyncio.Task[None] | None = None
        self._system_monitoring_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        # Cancellation and staleness detection enhancements
        self._last_cancellation_time: dict[str, float] = {}  # agent_id -> timestamp
        self._agent_interrupt_policies: dict[
            str, str
        ] = {}  # agent_id -> policy ("always" | "only_conflict")

        # Memory management
        memory_config = MemoryConfig(
            max_log_entries=config.max_log_entries,
            view_cache_size=config.view_cache_size,
            compaction_threshold=config.compaction_threshold,
            snapshot_interval=config.snapshot_interval,
            max_snapshots=config.max_snapshots,
            memory_check_interval_seconds=config.memory_check_interval_seconds,
            auto_compaction_enabled=config.auto_compaction_enabled,
        )
        self.memory_manager = MemoryManager(memory_config)

        # Logging and telemetry
        self._logger = get_logger("gunn.orchestrator", world_id=world_id)
        self._tracer = get_tracer("gunn.orchestrator")

        # Start system monitoring
        self._start_system_monitoring()

        self._logger.info(
            "Orchestrator initialized",
            world_id=world_id,
            max_agents=config.max_agents,
            sim_time_authority=self._sim_time_authority,
            dedup_store_type=type(self._dedup_store).__name__,
        )

    async def initialize(self) -> None:
        """Initialize orchestrator and start processing pipeline if needed."""
        await self._ensure_initialized()
        self._start_processing_loop()

    async def _ensure_initialized(self) -> None:
        """Lazily initialize orchestrator internals exactly once."""
        if self._initialized:
            return

        if self._initialize_lock is None:
            self._initialize_lock = asyncio.Lock()

        async with self._initialize_lock:
            if self._initialized:
                return  # type: ignore

            # Initialize deduplication store and reset shutdown state
            await self._dedup_store.initialize()
            self._shutdown_event.clear()

            # Start system monitoring task
            if hasattr(self, "_monitor_system_func"):
                self._system_monitoring_task = asyncio.create_task(
                    self._monitor_system_func()
                )

            self._initialized = True
            self._logger.info("Orchestrator dependencies initialized")

    def _start_processing_loop(self) -> None:
        """Ensure background intent processing task is running."""
        if not self._initialized:
            return

        if self._processing_task and not self._processing_task.done():
            return

        self._processing_task = asyncio.create_task(self._intent_processing_loop())

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
        self,
        agent_id: str,
        policy: ObservationPolicy,
        permissions: set[str] | None = None,
    ) -> AgentHandle:
        """Register a new agent with observation policy and permissions.

        Args:
            agent_id: Unique identifier for the agent
            policy: Observation policy for this agent
            permissions: Set of permissions for the agent. If None, uses default permissions.

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

        # Set agent permissions (use defaults if not provided)
        if permissions is None:
            # Default permissions for newly registered agents
            permissions = {
                "submit_intent",
                "intent:move",
                "intent:speak",
                "intent:interact",
                "intent:custom",
            }

        self.effect_validator.set_agent_permissions(agent_id, permissions)

        # Update active agents count metric
        update_active_agents_count(len(self.agent_handles))

        self._logger.info(
            "Agent registered",
            agent_id=agent_id,
            total_agents=len(self.agent_handles),
            policy_type=type(policy).__name__,
            permissions=list(permissions),
        )

        return handle

    def set_agent_backpressure_policy(self, agent_id: str, policy_name: str) -> None:
        """Set backpressure policy for a specific agent.

        Args:
            agent_id: Agent identifier
            policy_name: Backpressure policy name (defer, shed_oldest, drop_newest)

        Raises:
            ValueError: If agent_id is not registered or policy_name is invalid

        Requirements addressed:
        - 10.2: Configurable backpressure policies per agent class
        """
        if agent_id not in self.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        if policy_name not in self._backpressure_manager.available_policies:
            available = ", ".join(self._backpressure_manager.available_policies)
            raise ValueError(
                f"Invalid backpressure policy '{policy_name}'. Available: {available}"
            )

        self._agent_backpressure_policies[agent_id] = policy_name

        self._logger.info(
            "Agent backpressure policy updated",
            agent_id=agent_id,
            policy=policy_name,
        )

    async def broadcast_event(self, draft: EffectDraft) -> None:
        """Create complete effect from draft and distribute observations.

        Converts EffectDraft to complete Effect with priority completion,
        updates world state, generates observation deltas for affected agents,
        and delivers observations using timed queues with latency models.

        Args:
            draft: Effect draft to process and broadcast

        Raises:
            ValueError: If draft is missing required fields

        Requirements addressed:
        - 2.2: Generate ObservationDelta patches for affected agents
        - 2.5: Incremental updates via ObservationDelta patches
        - 6.4: ObservationDelta delivery latency ≤ 20ms
        - 6.5: Timed delivery using per-agent TimedQueues with latency models
        """
        # Enhanced telemetry for broadcast_event
        async with async_performance_timer(
            "broadcast_event",
            agent_id=draft.get("source_id"),
            global_seq=self._global_seq + 1,
            logger=self._logger,
            tracer_name="gunn.orchestrator",
        ):
            # Validate draft
            if not draft.get("kind"):
                raise ValueError("EffectDraft must have 'kind' field")
            if not draft.get("source_id"):
                raise ValueError("EffectDraft must have 'source_id' field")
            if not draft.get("schema_version"):
                raise ValueError("EffectDraft must have 'schema_version' field")

            # Priority completion: use config.default_priority if not specified
            payload = draft.get("payload", {})
            if "priority" not in payload:
                payload = payload.copy()
                payload["priority"] = self.config.default_priority

            # Create complete effect with orchestrator-managed fields
            effect: Effect = {
                "uuid": uuid.uuid4().hex,
                "kind": draft["kind"],
                "payload": payload,
                "source_id": draft["source_id"],
                "schema_version": draft["schema_version"],
                "sim_time": self._current_sim_time(),
                "global_seq": self._next_seq(),
            }

            # Append to event log with world_id in source_metadata
            await self.event_log.append(effect, req_id=None)

            # Update world state based on effect
            await self._apply_effect_to_world_state(effect)

            # Check if snapshot should be created and create it if needed
            await self.memory_manager.check_and_create_snapshot(
                effect["global_seq"], self.world_state, effect["sim_time"]
            )

            # Check memory limits and trigger compaction if needed
            await self.memory_manager.check_memory_limits(self.event_log)

            # Generate and distribute observations to affected agents
            await self._distribute_observations(effect)

    async def submit_intent(self, intent: Intent) -> str:
        """Two-phase commit: idempotency → quota → backpressure → priority → fairness → validator → commit.

        Args:
            intent: Intent to process

        Returns:
            Request ID for tracking

        Raises:
            ValueError: If intent is invalid
            StaleContextError: If intent context is stale
            QuotaExceededError: If agent quota is exceeded
            BackpressureError: If backpressure limits are exceeded
            ValidationError: If intent validation fails
        """
        # Validate intent structure
        if not intent.get("req_id"):
            raise ValueError("Intent must have 'req_id' field")
        if not intent.get("agent_id"):
            raise ValueError("Intent must have 'agent_id' field")
        if not intent.get("kind"):
            raise ValueError("Intent must have 'kind' field")

        await self._ensure_initialized()

        req_id = intent["req_id"]
        agent_id = intent["agent_id"]
        context_seq = intent.get("context_seq", 0)

        # Enhanced telemetry for submit_intent
        async with async_performance_timer(
            "submit_intent",
            agent_id=agent_id,
            req_id=req_id,
            global_seq=self._global_seq,
            view_seq=context_seq,
            logger=self._logger,
            tracer_name="gunn.orchestrator",
        ):
            try:
                # Phase 1: Idempotency check using persistent store
                existing_seq = await self._dedup_store.check_and_record(
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

                # Phase 2: Agent validation
                if agent_id not in self.agent_handles:
                    raise ValueError(f"Agent {agent_id} is not registered")

                # Phase 3: Staleness detection
                _ = self.agent_handles[agent_id]
                staleness = self._global_seq - context_seq
                if staleness > self.config.staleness_threshold:
                    # Record staleness conflict
                    record_conflict(agent_id, "staleness")
                    raise StaleContextError(
                        req_id,
                        context_seq,
                        self._global_seq,
                        self.config.staleness_threshold,
                    )

                # Phase 4: Quota checking
                await self._check_quota(agent_id, intent)

                # Phase 5: Backpressure checking
                await self._check_backpressure(agent_id)

                # Phase 6: Priority assignment and fairness scheduling
                priority = intent.get("priority", self.config.default_priority)
                # Normalize priority to 0-2 range (0=high, 1=normal, 2=low)
                normalized_priority = max(0, min(2, 2 - priority // 10))

                # Enqueue for fair processing
                self._scheduler.enqueue(intent, normalized_priority)

                self._logger.info(
                    "Intent enqueued for processing",
                    req_id=req_id,
                    agent_id=agent_id,
                    priority=priority,
                    normalized_priority=normalized_priority,
                    queue_depth=self._scheduler.get_queue_depth(agent_id),
                )

                self._start_processing_loop()

                # Record successful intent throughput
                record_intent_throughput(agent_id, intent["kind"], "success")

                return req_id

            except StaleContextError:
                record_intent_throughput(agent_id, intent["kind"], "stale")
                raise
            except QuotaExceededError:
                record_intent_throughput(agent_id, intent["kind"], "quota_exceeded")
                record_conflict(agent_id, "quota")
                raise
            except BackpressureError:
                record_intent_throughput(agent_id, intent["kind"], "backpressure")
                record_conflict(agent_id, "backpressure")
                raise
            except ValidationError:
                record_intent_throughput(agent_id, intent["kind"], "validation_failed")
                record_conflict(agent_id, "validation")
                raise
            except Exception:
                record_intent_throughput(agent_id, intent["kind"], "error")
                raise

    async def _check_quota(self, agent_id: str, intent: Intent) -> None:
        """Check if agent has quota available for this intent.

        Args:
            agent_id: Agent identifier
            intent: Intent to check quota for

        Raises:
            QuotaExceededError: If quota is exceeded
        """
        current_time = time.time()

        # Initialize quota tracking for agent if needed
        if agent_id not in self._quota_tracker:
            self._quota_tracker[agent_id] = {
                "intents": [],
                "tokens": [],
            }

        agent_quota = self._quota_tracker[agent_id]

        # Clean up old quota entries (older than 1 minute)
        cutoff_time = current_time - 60.0
        agent_quota["intents"] = [t for t in agent_quota["intents"] if t > cutoff_time]
        agent_quota["tokens"] = [t for t in agent_quota["tokens"] if t > cutoff_time]

        # Check intent quota
        if len(agent_quota["intents"]) >= self.config.quota_intents_per_minute:
            raise QuotaExceededError(
                agent_id,
                "intents_per_minute",
                self.config.quota_intents_per_minute,
                len(agent_quota["intents"]),
            )

        # Estimate token usage (simplified)
        estimated_tokens = len(str(intent.get("payload", {}))) // 4  # Rough estimate
        if (
            len(agent_quota["tokens"]) + estimated_tokens
            > self.config.quota_tokens_per_minute
        ):
            raise QuotaExceededError(
                agent_id,
                "tokens_per_minute",
                self.config.quota_tokens_per_minute,
                len(agent_quota["tokens"]) + estimated_tokens,
            )

        # Record quota usage
        agent_quota["intents"].append(current_time)
        for _ in range(estimated_tokens):
            agent_quota["tokens"].append(current_time)

    async def _check_backpressure(self, agent_id: str) -> None:
        """Check backpressure limits for agent.

        Args:
            agent_id: Agent identifier

        Raises:
            BackpressureError: If backpressure limits are exceeded
        """
        agent_queue_depth = self._scheduler.get_queue_depth(agent_id)

        # Get agent-specific backpressure policy or use default
        policy_name = self._agent_backpressure_policies.get(
            agent_id, self.config.backpressure_policy
        )

        # Check per-agent queue depth
        if agent_queue_depth >= self.config.max_queue_depth:
            # Record high watermark and backpressure event
            record_queue_high_watermark(agent_id, "agent_queue", agent_queue_depth)
            record_backpressure_event(agent_id, "agent_queue", policy_name)

            raise BackpressureError(
                agent_id,
                "agent_queue",
                agent_queue_depth,
                self.config.max_queue_depth,
                policy_name,
            )

        # Check total system queue depth
        total_depth = self._scheduler.get_queue_depth()
        system_threshold = self.config.max_queue_depth * self.config.max_agents // 2

        if total_depth >= system_threshold:
            # Record high watermark and backpressure event
            record_queue_high_watermark(agent_id, "system_queue", total_depth)
            record_backpressure_event(agent_id, "system_queue", policy_name)

            raise BackpressureError(
                agent_id,
                "system_queue",
                total_depth,
                system_threshold,
                policy_name,
            )

    def issue_cancel_token(
        self, agent_id: str, req_id: str, context_digest: str | None = None
    ) -> CancelToken:
        """Issue cancellation token for generation tracking.

        Creates a new cancellation token with tuple key tracking for the given
        agent and request. The token can be used to cancel long-running operations
        like LLM generation when context becomes stale.

        Args:
            agent_id: Agent identifier
            req_id: Request identifier
            context_digest: Optional context digest for staleness detection

        Returns:
            CancelToken for tracking cancellation

        Raises:
            ValueError: If agent_id or req_id is empty

        Requirements addressed:
        - 4.1: Issue cancel_token with current context_digest
        """
        if not agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        if not req_id.strip():
            raise ValueError("req_id cannot be empty")

        # Create token with context information
        token = CancelToken(req_id, agent_id)
        key = (self.world_id, agent_id, req_id)

        # Clean up any existing token for this key
        if key in self._cancel_tokens:
            old_token = self._cancel_tokens[key]
            if not old_token.cancelled:
                old_token.cancel("replaced_by_new_token")

        self._cancel_tokens[key] = token

        self._logger.info(
            "Cancel token issued",
            agent_id=agent_id,
            req_id=req_id,
            context_digest=context_digest,
            total_active_tokens=len(
                [t for t in self._cancel_tokens.values() if not t.cancelled]
            ),
        )

        return token

    async def cancel_if_stale(
        self, agent_id: str, req_id: str, new_view_seq: int
    ) -> bool:
        """Check staleness and cancel if needed with debounce logic.

        Evaluates whether the context has become stale based on the staleness
        threshold and applies debounce logic to prevent rapid successive
        interruptions.

        Args:
            agent_id: Agent identifier
            req_id: Request identifier
            new_view_seq: New view sequence number to check against

        Returns:
            True if cancellation occurred

        Requirements addressed:
        - 4.2: Evaluate staleness using latest_view_seq > context_seq + staleness_threshold
        - 4.3: Cancel current generation when context becomes stale
        - 4.7: Suppress rapid successive interruptions within debounce window
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

        # Check debounce logic to prevent rapid successive interruptions
        current_time = time.time()
        last_cancellation = self._last_cancellation_time.get(agent_id, 0)
        debounce_window_s = self.config.debounce_ms / 1000.0

        if current_time - last_cancellation < debounce_window_s:
            self._logger.debug(
                "Cancellation suppressed by debounce",
                agent_id=agent_id,
                req_id=req_id,
                time_since_last_ms=(current_time - last_cancellation) * 1000,
                debounce_ms=self.config.debounce_ms,
            )
            return False

        # Check staleness threshold
        staleness = new_view_seq - agent_handle.view_seq

        if staleness > self.config.staleness_threshold:
            # Record cancellation time for debounce
            self._last_cancellation_time[agent_id] = current_time

            # Cancel with detailed reason
            reason = f"stale_due_to_seq_gap_{staleness}_threshold_{self.config.staleness_threshold}"
            token.cancel(reason)

            self._logger.info(
                "Intent cancelled due to staleness",
                agent_id=agent_id,
                req_id=req_id,
                staleness=staleness,
                threshold=self.config.staleness_threshold,
                new_view_seq=new_view_seq,
                agent_view_seq=agent_handle.view_seq,
                reason=reason,
            )
            return True

        return False

    def set_agent_interrupt_policy(self, agent_id: str, policy: str) -> None:
        """Set interrupt policy for an agent.

        Args:
            agent_id: Agent identifier
            policy: Interrupt policy ("always" or "only_conflict")

        Raises:
            ValueError: If policy is not valid

        Requirements addressed:
        - 4.5: interrupt_on_new_info policy "always" triggers on any new information
        - 4.6: interrupt_on_new_info policy "only_conflict" triggers only on conflicts
        """
        if policy not in ("always", "only_conflict"):
            raise ValueError(
                f"Invalid interrupt policy: {policy}. Must be 'always' or 'only_conflict'"
            )

        self._agent_interrupt_policies[agent_id] = policy

        self._logger.info(
            "Agent interrupt policy set",
            agent_id=agent_id,
            policy=policy,
        )

    def get_agent_interrupt_policy(self, agent_id: str) -> str:
        """Get interrupt policy for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Interrupt policy ("always" or "only_conflict"), defaults to "always"
        """
        return self._agent_interrupt_policies.get(agent_id, "always")

    async def check_and_cancel_stale_tokens(self, effect: Effect) -> list[str]:
        """Check all active tokens for staleness and cancel if needed.

        This method is called during observation distribution to automatically
        cancel tokens when new events make their context stale.

        Args:
            effect: New effect that might make contexts stale

        Returns:
            List of request IDs that were cancelled

        Requirements addressed:
        - 4.2: Evaluate staleness when new events occur during generation
        - 4.3: Cancel current generation when context becomes stale
        - 4.4: Provide cancellation reason to the agent
        """
        cancelled_req_ids = []
        current_global_seq = effect["global_seq"]

        # Check all active cancel tokens
        for key, token in list(self._cancel_tokens.items()):
            if token.cancelled:
                continue

            world_id, agent_id, req_id = key
            if world_id != self.world_id:
                continue

            # Get agent's interrupt policy
            interrupt_policy = self.get_agent_interrupt_policy(agent_id)

            # Check if this effect should trigger interruption
            should_interrupt = False

            if interrupt_policy == "always":
                # Any new information triggers interruption
                should_interrupt = True
            elif interrupt_policy == "only_conflict":
                # Only conflicting information triggers interruption
                # For now, we'll consider any effect from a different agent as potentially conflicting
                # This can be enhanced with more sophisticated conflict detection
                should_interrupt = effect["source_id"] != agent_id

            if should_interrupt:
                # Use the enhanced cancel_if_stale method
                # The new_view_seq should be the current global_seq since that's what the agent will see
                was_cancelled = await self.cancel_if_stale(
                    agent_id, req_id, current_global_seq
                )
                if was_cancelled:
                    cancelled_req_ids.append(req_id)

        if cancelled_req_ids:
            self._logger.info(
                "Automatic cancellation due to new events",
                effect_kind=effect["kind"],
                effect_source=effect["source_id"],
                global_seq=current_global_seq,
                cancelled_count=len(cancelled_req_ids),
                cancelled_req_ids=cancelled_req_ids,
            )

        return cancelled_req_ids

    def cleanup_cancelled_tokens(self) -> int:
        """Clean up cancelled tokens to prevent memory leaks.

        Returns:
            Number of tokens cleaned up
        """
        initial_count = len(self._cancel_tokens)

        # Remove cancelled tokens
        self._cancel_tokens = {
            key: token
            for key, token in self._cancel_tokens.items()
            if not token.cancelled
        }

        cleaned_count = initial_count - len(self._cancel_tokens)

        if cleaned_count > 0:
            self._logger.debug(
                "Cleaned up cancelled tokens",
                cleaned_count=cleaned_count,
                remaining_count=len(self._cancel_tokens),
            )

        return cleaned_count

    def get_active_cancel_tokens(self) -> dict[str, list[str]]:
        """Get active cancel tokens grouped by agent.

        Returns:
            Dictionary mapping agent_id to list of active req_ids
        """
        active_tokens: dict[str, list[str]] = {}

        for key, token in self._cancel_tokens.items():
            if not token.cancelled:
                world_id, agent_id, req_id = key
                if world_id == self.world_id:
                    if agent_id not in active_tokens:
                        active_tokens[agent_id] = []
                    active_tokens[agent_id].append(req_id)

        return active_tokens

    def _next_seq(self) -> int:
        """Get next global sequence number.

        Returns:
            Next monotonically increasing sequence number
        """
        self._global_seq += 1

        # Update global sequence metric
        update_global_seq(self._global_seq)

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

    async def _intent_processing_loop(self) -> None:
        """Background loop for processing intents from the scheduler."""
        self._logger.info("Intent processing loop started")

        last_cleanup_time = time.time()
        cleanup_interval_s = 30.0  # Clean up cancelled tokens every 30 seconds

        try:
            _ = time.perf_counter()
            while not self._shutdown_event.is_set():
                try:
                    # Periodic cleanup of cancelled tokens
                    current_time = time.time()
                    if current_time - last_cleanup_time >= cleanup_interval_s:
                        cleaned_count = self.cleanup_cancelled_tokens()
                        last_cleanup_time = current_time

                        if cleaned_count > 0:
                            self._logger.debug(
                                "Periodic token cleanup completed",
                                cleaned_count=cleaned_count,
                                remaining_tokens=len(self._cancel_tokens),
                            )

                    # Batch dequeue multiple intents for parallel processing
                    intents_to_process = []
                    max_batch_size = 5  # Process up to 5 intents in parallel

                    # Collect available intents up to batch size
                    for _ in range(max_batch_size):
                        intent = self._scheduler.dequeue()
                        if intent is None:
                            break
                        intents_to_process.append(intent)

                    if not intents_to_process:
                        # No intents to process, wait a bit (reduced from 10ms to 1ms)
                        await asyncio.sleep(
                            0.001
                        )  # 1ms - significantly reduces idle time
                        continue

                    # Process intents in parallel
                    tasks = [
                        asyncio.create_task(self._process_intent(intent))
                        for intent in intents_to_process
                    ]

                    # Wait for all tasks to complete
                    await asyncio.gather(*tasks, return_exceptions=True)

                except Exception as e:
                    self._logger.error(
                        "Error in intent processing loop",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    # Continue processing other intents
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self._logger.info("Intent processing loop cancelled")
            raise
        except Exception as e:
            self._logger.error(
                "Intent processing loop failed",
                error=str(e),
                error_type=type(e).__name__,
            )
        finally:
            self._processing_task = None
            if not self._shutdown_event.is_set():
                # Auto-stop triggered by idle timeout: close dedup store to avoid hanging tasks
                try:
                    await self._dedup_store.close()
                finally:
                    self._initialized = False
                    self._initialize_lock = None

                self._logger.info(
                    "Intent processing loop exited (idle auto-stop)",
                    idle_timeout_ms=self.config.processing_idle_shutdown_ms,
                )
            else:
                self._logger.info("Intent processing loop exited")

    async def _process_intent(self, intent: Intent) -> None:
        """Process a single intent through validation and effect creation.

        Args:
            intent: Intent to process
        """
        req_id = intent["req_id"]
        agent_id = intent["agent_id"]
        start_time = time.perf_counter()

        try:
            # Phase 7: Validation
            validation_failures = []

            if not self.effect_validator.validate_intent(intent, self.world_state):
                validation_failures.append("effect_validator_failed")

            # Additional validation checks can be added here

            if validation_failures:
                raise ValidationError(intent, validation_failures)

            # Phase 8: Commit - Create effect from validated intent
            effect_draft: EffectDraft = {
                "kind": intent["kind"],
                "payload": intent["payload"],
                "source_id": agent_id,
                "schema_version": intent.get("schema_version", "1.0.0"),
            }

            # Broadcast the effect (this will assign the global_seq)
            await self.broadcast_event(effect_draft)

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            self._logger.info(
                "Intent processed successfully",
                req_id=req_id,
                agent_id=agent_id,
                intent_kind=intent["kind"],
                global_seq=self._global_seq,
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.error(
                "Intent processing failed",
                req_id=req_id,
                agent_id=agent_id,
                error=str(e),
                error_type=type(e).__name__,
                processing_time_ms=processing_time_ms,
            )
            # Intent processing failure - could implement retry logic here

    def get_processing_stats(self) -> dict[str, Any]:
        """Get intent processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        scheduler_stats = self._scheduler.get_stats()

        return {
            "scheduler": scheduler_stats,
            "quota_tracker": {
                agent_id: {
                    "intents_count": len(quotas["intents"]),
                    "tokens_count": len(quotas["tokens"]),
                }
                for agent_id, quotas in self._quota_tracker.items()
            },
            "global_seq": self._global_seq,
            "cancel_tokens_active": len(self._cancel_tokens),
            "agents_registered": len(self.agent_handles),
        }

    async def _apply_effect_to_world_state(self, effect: Effect) -> None:
        """Apply effect to world state based on effect kind and payload.

        Args:
            effect: Effect to apply to world state
        """
        effect_kind = effect["kind"]
        payload = effect["payload"]
        source_id = effect["source_id"]

        try:
            if effect_kind == "Move":
                # Handle unified position format: "to" field (new) or "position" field (legacy)
                position = payload.get("to") or payload.get("position")

                if position and isinstance(position, (list, tuple)):
                    # Convert 2D to 3D if needed
                    if len(position) == 2:
                        position = [float(position[0]), float(position[1]), 0.0]
                    elif len(position) >= 3:
                        position = [
                            float(position[0]),
                            float(position[1]),
                            float(position[2]),
                        ]
                    else:
                        position = None

                    if position:
                        # Update spatial index
                        self.world_state.spatial_index[source_id] = tuple(position)

                        # Update entity data
                        if source_id not in self.world_state.entities:
                            self.world_state.entities[source_id] = {}

                        entity_data = self.world_state.entities[source_id]
                        if isinstance(entity_data, dict):
                            entity_data.update(
                                {
                                    "last_position": position,
                                    "last_move_time": effect["sim_time"],
                                }
                            )

            elif effect_kind == "Speak" or effect_kind == "SpeakResponse":
                # Update entity with speaking information
                if source_id not in self.world_state.entities:
                    self.world_state.entities[source_id] = {}

                entity_data = self.world_state.entities[source_id]
                if isinstance(entity_data, dict):
                    entity_data.update(
                        {
                            "last_message": payload.get("text", ""),
                            "last_speak_time": effect["sim_time"],
                            "message_count": entity_data.get("message_count", 0) + 1,
                        }
                    )

            elif effect_kind == "Interact":
                # Handle interaction between entities
                target_id = payload.get("target_id")
                if target_id:
                    # Update relationships
                    if source_id not in self.world_state.relationships:
                        self.world_state.relationships[source_id] = []
                    if target_id not in self.world_state.relationships[source_id]:
                        self.world_state.relationships[source_id].append(target_id)

                    # Bidirectional relationship
                    if target_id not in self.world_state.relationships:
                        self.world_state.relationships[target_id] = []
                    if source_id not in self.world_state.relationships[target_id]:
                        self.world_state.relationships[target_id].append(source_id)

            elif effect_kind == "EntityCreated":
                # Create new entity
                entity_id = payload.get("entity_id", source_id)
                self.world_state.entities[entity_id] = payload.get("entity_data", {})

                # Set initial position if provided
                if "position" in payload:
                    position = payload["position"]
                    if isinstance(position, list | tuple) and len(position) >= 3:
                        self.world_state.spatial_index[entity_id] = (
                            float(position[0]),
                            float(position[1]),
                            float(position[2]),
                        )

            elif effect_kind == "EntityRemoved":
                # Remove entity
                entity_id = payload.get("entity_id", source_id)
                self.world_state.entities.pop(entity_id, None)
                self.world_state.spatial_index.pop(entity_id, None)
                self.world_state.relationships.pop(entity_id, None)

                # Remove from other entities' relationships
                for relations in self.world_state.relationships.values():
                    if entity_id in relations:
                        relations.remove(entity_id)

            elif effect_kind == "MessageEmitted":
                # Handle message emission (similar to Speak but more general)
                if source_id not in self.world_state.entities:
                    self.world_state.entities[source_id] = {}

                entity_data = self.world_state.entities[source_id]
                if isinstance(entity_data, dict):
                    entity_data.update(
                        {
                            "last_message": payload.get(
                                "text", payload.get("message", "")
                            ),
                            "last_emit_time": effect["sim_time"],
                            "emit_count": entity_data.get("emit_count", 0) + 1,
                        }
                    )

            # Update world metadata
            self.world_state.metadata.update(
                {
                    "last_effect_seq": effect["global_seq"],
                    "last_effect_time": effect["sim_time"],
                    "last_effect_kind": effect_kind,
                }
            )

            self._logger.debug(
                "Effect applied to world state",
                effect_kind=effect_kind,
                source_id=source_id,
                global_seq=effect["global_seq"],
            )

        except Exception as e:
            self._logger.error(
                "Failed to apply effect to world state",
                effect_kind=effect_kind,
                source_id=source_id,
                global_seq=effect["global_seq"],
                error=str(e),
                error_type=type(e).__name__,
            )
            # Don't re-raise - world state update failures shouldn't break observation distribution

    async def _distribute_observations(self, effect: Effect) -> None:
        """Generate and distribute observation deltas to affected agents.

        Also checks for stale cancel tokens and automatically cancels them
        when new events occur during generation.

        Args:
            effect: Effect that occurred and needs to be observed

        Requirements addressed:
        - 4.2: Evaluate staleness when new events occur during generation
        - 4.3: Cancel current generation when context becomes stale
        """
        if not self.agent_handles:
            return  # No agents to notify

        distribution_start = time.perf_counter()
        agents_notified = 0

        # First, check for stale tokens and cancel them automatically
        _ = await self.check_and_cancel_stale_tokens(effect)

        # Parallel observation generation for better performance
        async def process_agent_observation(
            agent_id: str, agent_handle: AgentHandle
        ) -> tuple[str, bool]:
            try:
                # Check if agent should observe this effect
                observation_policy = self.observation_policies[agent_id]
                should_observe = observation_policy.should_observe_event(
                    effect, agent_id, self.world_state
                )

                if not should_observe:
                    return agent_id, False

                # Generate new view for the agent
                new_view = observation_policy.filter_world_state(
                    self.world_state, agent_id
                )
                new_view.view_seq = effect["global_seq"]

                # Get agent's current view to generate delta
                current_view_seq = agent_handle.view_seq

                # For the first observation or if we don't have previous view, create a full view
                if current_view_seq == 0:
                    # First observation - create delta with full state
                    observation_delta = ObservationDelta(
                        view_seq=new_view.view_seq,
                        patches=[
                            {
                                "op": "replace",
                                "path": "/visible_entities",
                                "value": new_view.visible_entities,
                            },
                            {
                                "op": "replace",
                                "path": "/visible_relationships",
                                "value": new_view.visible_relationships,
                            },
                        ],
                        context_digest=new_view.context_digest,
                        schema_version="1.0.0",
                    )
                else:
                    # Generate incremental delta
                    # For now, create a simplified delta based on the effect
                    observation_delta = self._create_effect_based_delta(
                        effect, new_view, agent_id, observation_policy
                    )

                # Calculate delivery delay using latency model
                delivery_delay = observation_policy.latency_model.calculate_delay(
                    effect["source_id"], agent_id, effect
                )

                # Schedule delivery via timed queue
                agent_queue = self._per_agent_queues[agent_id]
                loop = asyncio.get_running_loop()
                deliver_at = loop.time() + delivery_delay

                await agent_queue.put_at(deliver_at, observation_delta)

                # Record observation delivery latency and bandwidth
                record_observation_delivery_latency(agent_id, delivery_delay)

                # Calculate patch size for bandwidth monitoring
                import orjson

                patch_size = len(orjson.dumps(observation_delta))
                patch_count = len(observation_delta.get("patches", []))
                bandwidth_monitor.record_patch_bandwidth(
                    agent_id, patch_size, patch_count, False
                )

                self._logger.debug(
                    "Observation scheduled for delivery",
                    agent_id=agent_id,
                    effect_kind=effect["kind"],
                    global_seq=effect["global_seq"],
                    delivery_delay_ms=delivery_delay * 1000,
                    view_seq=new_view.view_seq,
                )

                return agent_id, True

            except Exception as e:
                self._logger.error(
                    "Failed to generate observation for agent",
                    agent_id=agent_id,
                    effect_kind=effect["kind"],
                    global_seq=effect["global_seq"],
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return agent_id, False

        # Execute all agent observation processing in parallel
        tasks = [
            asyncio.create_task(process_agent_observation(agent_id, agent_handle))
            for agent_id, agent_handle in self.agent_handles.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful notifications
        agents_notified = sum(
            1 for result in results if isinstance(result, tuple) and result[1]
        )

        distribution_time_ms = (time.perf_counter() - distribution_start) * 1000

        self._logger.info(
            "Observation distribution complete",
            effect_kind=effect["kind"],
            global_seq=effect["global_seq"],
            agents_notified=agents_notified,
            total_agents=len(self.agent_handles),
            distribution_time_ms=distribution_time_ms,
        )

    def _create_effect_based_delta(
        self,
        effect: Effect,
        new_view: Any,
        agent_id: str,
        observation_policy: ObservationPolicy,
    ) -> ObservationDelta:
        """Create observation delta based on effect type and content.

        Args:
            effect: Effect that occurred
            new_view: New view for the agent
            agent_id: Agent receiving the observation
            observation_policy: Agent's observation policy

        Returns:
            ObservationDelta dictionary with patches
        """
        effect_kind = effect["kind"]
        payload = effect["payload"]
        source_id = effect["source_id"]

        patches = []

        try:
            if effect_kind == "Move":
                # Update position in visible entities
                if source_id in new_view.visible_entities:
                    patches.append(
                        {
                            "op": "replace",
                            "path": f"/visible_entities/{source_id}/last_position",
                            "value": payload.get("position"),
                        }
                    )
                    patches.append(
                        {
                            "op": "replace",
                            "path": f"/visible_entities/{source_id}/last_move_time",
                            "value": effect["sim_time"],
                        }
                    )

            elif effect_kind in ["Speak", "SpeakResponse", "MessageEmitted"]:
                # Update message in visible entities
                if source_id in new_view.visible_entities:
                    message_field = "last_message"
                    message_value = payload.get("text", payload.get("message", ""))

                    patches.append(
                        {
                            "op": "replace",
                            "path": f"/visible_entities/{source_id}/{message_field}",
                            "value": message_value,
                        }
                    )
                    patches.append(
                        {
                            "op": "replace",
                            "path": f"/visible_entities/{source_id}/last_speak_time",
                            "value": effect["sim_time"],
                        }
                    )

            elif effect_kind == "Interact":
                # Update relationships if both entities are visible
                target_id = payload.get("target_id")
                if (
                    target_id
                    and source_id in new_view.visible_entities
                    and target_id in new_view.visible_entities
                ):
                    # Add relationship if not already present
                    if source_id in new_view.visible_relationships:
                        if target_id not in new_view.visible_relationships[source_id]:
                            patches.append(
                                {
                                    "op": "add",
                                    "path": f"/visible_relationships/{source_id}/-",
                                    "value": target_id,
                                }
                            )
                    else:
                        patches.append(
                            {
                                "op": "add",
                                "path": f"/visible_relationships/{source_id}",
                                "value": [target_id],
                            }
                        )

            elif effect_kind == "EntityCreated":
                # Add new entity if it should be visible
                entity_id = payload.get("entity_id", source_id)
                if entity_id in new_view.visible_entities:
                    patches.append(
                        {
                            "op": "add",
                            "path": f"/visible_entities/{entity_id}",
                            "value": new_view.visible_entities[entity_id],
                        }
                    )

            elif effect_kind == "EntityRemoved":
                # Remove entity if it was visible
                entity_id = payload.get("entity_id", source_id)
                patches.append(
                    {
                        "op": "remove",
                        "path": f"/visible_entities/{entity_id}",
                    }
                )
                # Also remove from relationships
                if entity_id in new_view.visible_relationships:
                    patches.append(
                        {
                            "op": "remove",
                            "path": f"/visible_relationships/{entity_id}",
                        }
                    )

            # Check if patches exceed max_patch_ops threshold
            if len(patches) > observation_policy.config.max_patch_ops:
                # Fallback to full snapshot
                patches = [
                    {
                        "op": "replace",
                        "path": "/visible_entities",
                        "value": new_view.visible_entities,
                    },
                    {
                        "op": "replace",
                        "path": "/visible_relationships",
                        "value": new_view.visible_relationships,
                    },
                ]

        except Exception as e:
            self._logger.warning(
                "Failed to create effect-based delta, falling back to full snapshot",
                effect_kind=effect_kind,
                agent_id=agent_id,
                error=str(e),
            )
            # Fallback to full snapshot
            patches = [
                {
                    "op": "replace",
                    "path": "/visible_entities",
                    "value": new_view.visible_entities,
                },
                {
                    "op": "replace",
                    "path": "/visible_relationships",
                    "value": new_view.visible_relationships,
                },
            ]

        return ObservationDelta(
            view_seq=new_view.view_seq,
            patches=patches,
            context_digest=new_view.context_digest,
            schema_version="1.0.0",
        )

    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory management statistics.

        Returns:
            Dictionary with detailed memory statistics including log entries,
            snapshots, view cache, and compaction information.
        """
        return self.memory_manager.get_detailed_stats(self.event_log)

    async def force_compaction(self) -> int:
        """Force log compaction regardless of thresholds.

        Returns:
            Number of entries removed during compaction
        """
        return await self.memory_manager.compact_log(self.event_log)

    async def create_snapshot(self) -> Any:
        """Force creation of a WorldState snapshot.

        Returns:
            Created snapshot
        """
        return await self.memory_manager.snapshot_manager.create_snapshot(
            self._global_seq, self.world_state, self._current_sim_time()
        )

    def evict_old_views(self, max_age_seconds: float = 3600.0) -> int:
        """Evict old cached views to free memory.

        Args:
            max_age_seconds: Maximum age for cached views

        Returns:
            Number of views evicted
        """
        return self.memory_manager.evict_old_views(max_age_seconds)

    async def shutdown(self) -> None:
        """Shutdown orchestrator and clean up resources."""
        self._logger.info("Starting orchestrator shutdown")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            finally:
                self._processing_task = None

        # Stop system monitoring task
        if hasattr(self, "_system_monitoring_task") and self._system_monitoring_task:
            self._system_monitoring_task.cancel()
            try:
                await self._system_monitoring_task
            except asyncio.CancelledError:
                pass
            finally:
                self._system_monitoring_task = None

        # Close deduplication store
        await self._dedup_store.close()

        # Close all agent queues
        for queue in self._per_agent_queues.values():
            await queue.close()

        # Cleanup memory manager
        await self.memory_manager.cleanup()

        # Clear state
        self.agent_handles.clear()
        self.observation_policies.clear()
        self._per_agent_queues.clear()
        self._cancel_tokens.clear()
        self._quota_tracker.clear()

        self._initialized = False
        self._initialize_lock = None
        self._logger.info("Orchestrator shutdown complete")

    def _start_system_monitoring(self) -> None:
        """Start background system monitoring task."""

        async def monitor_system() -> None:
            while not self._shutdown_event.is_set():
                try:
                    # Record system metrics
                    system_monitor.record_memory_usage("orchestrator")
                    system_monitor.record_cpu_usage("orchestrator")

                    # Record queue depths
                    for agent_id, queue in self._per_agent_queues.items():
                        queue_depth = queue.qsize() if hasattr(queue, "qsize") else 0
                        record_queue_depth(agent_id, queue_depth)

                    # Wait before next monitoring cycle
                    await asyncio.sleep(30.0)  # Monitor every 30 seconds

                except Exception as e:
                    self._logger.warning("System monitoring error", error=str(e))
                    await asyncio.sleep(60.0)  # Back off on error

        # Store monitor_system function for later use in initialize()
        self._monitor_system_func = monitor_system
