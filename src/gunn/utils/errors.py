"""Structured error types for simulation core.

This module provides structured exceptions with recovery actions
for handling various error scenarios in the multi-agent simulation.
"""

from enum import Enum
from typing import Any

from gunn.schemas.types import Effect, Intent


class RecoveryAction(Enum):
    """Recovery actions for error handling."""

    RETRY = "retry"
    ABORT = "abort"
    REGENERATE = "regenerate"
    RETRY_WITH_DELAY = "retry_with_delay"
    MODIFY_INTENT = "modify_intent"
    DEFER = "defer"
    SHED_OLDEST = "shed_oldest"


class SimulationError(Exception):
    """Base exception for simulation errors."""

    def __init__(
        self, message: str, recovery_action: RecoveryAction = RecoveryAction.ABORT
    ):
        """Initialize simulation error.

        Args:
            message: Error message
            recovery_action: Suggested recovery action
        """
        super().__init__(message)
        self.recovery_action = recovery_action


class StaleContextError(SimulationError):
    """Error raised when intent context becomes stale.

    This occurs when an agent's view_seq is outdated compared to
    the latest global sequence, indicating the agent is working
    with stale information.
    """

    def __init__(self, req_id: str, expected_seq: int, actual_seq: int):
        """Initialize stale context error.

        Args:
            req_id: Request identifier
            expected_seq: Expected sequence number
            actual_seq: Actual sequence number
        """
        self.req_id = req_id
        self.expected_seq = expected_seq
        self.actual_seq = actual_seq

        staleness = actual_seq - expected_seq
        message = (
            f"Intent {req_id} has stale context: expected seq {expected_seq}, "
            f"actual seq {actual_seq} (staleness: {staleness})"
        )

        super().__init__(message, RecoveryAction.REGENERATE)


class IntentConflictError(SimulationError):
    """Error raised when intent conflicts with existing effects.

    This occurs when an intent cannot be executed due to conflicts
    with effects that have already been applied to the world state.
    """

    def __init__(self, intent: Intent, conflicting_effects: list[Effect]):
        """Initialize intent conflict error.

        Args:
            intent: The conflicting intent
            conflicting_effects: List of conflicting effects
        """
        self.intent = intent
        self.conflicting_effects = conflicting_effects

        conflict_ids = [effect["uuid"] for effect in conflicting_effects]
        message = f"Intent {intent['req_id']} conflicts with effects: {conflict_ids}"

        super().__init__(message, RecoveryAction.RETRY_WITH_DELAY)


class QuotaExceededError(SimulationError):
    """Error raised when agent exceeds quota limits.

    This occurs when an agent attempts to perform an action that
    would exceed their configured quota limits.
    """

    def __init__(self, agent_id: str, quota_type: str, limit: int, current: int = 0):
        """Initialize quota exceeded error.

        Args:
            agent_id: Agent identifier
            quota_type: Type of quota exceeded
            limit: Quota limit
            current: Current usage (optional)
        """
        self.agent_id = agent_id
        self.quota_type = quota_type
        self.limit = limit
        self.current = current

        if current > 0:
            message = (
                f"Agent {agent_id} exceeded {quota_type} quota: " f"{current}/{limit}"
            )
        else:
            message = f"Agent {agent_id} exceeded {quota_type} quota limit: {limit}"

        super().__init__(message, RecoveryAction.DEFER)


class CircuitBreakerOpenError(SimulationError):
    """Error raised when circuit breaker is open.

    This occurs when a circuit breaker is in the open state,
    preventing operations from being executed.
    """

    def __init__(self, component: str, failure_count: int, threshold: int):
        """Initialize circuit breaker error.

        Args:
            component: Component name
            failure_count: Current failure count
            threshold: Failure threshold
        """
        self.component = component
        self.failure_count = failure_count
        self.threshold = threshold

        message = (
            f"Circuit breaker open for {component}: "
            f"{failure_count} failures (threshold: {threshold})"
        )

        super().__init__(message, RecoveryAction.RETRY_WITH_DELAY)


class ValidationError(SimulationError):
    """Error raised when intent validation fails.

    This occurs when an intent fails validation checks
    before being converted to an effect.
    """

    def __init__(self, intent: Intent, validation_errors: list[str]):
        """Initialize validation error.

        Args:
            intent: The invalid intent
            validation_errors: List of validation error messages
        """
        self.intent = intent
        self.validation_errors = validation_errors

        errors_str = "; ".join(validation_errors)
        message = f"Intent {intent['req_id']} validation failed: {errors_str}"

        super().__init__(message, RecoveryAction.MODIFY_INTENT)


class BackpressureError(SimulationError):
    """Error raised when backpressure limits are exceeded.

    This occurs when queue depths or processing rates exceed
    configured thresholds, triggering backpressure policies.
    """

    def __init__(
        self,
        agent_id: str,
        queue_type: str,
        current_depth: int,
        threshold: int,
        policy: str = "defer",
    ):
        """Initialize backpressure error.

        Args:
            agent_id: Agent identifier
            queue_type: Type of queue experiencing backpressure
            current_depth: Current queue depth
            threshold: Backpressure threshold
            policy: Backpressure policy being applied
        """
        self.agent_id = agent_id
        self.queue_type = queue_type
        self.current_depth = current_depth
        self.threshold = threshold
        self.policy = policy

        message = (
            f"Backpressure triggered for agent {agent_id} {queue_type}: "
            f"{current_depth}/{threshold} (policy: {policy})"
        )

        # Map policy to recovery action
        policy_to_action = {
            "defer": RecoveryAction.DEFER,
            "shed_oldest": RecoveryAction.SHED_OLDEST,
            "drop_newest": RecoveryAction.ABORT,
        }

        recovery_action = policy_to_action.get(policy, RecoveryAction.DEFER)
        super().__init__(message, recovery_action)


class TimeoutError(SimulationError):
    """Error raised when operations exceed timeout limits.

    This occurs when operations take longer than configured
    timeout thresholds.
    """

    def __init__(self, operation: str, timeout_ms: float, elapsed_ms: float):
        """Initialize timeout error.

        Args:
            operation: Operation that timed out
            timeout_ms: Timeout threshold in milliseconds
            elapsed_ms: Actual elapsed time in milliseconds
        """
        self.operation = operation
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms

        message = (
            f"Operation {operation} timed out: "
            f"{elapsed_ms:.1f}ms > {timeout_ms:.1f}ms"
        )

        super().__init__(message, RecoveryAction.ABORT)


class ErrorRecoveryPolicy:
    """Policy for handling error recovery strategies."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize error recovery policy.

        Args:
            config: Configuration for recovery strategies
        """
        self.config = config or {}

    def handle_stale_context(self, error: StaleContextError) -> RecoveryAction:
        """Handle stale context error.

        Args:
            error: Stale context error

        Returns:
            Recovery action to take
        """
        # Default: regenerate with fresh context
        return RecoveryAction.REGENERATE

    def handle_intent_conflict(self, error: IntentConflictError) -> RecoveryAction:
        """Handle intent conflict error.

        Args:
            error: Intent conflict error

        Returns:
            Recovery action to take
        """
        # Default: retry with delay to allow conflicts to resolve
        return RecoveryAction.RETRY_WITH_DELAY

    def handle_quota_exceeded(self, error: QuotaExceededError) -> RecoveryAction:
        """Handle quota exceeded error.

        Args:
            error: Quota exceeded error

        Returns:
            Recovery action to take
        """
        # Default: defer until quota resets
        return RecoveryAction.DEFER

    def handle_validation_error(self, error: ValidationError) -> RecoveryAction:
        """Handle validation error.

        Args:
            error: Validation error

        Returns:
            Recovery action to take
        """
        # Default: modify intent to fix validation issues
        return RecoveryAction.MODIFY_INTENT

    def handle_backpressure(self, error: BackpressureError) -> RecoveryAction:
        """Handle backpressure error.

        Args:
            error: Backpressure error

        Returns:
            Recovery action to take
        """
        # Use the policy specified in the error
        return error.recovery_action

    def handle_timeout(self, error: TimeoutError) -> RecoveryAction:
        """Handle timeout error.

        Args:
            error: Timeout error

        Returns:
            Recovery action to take
        """
        # Default: abort timed out operations
        return RecoveryAction.ABORT
