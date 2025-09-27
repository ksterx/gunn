"""Structured error types for multi-agent simulation.

This module defines error types for various failure scenarios in the
simulation system with recovery actions and structured information.
"""

from collections.abc import Callable
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
    DROP_NEWEST = "drop_newest"


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

    This error occurs when an agent's context (view_seq) is outdated
    compared to the current world state, indicating the intent was
    based on stale information.

    Requirements addressed:
    - 4.2: Staleness detection using latest_view_seq > context_seq + threshold
    - 10.3: Structured error codes including STALE_CONTEXT
    """

    def __init__(
        self, req_id: str, expected_seq: int, actual_seq: int, threshold: int = 0
    ):
        """Initialize stale context error.

        Args:
            req_id: Request ID of the stale intent
            expected_seq: Expected view sequence number
            actual_seq: Actual current sequence number
            threshold: Staleness threshold that was exceeded
        """
        staleness = actual_seq - expected_seq
        message = (
            f"Intent {req_id} has stale context: expected_seq={expected_seq}, "
            f"actual_seq={actual_seq}, staleness={staleness}, threshold={threshold}"
        )
        super().__init__(message, RecoveryAction.REGENERATE)

        self.req_id = req_id
        self.expected_seq = expected_seq
        self.actual_seq = actual_seq
        self.staleness = staleness
        self.threshold = threshold


class IntentConflictError(SimulationError):
    """Error raised when intent conflicts with existing effects.

    This error occurs when an intent cannot be executed due to
    conflicts with recently processed effects or other intents.

    Requirements addressed:
    - 3.4: Intent conflict resolution
    - 10.3: Structured error codes including INTENT_CONFLICT
    """

    def __init__(self, intent: Intent, conflicting_effects: list[Effect]):
        """Initialize intent conflict error.

        Args:
            intent: The conflicting intent
            conflicting_effects: List of effects that conflict with the intent
        """
        conflict_uuids = [effect["uuid"] for effect in conflicting_effects]
        message = f"Intent {intent['req_id']} conflicts with effects: {conflict_uuids}"
        super().__init__(message, RecoveryAction.RETRY_WITH_DELAY)

        self.intent = intent
        self.conflicting_effects = conflicting_effects


class QuotaExceededError(SimulationError):
    """Error raised when agent exceeds quota limits.

    This error occurs when an agent attempts to perform an action
    that would exceed their configured quota limits.

    Requirements addressed:
    - 1.5: Quota limits and permissions
    - 10.3: Structured error codes including QUOTA_EXCEEDED
    """

    def __init__(self, agent_id: str, quota_type: str, limit: int, current: int = 0):
        """Initialize quota exceeded error.

        Args:
            agent_id: Agent that exceeded quota
            quota_type: Type of quota that was exceeded
            limit: The quota limit
            current: Current usage (optional)
        """
        message = (
            f"Agent {agent_id} exceeded {quota_type} quota: "
            f"limit={limit}, current={current}"
        )
        super().__init__(message, RecoveryAction.DEFER)

        self.agent_id = agent_id
        self.quota_type = quota_type
        self.limit = limit
        self.current = current


class TimeoutError(SimulationError):
    """Error raised when operations exceed deadline.

    This error occurs when intent processing or generation
    takes longer than the configured deadline.

    Requirements addressed:
    - 3.5: Deadline enforcement for intent processing
    - 10.3: Structured error codes including TIMEOUT
    """

    def __init__(self, operation: str, deadline_ms: float, actual_ms: float):
        """Initialize timeout error.

        Args:
            operation: Name of the operation that timed out
            deadline_ms: Configured deadline in milliseconds
            actual_ms: Actual time taken in milliseconds
        """
        message = (
            f"Operation '{operation}' timed out: "
            f"deadline={deadline_ms}ms, actual={actual_ms}ms"
        )
        super().__init__(message, RecoveryAction.ABORT)

        self.operation = operation
        self.deadline_ms = deadline_ms
        self.actual_ms = actual_ms


class BackpressureError(SimulationError):
    """Error raised when backpressure limits are exceeded.

    This error occurs when queue depths or processing rates
    exceed configured thresholds, triggering backpressure policies.

    Requirements addressed:
    - 10.2: Backpressure policies (defer, shed oldest, drop newest)
    - 10.5: Queue depth monitoring and backpressure triggers
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
            agent_id: Agent experiencing backpressure
            queue_type: Type of queue that triggered backpressure
            current_depth: Current queue depth
            threshold: Threshold that was exceeded
            policy: Backpressure policy being applied
        """
        message = (
            f"Backpressure triggered for agent {agent_id} on {queue_type}: "
            f"depth={current_depth}, threshold={threshold}, policy={policy}"
        )

        # Map policy to recovery action
        policy_to_action = {
            "defer": RecoveryAction.DEFER,
            "shed_oldest": RecoveryAction.SHED_OLDEST,
            "drop_newest": RecoveryAction.DROP_NEWEST,
        }
        recovery_action = policy_to_action.get(policy, RecoveryAction.DEFER)

        super().__init__(message, recovery_action)

        self.agent_id = agent_id
        self.queue_type = queue_type
        self.current_depth = current_depth
        self.threshold = threshold
        self.policy = policy


class ValidationError(SimulationError):
    """Error raised when intent validation fails.

    This error occurs when an intent fails validation checks
    before being converted to an effect.

    Requirements addressed:
    - 1.5: Intent validation through EffectValidator
    - 3.4: Validation for quota limits, cooldowns, and permissions
    """

    def __init__(self, intent: Intent, validation_failures: list[str]):
        """Initialize validation error.

        Args:
            intent: The intent that failed validation
            validation_failures: List of validation failure reasons
        """
        failures_str = ", ".join(validation_failures)
        message = f"Intent {intent['req_id']} failed validation: {failures_str}"
        super().__init__(message, RecoveryAction.ABORT)

        self.intent = intent
        self.validation_failures = validation_failures


class CircuitBreakerOpenError(SimulationError):
    """Error raised when circuit breaker is open.

    This error occurs when a circuit breaker is in the open state,
    preventing operations from being executed to avoid cascading failures.

    Requirements addressed:
    - 10.5: Circuit breaker patterns per agent or adapter
    """

    def __init__(self, component: str, failure_count: int, threshold: int):
        """Initialize circuit breaker open error.

        Args:
            component: Component with open circuit breaker
            failure_count: Current failure count
            threshold: Failure threshold that triggered the breaker
        """
        message = (
            f"Circuit breaker open for {component}: "
            f"failures={failure_count}, threshold={threshold}"
        )
        super().__init__(message, RecoveryAction.RETRY_WITH_DELAY)

        self.component = component
        self.failure_count = failure_count
        self.threshold = threshold


class CircuitBreaker:
    """Circuit breaker for fault tolerance with failure thresholds.

    Implements the circuit breaker pattern to prevent cascading failures
    by temporarily blocking operations when failure rates exceed thresholds.

    Requirements addressed:
    - 10.5: Circuit breaker patterns per agent or adapter
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "OPEN"

    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == "HALF_OPEN"

    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == "CLOSED"

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.state != "OPEN":
            return False

        import time

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to half-open state."""
        self.state = "HALF_OPEN"
        self.half_open_calls = 0

    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to closed state."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.half_open_calls = 0

    def _transition_to_open(self) -> None:
        """Transition circuit breaker to open state."""
        self.state = "OPEN"
        import time

        self.last_failure_time = time.time()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
        """
        # Check if we should attempt reset
        if self._should_attempt_reset():
            self._transition_to_half_open()

        # Block calls if circuit is open
        if self.is_open:
            raise CircuitBreakerOpenError(
                component=func.__name__ if hasattr(func, "__name__") else "unknown",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )

        # Limit calls in half-open state
        if self.is_half_open:
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    component=func.__name__ if hasattr(func, "__name__") else "unknown",
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold,
                )
            self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)

            # Success - reset if in half-open state
            if self.is_half_open:
                self._transition_to_closed()

            return result

        except Exception:
            self.failure_count += 1

            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()

            raise

    async def async_call(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
        """
        # Check if we should attempt reset
        if self._should_attempt_reset():
            self._transition_to_half_open()

        # Block calls if circuit is open
        if self.is_open:
            raise CircuitBreakerOpenError(
                component=func.__name__ if hasattr(func, "__name__") else "unknown",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )

        # Limit calls in half-open state
        if self.is_half_open:
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    component=func.__name__ if hasattr(func, "__name__") else "unknown",
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold,
                )
            self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)

            # Success - reset if in half-open state
            if self.is_half_open:
                self._transition_to_closed()

            return result

        except Exception:
            self.failure_count += 1

            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()

            raise


class ErrorRecoveryPolicy:
    """Policy for handling different types of errors with recovery strategies.

    Provides configurable recovery strategies for different error types
    to enable graceful degradation and automatic recovery.

    Requirements addressed:
    - 10.4: Partial failures remain consistent with event log
    - 10.5: Circuit breaker patterns and error recovery
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_ms: float = 100.0,
        backoff_multiplier: float = 2.0,
        max_delay_ms: float = 5000.0,
    ):
        """Initialize error recovery policy.

        Args:
            max_retries: Maximum number of retries for recoverable errors
            retry_delay_ms: Initial retry delay in milliseconds
            backoff_multiplier: Exponential backoff multiplier
            max_delay_ms: Maximum retry delay in milliseconds
        """
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self.backoff_multiplier = backoff_multiplier
        self.max_delay_ms = max_delay_ms

    def handle_stale_context(self, error: StaleContextError) -> RecoveryAction:
        """Handle stale context error.

        Args:
            error: The stale context error

        Returns:
            Recovery action (typically REGENERATE)
        """
        # For stale context, always regenerate with fresh context
        return RecoveryAction.REGENERATE

    def handle_intent_conflict(self, error: IntentConflictError) -> RecoveryAction:
        """Handle intent conflict error.

        Args:
            error: The intent conflict error

        Returns:
            Recovery action (typically RETRY_WITH_DELAY)
        """
        # For conflicts, retry with delay to allow conflicts to resolve
        return RecoveryAction.RETRY_WITH_DELAY

    def handle_quota_exceeded(self, error: QuotaExceededError) -> RecoveryAction:
        """Handle quota exceeded error.

        Args:
            error: The quota exceeded error

        Returns:
            Recovery action (typically DEFER)
        """
        # For quota exceeded, defer until quota resets
        return RecoveryAction.DEFER

    def handle_timeout(self, error: TimeoutError) -> RecoveryAction:
        """Handle timeout error.

        Args:
            error: The timeout error

        Returns:
            Recovery action (typically ABORT)
        """
        # For timeouts, usually abort to avoid further delays
        return RecoveryAction.ABORT

    def handle_backpressure(self, error: BackpressureError) -> RecoveryAction:
        """Handle backpressure error.

        Args:
            error: The backpressure error

        Returns:
            Recovery action based on backpressure policy
        """
        # Use the recovery action from the error (based on policy)
        return error.recovery_action

    def handle_validation(self, error: ValidationError) -> RecoveryAction:
        """Handle validation error.

        Args:
            error: The validation error

        Returns:
            Recovery action (typically ABORT)
        """
        # For validation errors, usually abort since intent is invalid
        return RecoveryAction.ABORT

    def handle_circuit_breaker(self, error: CircuitBreakerOpenError) -> RecoveryAction:
        """Handle circuit breaker open error.

        Args:
            error: The circuit breaker error

        Returns:
            Recovery action (typically RETRY_WITH_DELAY)
        """
        # For circuit breaker, retry with delay to allow recovery
        return RecoveryAction.RETRY_WITH_DELAY

    def calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt: Retry attempt number (0-based)

        Returns:
            Delay in milliseconds
        """
        delay = self.retry_delay_ms * (self.backoff_multiplier**attempt)
        return min(delay, self.max_delay_ms)

    def should_retry(self, error: SimulationError, attempt: int) -> bool:
        """Determine if error should be retried.

        Args:
            error: The error that occurred
            attempt: Current attempt number (0-based)

        Returns:
            True if should retry, False otherwise
        """
        if attempt >= self.max_retries:
            return False

        # Only retry for certain recovery actions
        retryable_actions = {
            RecoveryAction.RETRY,
            RecoveryAction.RETRY_WITH_DELAY,
            RecoveryAction.REGENERATE,
        }

        return error.recovery_action in retryable_actions
