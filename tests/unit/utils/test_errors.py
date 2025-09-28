"""Unit tests for error handling and recovery system."""

import asyncio
from unittest.mock import patch

import pytest

from gunn.schemas.types import Effect, Intent
from gunn.utils.errors import (
    BackpressureError,
    CircuitBreaker,
    CircuitBreakerOpenError,
    ErrorRecoveryPolicy,
    IntentConflictError,
    QuotaExceededError,
    RecoveryAction,
    SimulationError,
    StaleContextError,
    TimeoutError,
    ValidationError,
)


class TestSimulationError:
    """Test base SimulationError class."""

    def test_initialization(self) -> None:
        """Test error initialization with message and recovery action."""
        error = SimulationError("Test error", RecoveryAction.RETRY)
        assert str(error) == "Test error"
        assert error.recovery_action == RecoveryAction.RETRY

    def test_default_recovery_action(self) -> None:
        """Test default recovery action is ABORT."""
        error = SimulationError("Test error")
        assert error.recovery_action == RecoveryAction.ABORT


class TestStaleContextError:
    """Test StaleContextError class."""

    def test_initialization(self) -> None:
        """Test stale context error initialization."""
        error = StaleContextError("req_123", 10, 15, 2)

        assert error.req_id == "req_123"
        assert error.expected_seq == 10
        assert error.actual_seq == 15
        assert error.staleness == 5
        assert error.threshold == 2
        assert error.recovery_action == RecoveryAction.REGENERATE

    def test_message_format(self) -> None:
        """Test error message format."""
        error = StaleContextError("req_123", 10, 15, 2)
        expected = (
            "Intent req_123 has stale context: expected_seq=10, "
            "actual_seq=15, staleness=5, threshold=2"
        )
        assert str(error) == expected

    def test_default_threshold(self) -> None:
        """Test default threshold is 0."""
        error = StaleContextError("req_123", 10, 15)
        assert error.threshold == 0


class TestIntentConflictError:
    """Test IntentConflictError class."""

    def test_initialization(self) -> None:
        """Test intent conflict error initialization."""
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 10,
            "req_id": "req_123",
            "agent_id": "agent_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        effects: list[Effect] = [
            {
                "uuid": "effect_1",
                "kind": "Speak",
                "payload": {"text": "Hi"},
                "global_seq": 11,
                "sim_time": 1.0,
                "source_id": "agent_2",
                "schema_version": "1.0.0",
            }
        ]

        error = IntentConflictError(intent, effects)

        assert error.intent == intent
        assert error.conflicting_effects == effects
        assert error.recovery_action == RecoveryAction.RETRY_WITH_DELAY

    def test_message_format(self) -> None:
        """Test error message includes conflict UUIDs."""
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 10,
            "req_id": "req_123",
            "agent_id": "agent_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        effects: list[Effect] = [
            {
                "uuid": "effect_1",
                "kind": "Speak",
                "payload": {"text": "Hi"},
                "global_seq": 11,
                "sim_time": 1.0,
                "source_id": "agent_2",
                "schema_version": "1.0.0",
            },
            {
                "uuid": "effect_2",
                "kind": "Move",
                "payload": {"x": 1, "y": 2},
                "global_seq": 12,
                "sim_time": 1.1,
                "source_id": "agent_3",
                "schema_version": "1.0.0",
            },
        ]

        error = IntentConflictError(intent, effects)
        assert "req_123" in str(error)
        assert "effect_1" in str(error)
        assert "effect_2" in str(error)


class TestQuotaExceededError:
    """Test QuotaExceededError class."""

    def test_initialization(self) -> None:
        """Test quota exceeded error initialization."""
        error = QuotaExceededError("agent_1", "intents_per_minute", 100, 105)

        assert error.agent_id == "agent_1"
        assert error.quota_type == "intents_per_minute"
        assert error.limit == 100
        assert error.current == 105
        assert error.recovery_action == RecoveryAction.DEFER

    def test_message_format(self) -> None:
        """Test error message format."""
        error = QuotaExceededError("agent_1", "intents_per_minute", 100, 105)
        expected = (
            "Agent agent_1 exceeded intents_per_minute quota: limit=100, current=105"
        )
        assert str(error) == expected

    def test_default_current(self) -> None:
        """Test default current value is 0."""
        error = QuotaExceededError("agent_1", "tokens", 1000)
        assert error.current == 0


class TestTimeoutError:
    """Test TimeoutError class."""

    def test_initialization(self) -> None:
        """Test timeout error initialization."""
        error = TimeoutError("intent_processing", 5000.0, 7500.0)

        assert error.operation == "intent_processing"
        assert error.deadline_ms == 5000.0
        assert error.actual_ms == 7500.0
        assert error.recovery_action == RecoveryAction.ABORT

    def test_message_format(self) -> None:
        """Test error message format."""
        error = TimeoutError("intent_processing", 5000.0, 7500.0)
        expected = (
            "Operation 'intent_processing' timed out: "
            "deadline=5000.0ms, actual=7500.0ms"
        )
        assert str(error) == expected


class TestBackpressureError:
    """Test BackpressureError class."""

    def test_initialization(self) -> None:
        """Test backpressure error initialization."""
        error = BackpressureError("agent_1", "intent_queue", 150, 100, "defer")

        assert error.agent_id == "agent_1"
        assert error.queue_type == "intent_queue"
        assert error.current_depth == 150
        assert error.threshold == 100
        assert error.policy == "defer"
        assert error.recovery_action == RecoveryAction.DEFER

    def test_policy_to_recovery_action_mapping(self) -> None:
        """Test policy names map to correct recovery actions."""
        defer_error = BackpressureError("agent_1", "queue", 150, 100, "defer")
        assert defer_error.recovery_action == RecoveryAction.DEFER

        shed_error = BackpressureError("agent_1", "queue", 150, 100, "shed_oldest")
        assert shed_error.recovery_action == RecoveryAction.SHED_OLDEST

        drop_error = BackpressureError("agent_1", "queue", 150, 100, "drop_newest")
        assert drop_error.recovery_action == RecoveryAction.DROP_NEWEST

    def test_unknown_policy_defaults_to_defer(self) -> None:
        """Test unknown policy defaults to DEFER recovery action."""
        error = BackpressureError("agent_1", "queue", 150, 100, "unknown_policy")
        assert error.recovery_action == RecoveryAction.DEFER

    def test_message_format(self) -> None:
        """Test error message format."""
        error = BackpressureError("agent_1", "intent_queue", 150, 100, "defer")
        expected = (
            "Backpressure triggered for agent agent_1 on intent_queue: "
            "depth=150, threshold=100, policy=defer"
        )
        assert str(error) == expected


class TestValidationError:
    """Test ValidationError class."""

    def test_initialization(self) -> None:
        """Test validation error initialization."""
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 10,
            "req_id": "req_123",
            "agent_id": "agent_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        failures = ["Invalid payload format", "Missing required field"]
        error = ValidationError(intent, failures)

        assert error.intent == intent
        assert error.validation_failures == failures
        assert error.recovery_action == RecoveryAction.ABORT

    def test_message_format(self) -> None:
        """Test error message includes all failures."""
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 10,
            "req_id": "req_123",
            "agent_id": "agent_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        failures = ["Invalid payload format", "Missing required field"]
        error = ValidationError(intent, failures)

        message = str(error)
        assert "req_123" in message
        assert "Invalid payload format" in message
        assert "Missing required field" in message


class TestCircuitBreakerOpenError:
    """Test CircuitBreakerOpenError class."""

    def test_initialization(self):
        """Test circuit breaker open error initialization."""
        error = CircuitBreakerOpenError("llm_adapter", 5, 3)

        assert error.component == "llm_adapter"
        assert error.failure_count == 5
        assert error.threshold == 3
        assert error.recovery_action == RecoveryAction.RETRY_WITH_DELAY

    def test_message_format(self):
        """Test error message format."""
        error = CircuitBreakerOpenError("llm_adapter", 5, 3)
        expected = "Circuit breaker open for llm_adapter: failures=5, threshold=3"
        assert str(error) == expected


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_initialization(self):
        """Test circuit breaker initialization with defaults."""
        cb = CircuitBreaker()

        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30.0
        assert cb.half_open_max_calls == 3
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open

    def test_custom_initialization(self):
        """Test circuit breaker initialization with custom values."""
        cb = CircuitBreaker(
            failure_threshold=3, recovery_timeout=60.0, half_open_max_calls=5
        )

        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 60.0
        assert cb.half_open_max_calls == 5

    def test_successful_call(self):
        """Test successful function call through circuit breaker."""
        cb = CircuitBreaker()

        def test_func(x, y):
            return x + y

        result = cb.call(test_func, 2, 3)
        assert result == 5
        assert cb.is_closed
        assert cb.failure_count == 0

    def test_failed_call(self):
        """Test failed function call increments failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        def failing_func():
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.failure_count == 1
        assert cb.is_closed

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold is reached."""
        cb = CircuitBreaker(failure_threshold=2)

        def failing_func():
            raise ValueError("Test error")

        # Reach failure threshold
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.is_open
        assert cb.failure_count == 2

    def test_open_circuit_blocks_calls(self):
        """Test open circuit blocks all calls."""
        cb = CircuitBreaker(failure_threshold=1)

        def failing_func():
            raise ValueError("Test error")

        def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.is_open

        # Now even successful functions should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(success_func)

    @patch("time.time")
    def test_circuit_transitions_to_half_open(self, mock_time):
        """Test circuit transitions to half-open after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0)

        def failing_func():
            raise ValueError("Test error")

        def success_func():
            return "success"

        # Set initial time
        mock_time.return_value = 1000.0

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.is_open

        # Time passes but not enough for recovery
        mock_time.return_value = 1020.0
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(success_func)

        # Enough time passes for recovery
        mock_time.return_value = 1035.0
        result = cb.call(success_func)

        assert result == "success"
        assert cb.is_closed  # Should transition to closed after successful call

    @patch("time.time")
    def test_half_open_limits_calls(self, mock_time):
        """Test half-open state limits number of calls."""
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=30.0, half_open_max_calls=2
        )

        def failing_func():
            raise ValueError("Test error")

        def success_func():
            return "success"

        # Set initial time and open circuit
        mock_time.return_value = 1000.0
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Transition to half-open
        mock_time.return_value = 1035.0

        # First call should work and transition to half-open
        result = cb.call(success_func)
        assert result == "success"
        assert cb.is_closed  # Should close after successful call

    @pytest.mark.asyncio
    async def test_async_call_success(self):
        """Test successful async function call."""
        cb = CircuitBreaker()

        async def async_func(x, y):
            await asyncio.sleep(0.01)
            return x * y

        result = await cb.async_call(async_func, 3, 4)
        assert result == 12
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_async_call_failure(self):
        """Test failed async function call."""
        cb = CircuitBreaker(failure_threshold=2)

        async def failing_async_func():
            await asyncio.sleep(0.01)
            raise ValueError("Async error")

        # First failure
        with pytest.raises(ValueError):
            await cb.async_call(failing_async_func)

        assert cb.failure_count == 1
        assert cb.is_closed

        # Second failure should open circuit
        with pytest.raises(ValueError):
            await cb.async_call(failing_async_func)

        assert cb.is_open

    @pytest.mark.asyncio
    async def test_async_open_circuit_blocks_calls(self):
        """Test open circuit blocks async calls."""
        cb = CircuitBreaker(failure_threshold=1)

        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.async_call(failing_func)

        assert cb.is_open

        # Should block even successful functions
        with pytest.raises(CircuitBreakerOpenError):
            await cb.async_call(success_func)


class TestErrorRecoveryPolicy:
    """Test ErrorRecoveryPolicy class."""

    def test_initialization(self):
        """Test policy initialization with defaults."""
        policy = ErrorRecoveryPolicy()

        assert policy.max_retries == 3
        assert policy.retry_delay_ms == 100.0
        assert policy.backoff_multiplier == 2.0
        assert policy.max_delay_ms == 5000.0

    def test_custom_initialization(self):
        """Test policy initialization with custom values."""
        policy = ErrorRecoveryPolicy(
            max_retries=5,
            retry_delay_ms=200.0,
            backoff_multiplier=1.5,
            max_delay_ms=10000.0,
        )

        assert policy.max_retries == 5
        assert policy.retry_delay_ms == 200.0
        assert policy.backoff_multiplier == 1.5
        assert policy.max_delay_ms == 10000.0

    def test_handle_stale_context(self):
        """Test stale context error handling."""
        policy = ErrorRecoveryPolicy()
        error = StaleContextError("req_123", 10, 15)

        action = policy.handle_stale_context(error)
        assert action == RecoveryAction.REGENERATE

    def test_handle_intent_conflict(self):
        """Test intent conflict error handling."""
        policy = ErrorRecoveryPolicy()
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 10,
            "req_id": "req_123",
            "agent_id": "agent_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }
        error = IntentConflictError(intent, [])

        action = policy.handle_intent_conflict(error)
        assert action == RecoveryAction.RETRY_WITH_DELAY

    def test_handle_quota_exceeded(self):
        """Test quota exceeded error handling."""
        policy = ErrorRecoveryPolicy()
        error = QuotaExceededError("agent_1", "intents", 100)

        action = policy.handle_quota_exceeded(error)
        assert action == RecoveryAction.DEFER

    def test_handle_timeout(self):
        """Test timeout error handling."""
        policy = ErrorRecoveryPolicy()
        error = TimeoutError("operation", 5000.0, 7500.0)

        action = policy.handle_timeout(error)
        assert action == RecoveryAction.ABORT

    def test_handle_backpressure(self):
        """Test backpressure error handling uses error's recovery action."""
        policy = ErrorRecoveryPolicy()

        defer_error = BackpressureError("agent_1", "queue", 150, 100, "defer")
        assert policy.handle_backpressure(defer_error) == RecoveryAction.DEFER

        shed_error = BackpressureError("agent_1", "queue", 150, 100, "shed_oldest")
        assert policy.handle_backpressure(shed_error) == RecoveryAction.SHED_OLDEST

    def test_handle_validation(self):
        """Test validation error handling."""
        policy = ErrorRecoveryPolicy()
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 10,
            "req_id": "req_123",
            "agent_id": "agent_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }
        error = ValidationError(intent, ["Invalid format"])

        action = policy.handle_validation(error)
        assert action == RecoveryAction.ABORT

    def test_handle_circuit_breaker(self):
        """Test circuit breaker error handling."""
        policy = ErrorRecoveryPolicy()
        error = CircuitBreakerOpenError("component", 5, 3)

        action = policy.handle_circuit_breaker(error)
        assert action == RecoveryAction.RETRY_WITH_DELAY

    def test_calculate_retry_delay(self):
        """Test retry delay calculation with exponential backoff."""
        policy = ErrorRecoveryPolicy(
            retry_delay_ms=100.0, backoff_multiplier=2.0, max_delay_ms=1000.0
        )

        assert policy.calculate_retry_delay(0) == 100.0  # 100 * 2^0
        assert policy.calculate_retry_delay(1) == 200.0  # 100 * 2^1
        assert policy.calculate_retry_delay(2) == 400.0  # 100 * 2^2
        assert policy.calculate_retry_delay(3) == 800.0  # 100 * 2^3
        assert policy.calculate_retry_delay(4) == 1000.0  # Capped at max_delay_ms

    def test_should_retry_with_retryable_actions(self):
        """Test should_retry returns True for retryable actions."""
        policy = ErrorRecoveryPolicy(max_retries=3)

        retry_error = SimulationError("test", RecoveryAction.RETRY)
        assert policy.should_retry(retry_error, 0)
        assert policy.should_retry(retry_error, 2)
        assert not policy.should_retry(retry_error, 3)  # Exceeds max_retries

        retry_delay_error = SimulationError("test", RecoveryAction.RETRY_WITH_DELAY)
        assert policy.should_retry(retry_delay_error, 1)

        regenerate_error = SimulationError("test", RecoveryAction.REGENERATE)
        assert policy.should_retry(regenerate_error, 0)

    def test_should_retry_with_non_retryable_actions(self):
        """Test should_retry returns False for non-retryable actions."""
        policy = ErrorRecoveryPolicy(max_retries=3)

        abort_error = SimulationError("test", RecoveryAction.ABORT)
        assert not policy.should_retry(abort_error, 0)

        defer_error = SimulationError("test", RecoveryAction.DEFER)
        assert not policy.should_retry(defer_error, 0)

        shed_error = SimulationError("test", RecoveryAction.SHED_OLDEST)
        assert not policy.should_retry(shed_error, 0)

    def test_should_retry_exceeds_max_attempts(self):
        """Test should_retry returns False when max attempts exceeded."""
        policy = ErrorRecoveryPolicy(max_retries=2)

        retry_error = SimulationError("test", RecoveryAction.RETRY)
        assert policy.should_retry(retry_error, 0)
        assert policy.should_retry(retry_error, 1)
        assert not policy.should_retry(retry_error, 2)  # Equals max_retries
        assert not policy.should_retry(retry_error, 3)  # Exceeds max_retries
