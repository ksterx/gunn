"""Integration tests for error handling and recovery system."""

import asyncio
from unittest.mock import patch

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import DefaultObservationPolicy
from gunn.schemas.types import Intent
from gunn.utils.backpressure import BackpressureManager, backpressure_manager
from gunn.utils.errors import (
    BackpressureError,
    CircuitBreaker,
    CircuitBreakerOpenError,
    ErrorRecoveryPolicy,
    IntentConflictError,
    QuotaExceededError,
    RecoveryAction,
    StaleContextError,
    ValidationError,
)


class TestErrorHandlingIntegration:
    """Integration tests for error handling and recovery system."""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for testing."""
        config = OrchestratorConfig(
            max_agents=5,
            max_queue_depth=3,
            backpressure_policy="defer",
            use_in_memory_dedup=True,
            processing_idle_shutdown_ms=100.0,
        )
        orchestrator = Orchestrator(config, world_id="test_error_handling")
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.fixture
    def observation_policy(self):
        """Create observation policy for testing."""
        from gunn.policies.observation import PolicyConfig

        config = PolicyConfig()
        return DefaultObservationPolicy(config)

    @pytest.mark.asyncio
    async def test_backpressure_error_handling(self, orchestrator, observation_policy):
        """Test backpressure error handling with different policies."""
        # Register agent
        await orchestrator.register_agent("test_agent", observation_policy)

        # Fill queue to trigger backpressure
        intents = []
        for i in range(orchestrator.config.max_queue_depth):
            intent: Intent = {
                "kind": "Speak",
                "payload": {"text": f"Message {i}"},
                "context_seq": 1,
                "req_id": f"req_{i}",
                "agent_id": "test_agent",
                "priority": 0,
                "schema_version": "1.0.0",
            }
            intents.append(intent)
            await orchestrator.submit_intent(intent)

        # Next intent should trigger backpressure
        overflow_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Overflow message"},
            "context_seq": 1,
            "req_id": "overflow_req",
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        with pytest.raises(BackpressureError) as exc_info:
            await orchestrator.submit_intent(overflow_intent)

        error = exc_info.value
        assert error.agent_id == "test_agent"
        assert error.queue_type == "agent_queue"
        assert error.current_depth >= orchestrator.config.max_queue_depth
        assert error.policy == "defer"
        assert error.recovery_action == RecoveryAction.DEFER

    @pytest.mark.asyncio
    async def test_agent_specific_backpressure_policy(
        self, orchestrator, observation_policy
    ):
        """Test agent-specific backpressure policy configuration."""
        # Register agent with default policy
        await orchestrator.register_agent("agent_defer", observation_policy)

        # Set custom backpressure policy for specific agent
        orchestrator.set_agent_backpressure_policy("agent_defer", "shed_oldest")

        # Verify policy was set
        assert orchestrator._agent_backpressure_policies["agent_defer"] == "shed_oldest"

    @pytest.mark.asyncio
    async def test_invalid_backpressure_policy_raises_error(
        self, orchestrator, observation_policy
    ):
        """Test setting invalid backpressure policy raises error."""
        await orchestrator.register_agent("test_agent", observation_policy)

        with pytest.raises(ValueError) as exc_info:
            orchestrator.set_agent_backpressure_policy("test_agent", "invalid_policy")

        assert "Invalid backpressure policy 'invalid_policy'" in str(exc_info.value)
        assert "defer, shed_oldest, drop_newest" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unregistered_agent_backpressure_policy_raises_error(
        self, orchestrator
    ):
        """Test setting backpressure policy for unregistered agent raises error."""
        with pytest.raises(ValueError) as exc_info:
            orchestrator.set_agent_backpressure_policy("nonexistent_agent", "defer")

        assert "Agent nonexistent_agent is not registered" in str(exc_info.value)

    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with error recovery."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        policy = ErrorRecoveryPolicy()

        def failing_operation():
            raise ValueError("Operation failed")

        def successful_operation():
            return "success"

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_operation)
        assert cb.failure_count == 1
        assert cb.is_closed

        # Second failure should open circuit
        with pytest.raises(ValueError):
            cb.call(failing_operation)
        assert cb.failure_count == 2
        assert cb.is_open

        # Now circuit should block calls
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            cb.call(successful_operation)

        error = exc_info.value
        assert error.component == "successful_operation"
        assert error.failure_count == 2
        assert error.threshold == 2

        # Test recovery action
        recovery_action = policy.handle_circuit_breaker(error)
        assert recovery_action == RecoveryAction.RETRY_WITH_DELAY

    @patch("time.time")
    def test_circuit_breaker_recovery_after_timeout(self, mock_time):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0)

        def failing_operation():
            raise ValueError("Failure")

        def successful_operation():
            return "success"

        # Set initial time and trigger failure
        mock_time.return_value = 1000.0
        with pytest.raises(ValueError):
            cb.call(failing_operation)
        assert cb.is_open

        # Time passes but not enough for recovery
        mock_time.return_value = 1020.0
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(successful_operation)

        # Enough time passes for recovery
        mock_time.return_value = 1035.0
        result = cb.call(successful_operation)
        assert result == "success"
        assert cb.is_closed

    def test_error_recovery_policy_retry_logic(self):
        """Test error recovery policy retry logic."""
        policy = ErrorRecoveryPolicy(
            max_retries=3, retry_delay_ms=100.0, backoff_multiplier=2.0
        )

        # Test retry delays
        assert policy.calculate_retry_delay(0) == 100.0
        assert policy.calculate_retry_delay(1) == 200.0
        assert policy.calculate_retry_delay(2) == 400.0

        # Test should_retry logic
        retry_error = StaleContextError("req_1", 10, 15)
        assert policy.should_retry(retry_error, 0)
        assert policy.should_retry(retry_error, 2)
        assert not policy.should_retry(retry_error, 3)  # Exceeds max_retries

        # Test non-retryable error
        abort_error = ValidationError(
            {
                "kind": "Speak",
                "payload": {"text": "Hello"},
                "context_seq": 10,
                "req_id": "req_123",
                "agent_id": "agent_1",
                "priority": 1,
                "schema_version": "1.0.0",
            },
            ["Invalid format"],
        )
        assert not policy.should_retry(abort_error, 0)

    def test_backpressure_manager_integration(self):
        """Test backpressure manager integration."""
        manager = BackpressureManager()

        # Test creating different policies
        defer_policy = manager.create_policy("defer", threshold=10, agent_id="agent_1")
        assert defer_policy.policy_name == "defer"
        assert defer_policy.threshold == 10

        shed_policy = manager.create_policy(
            "shed_oldest", threshold=20, agent_id="agent_2"
        )
        assert shed_policy.policy_name == "shed_oldest"
        assert shed_policy.threshold == 20

        drop_policy = manager.create_policy(
            "drop_newest", threshold=15, agent_id="agent_3"
        )
        assert drop_policy.policy_name == "drop_newest"
        assert drop_policy.threshold == 15

        # Test global manager instance
        assert backpressure_manager is not None
        global_defer = backpressure_manager.create_policy("defer", threshold=5)
        assert global_defer.policy_name == "defer"

    @patch("gunn.utils.telemetry.record_queue_high_watermark")
    @patch("gunn.utils.telemetry.record_backpressure_event")
    @patch("gunn.utils.telemetry.record_error_recovery_action")
    def test_metrics_integration(
        self, mock_recovery, mock_backpressure, mock_watermark
    ):
        """Test metrics are recorded for error scenarios."""
        from gunn.utils.telemetry import (
            record_backpressure_event,
            record_error_recovery_action,
            record_queue_high_watermark,
        )

        # Test metrics recording
        record_queue_high_watermark("agent_1", "queue", 150)
        mock_watermark.assert_called_once_with("agent_1", "queue", 150)

        record_backpressure_event("agent_1", "queue", "defer")
        mock_backpressure.assert_called_once_with("agent_1", "queue", "defer")

        record_error_recovery_action("StaleContextError", "REGENERATE", "agent_1")
        mock_recovery.assert_called_once_with(
            "StaleContextError", "REGENERATE", "agent_1"
        )

    def test_error_types_integration(self):
        """Test all error types work together with recovery policy."""
        policy = ErrorRecoveryPolicy()

        # Test StaleContextError
        stale_error = StaleContextError("req_1", 10, 15, 2)
        assert policy.handle_stale_context(stale_error) == RecoveryAction.REGENERATE
        assert stale_error.staleness == 5

        # Test IntentConflictError
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 10,
            "req_id": "req_123",
            "agent_id": "agent_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }
        conflict_error = IntentConflictError(intent, [])
        assert (
            policy.handle_intent_conflict(conflict_error)
            == RecoveryAction.RETRY_WITH_DELAY
        )

        # Test QuotaExceededError
        quota_error = QuotaExceededError("agent_1", "intents_per_minute", 100, 105)
        assert policy.handle_quota_exceeded(quota_error) == RecoveryAction.DEFER

        # Test BackpressureError with different policies
        defer_error = BackpressureError("agent_1", "queue", 150, 100, "defer")
        assert policy.handle_backpressure(defer_error) == RecoveryAction.DEFER

        shed_error = BackpressureError("agent_1", "queue", 150, 100, "shed_oldest")
        assert policy.handle_backpressure(shed_error) == RecoveryAction.SHED_OLDEST

        # Test ValidationError
        validation_error = ValidationError(intent, ["Invalid format"])
        assert policy.handle_validation(validation_error) == RecoveryAction.ABORT

        # Test CircuitBreakerOpenError
        cb_error = CircuitBreakerOpenError("component", 5, 3)
        assert (
            policy.handle_circuit_breaker(cb_error) == RecoveryAction.RETRY_WITH_DELAY
        )

    @pytest.mark.asyncio
    async def test_end_to_end_error_recovery_scenario(self):
        """Test end-to-end error recovery scenario."""
        # Create circuit breaker and recovery policy
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        policy = ErrorRecoveryPolicy(max_retries=3, retry_delay_ms=50.0)

        call_count = [0]  # Use mutable list for closure

        async def unreliable_operation(success_after: int = 3):
            """Operation that fails a few times then succeeds."""
            call_count[0] += 1

            if call_count[0] < success_after:
                raise ValueError(f"Failure {call_count[0]}")

            return f"Success after {call_count[0]} attempts"

        # Simulate retry logic with circuit breaker
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                if cb.is_open:
                    # Wait for circuit breaker recovery
                    await asyncio.sleep(0.15)

                result = await cb.async_call(unreliable_operation, success_after=3)
                assert "Success after 3 attempts" in result
                break

            except (ValueError, CircuitBreakerOpenError) as e:
                if isinstance(e, CircuitBreakerOpenError):
                    recovery_action = policy.handle_circuit_breaker(e)
                    assert recovery_action == RecoveryAction.RETRY_WITH_DELAY

                if attempt < max_attempts - 1:
                    delay = (
                        policy.calculate_retry_delay(attempt) / 1000.0
                    )  # Convert to seconds
                    await asyncio.sleep(delay)
                else:
                    pytest.fail("Operation failed after all retry attempts")

        assert cb.is_closed  # Should be closed after successful operation

    def test_error_message_formats(self):
        """Test error message formats are consistent and informative."""
        # Test StaleContextError message
        stale_error = StaleContextError("req_123", 10, 15, 2)
        expected = (
            "Intent req_123 has stale context: expected_seq=10, "
            "actual_seq=15, staleness=5, threshold=2"
        )
        assert str(stale_error) == expected

        # Test BackpressureError message
        bp_error = BackpressureError("agent_1", "intent_queue", 150, 100, "defer")
        expected = (
            "Backpressure triggered for agent agent_1 on intent_queue: "
            "depth=150, threshold=100, policy=defer"
        )
        assert str(bp_error) == expected

        # Test CircuitBreakerOpenError message
        cb_error = CircuitBreakerOpenError("llm_adapter", 5, 3)
        expected = "Circuit breaker open for llm_adapter: failures=5, threshold=3"
        assert str(cb_error) == expected

    def test_recovery_action_enum_completeness(self):
        """Test that all recovery actions are properly defined."""
        expected_actions = {
            "retry",
            "abort",
            "regenerate",
            "retry_with_delay",
            "modify_intent",
            "defer",
            "shed_oldest",
            "drop_newest",
        }

        actual_actions = {action.value for action in RecoveryAction}
        assert actual_actions == expected_actions

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self):
        """Test error handling under concurrent load."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        async def concurrent_operation(operation_id: int):
            """Concurrent operation that may fail."""
            if operation_id % 3 == 0:  # Every 3rd operation fails
                raise ValueError(f"Operation {operation_id} failed")
            return f"Operation {operation_id} succeeded"

        # Run multiple concurrent operations
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                self._safe_circuit_breaker_call(cb, concurrent_operation, i)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that we got a mix of successes and failures
        successes = [r for r in results if isinstance(r, str) and "succeeded" in r]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) > 0, "Should have some successful operations"
        assert len(failures) > 0, "Should have some failed operations"

    async def _safe_circuit_breaker_call(self, cb: CircuitBreaker, func, *args):
        """Safely call function through circuit breaker, handling open circuit."""
        try:
            return await cb.async_call(func, *args)
        except (ValueError, CircuitBreakerOpenError) as e:
            return e
