"""Additional tests for error recovery scenarios and edge cases."""

import asyncio

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import Intent
from gunn.utils.errors import (
    BackpressureError,
    CircuitBreaker,
    ErrorRecoveryPolicy,
    QuotaExceededError,
    RecoveryAction,
    StaleContextError,
)


class TestErrorRecoveryScenarios:
    """Test comprehensive error recovery scenarios."""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for testing."""
        config = OrchestratorConfig(
            max_agents=10,
            max_queue_depth=5,
            quota_intents_per_minute=10,
            quota_tokens_per_minute=1000,
            backpressure_policy="defer",
            use_in_memory_dedup=True,
            processing_idle_shutdown_ms=100.0,
        )
        orchestrator = Orchestrator(config, world_id="test_error_recovery")
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.fixture
    def observation_policy(self):
        """Create observation policy for testing."""
        config = PolicyConfig()
        return DefaultObservationPolicy(config)

    @pytest.mark.asyncio
    async def test_quota_exceeded_error_scenario(
        self, orchestrator, observation_policy
    ):
        """Test quota exceeded error handling and recovery."""
        await orchestrator.register_agent("quota_test_agent", observation_policy)

        # Submit intents rapidly to exceed quota
        intents_submitted = 0
        quota_exceeded_count = 0
        backpressure_count = 0

        for i in range(15):  # More than quota_intents_per_minute (10)
            intent = Intent(
                kind="Speak",
                payload={"text": f"Message {i}"},
                context_seq=1,
                req_id=f"quota_req_{i}",
                agent_id="quota_test_agent",
                priority=0,
                schema_version="1.0.0",
            )

            try:
                await orchestrator.submit_intent(intent)
                intents_submitted += 1
            except QuotaExceededError as e:
                quota_exceeded_count += 1
                assert e.agent_id == "quota_test_agent"
                assert e.quota_type in ["intents_per_minute", "tokens_per_minute"]
                assert e.recovery_action == RecoveryAction.DEFER
            except BackpressureError as e:
                backpressure_count += 1
                assert e.agent_id == "quota_test_agent"
                assert e.recovery_action == RecoveryAction.DEFER

        # Should have submitted some intents but hit limits (quota or backpressure)
        assert intents_submitted > 0
        assert (quota_exceeded_count + backpressure_count) > 0
        assert intents_submitted + quota_exceeded_count + backpressure_count == 15

    @pytest.mark.asyncio
    async def test_staleness_detection_and_cancellation(
        self, orchestrator, observation_policy
    ):
        """Test staleness detection and automatic cancellation."""
        await orchestrator.register_agent("staleness_test_agent", observation_policy)

        # Issue a cancel token
        cancel_token = orchestrator.issue_cancel_token(
            "staleness_test_agent", "stale_req_1"
        )
        assert not cancel_token.cancelled

        # Simulate staleness by advancing view sequence
        agent_handle = orchestrator.agent_handles["staleness_test_agent"]
        agent_handle.view_seq = 10

        # Check staleness with a higher sequence (should trigger cancellation)
        was_cancelled = await orchestrator.cancel_if_stale(
            "staleness_test_agent", "stale_req_1", 15
        )

        assert was_cancelled
        assert cancel_token.cancelled
        assert "stale_due_to_seq_gap" in cancel_token.reason

    @pytest.mark.asyncio
    async def test_debounce_logic_prevents_rapid_cancellations(
        self, orchestrator, observation_policy
    ):
        """Test debounce logic prevents rapid successive cancellations."""
        # Set short debounce time for testing
        orchestrator.config.debounce_ms = 50.0

        await orchestrator.register_agent("debounce_test_agent", observation_policy)

        # Issue first cancel token
        token1 = orchestrator.issue_cancel_token(
            "debounce_test_agent", "debounce_req_1"
        )

        # First cancellation should work
        agent_handle = orchestrator.agent_handles["debounce_test_agent"]
        agent_handle.view_seq = 5

        was_cancelled1 = await orchestrator.cancel_if_stale(
            "debounce_test_agent", "debounce_req_1", 10
        )
        assert was_cancelled1
        assert token1.cancelled

        # Issue second cancel token immediately
        token2 = orchestrator.issue_cancel_token(
            "debounce_test_agent", "debounce_req_2"
        )

        # Second cancellation should be debounced (blocked)
        was_cancelled2 = await orchestrator.cancel_if_stale(
            "debounce_test_agent", "debounce_req_2", 15
        )
        assert not was_cancelled2
        assert not token2.cancelled

        # Wait for debounce period to pass
        await asyncio.sleep(0.06)  # 60ms > 50ms debounce

        # Now cancellation should work again
        was_cancelled3 = await orchestrator.cancel_if_stale(
            "debounce_test_agent", "debounce_req_2", 20
        )
        assert was_cancelled3
        assert token2.cancelled

    @pytest.mark.asyncio
    async def test_backpressure_policy_switching(
        self, orchestrator, observation_policy
    ):
        """Test switching backpressure policies per agent."""
        await orchestrator.register_agent("policy_test_agent", observation_policy)

        # Start with default policy (defer)
        assert (
            orchestrator._agent_backpressure_policies.get("policy_test_agent") is None
        )

        # Switch to shed_oldest policy
        orchestrator.set_agent_backpressure_policy("policy_test_agent", "shed_oldest")
        assert (
            orchestrator._agent_backpressure_policies["policy_test_agent"]
            == "shed_oldest"
        )

        # Switch to drop_newest policy
        orchestrator.set_agent_backpressure_policy("policy_test_agent", "drop_newest")
        assert (
            orchestrator._agent_backpressure_policies["policy_test_agent"]
            == "drop_newest"
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_async_operations(self):
        """Test circuit breaker with async operations and recovery."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        failure_count = [0]  # Use list for mutable closure

        async def flaky_async_operation():
            """Async operation that fails first few times."""
            failure_count[0] += 1
            if failure_count[0] <= 2:
                raise ValueError(f"Async failure {failure_count[0]}")
            return f"Async success after {failure_count[0]} attempts"

        # First two calls should fail and open circuit
        with pytest.raises(ValueError):
            await cb.async_call(flaky_async_operation)
        assert cb.failure_count == 1

        with pytest.raises(ValueError):
            await cb.async_call(flaky_async_operation)
        assert cb.failure_count == 2
        assert cb.is_open

        # Circuit should block further calls
        from gunn.utils.errors import CircuitBreakerOpenError

        with pytest.raises(CircuitBreakerOpenError):
            await cb.async_call(flaky_async_operation)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should succeed after recovery
        result = await cb.async_call(flaky_async_operation)
        assert "Async success after 3 attempts" in result
        assert cb.is_closed

    def test_error_recovery_policy_edge_cases(self):
        """Test error recovery policy edge cases."""
        policy = ErrorRecoveryPolicy(
            max_retries=2,
            retry_delay_ms=100.0,
            backoff_multiplier=3.0,
            max_delay_ms=500.0,
        )

        # Test delay calculation with high multiplier
        assert policy.calculate_retry_delay(0) == 100.0  # 100 * 3^0
        assert policy.calculate_retry_delay(1) == 300.0  # 100 * 3^1
        assert policy.calculate_retry_delay(2) == 500.0  # Capped at max_delay_ms

        # Test should_retry with edge cases
        stale_error = StaleContextError("req_1", 10, 15)

        # At max retries boundary
        assert policy.should_retry(stale_error, 1)  # attempt 1 < max_retries (2)
        assert not policy.should_retry(stale_error, 2)  # attempt 2 == max_retries
        assert not policy.should_retry(stale_error, 3)  # attempt 3 > max_retries

    def test_comprehensive_metrics_recording(self):
        """Test comprehensive metrics recording for all error scenarios."""
        # Import the actual functions to test them directly
        from gunn.utils.telemetry import (
            record_backpressure_event,
            record_error_recovery_action,
            record_queue_high_watermark,
        )

        # Test that the functions exist and can be called without error
        # (The actual metrics recording is tested in other integration tests)
        try:
            record_queue_high_watermark("agent_1", "intent_queue", 150)
            record_backpressure_event("agent_1", "intent_queue", "shed_oldest")
            record_error_recovery_action("StaleContextError", "REGENERATE", "agent_1")
        except Exception as e:
            pytest.fail(f"Metrics recording functions should not raise exceptions: {e}")

        # Test that the functions are properly imported and available
        assert callable(record_queue_high_watermark)
        assert callable(record_backpressure_event)
        assert callable(record_error_recovery_action)

    def test_all_error_types_have_proper_recovery_actions(self):
        """Test that all error types have appropriate recovery actions."""
        policy = ErrorRecoveryPolicy()

        # Test each error type has a defined recovery action
        stale_error = StaleContextError("req_1", 10, 15)
        assert policy.handle_stale_context(stale_error) == RecoveryAction.REGENERATE

        intent = Intent(
            kind="Speak",
            payload={"text": "Hello"},
            context_seq=10,
            req_id="req_123",
            agent_id="agent_1",
            priority=1,
            schema_version="1.0.0",
        )

        from gunn.utils.errors import (
            CircuitBreakerOpenError,
            IntentConflictError,
            TimeoutError,
            ValidationError,
        )

        conflict_error = IntentConflictError(intent, [])
        assert (
            policy.handle_intent_conflict(conflict_error)
            == RecoveryAction.RETRY_WITH_DELAY
        )

        quota_error = QuotaExceededError("agent_1", "intents", 100)
        assert policy.handle_quota_exceeded(quota_error) == RecoveryAction.DEFER

        timeout_error = TimeoutError("operation", 5000.0, 7500.0)
        assert policy.handle_timeout(timeout_error) == RecoveryAction.ABORT

        validation_error = ValidationError(intent, ["Invalid format"])
        assert policy.handle_validation(validation_error) == RecoveryAction.ABORT

        cb_error = CircuitBreakerOpenError("component", 5, 3)
        assert (
            policy.handle_circuit_breaker(cb_error) == RecoveryAction.RETRY_WITH_DELAY
        )

        # Test backpressure errors with different policies
        defer_bp_error = BackpressureError("agent_1", "queue", 150, 100, "defer")
        assert policy.handle_backpressure(defer_bp_error) == RecoveryAction.DEFER

        shed_bp_error = BackpressureError("agent_1", "queue", 150, 100, "shed_oldest")
        assert policy.handle_backpressure(shed_bp_error) == RecoveryAction.SHED_OLDEST

        drop_bp_error = BackpressureError("agent_1", "queue", 150, 100, "drop_newest")
        assert policy.handle_backpressure(drop_bp_error) == RecoveryAction.DROP_NEWEST

    @pytest.mark.asyncio
    async def test_error_handling_under_concurrent_load(
        self, orchestrator, observation_policy
    ):
        """Test error handling system under concurrent load."""
        # Register multiple agents
        agents = []
        for i in range(5):
            agent_id = f"concurrent_agent_{i}"
            await orchestrator.register_agent(agent_id, observation_policy)
            agents.append(agent_id)

        # Submit intents concurrently from all agents
        async def submit_intents_for_agent(agent_id: str, intent_count: int):
            """Submit multiple intents for an agent."""
            results = []
            for i in range(intent_count):
                intent = Intent(
                    kind="Speak",
                    payload={"text": f"Message {i} from {agent_id}"},
                    context_seq=1,
                    req_id=f"{agent_id}_req_{i}",
                    agent_id=agent_id,
                    priority=0,
                    schema_version="1.0.0",
                )

                try:
                    req_id = await orchestrator.submit_intent(intent)
                    results.append(("success", req_id))
                except Exception as e:
                    results.append(("error", type(e).__name__))

            return results

        # Run concurrent submissions
        tasks = []
        for agent_id in agents:
            task = asyncio.create_task(submit_intents_for_agent(agent_id, 8))
            tasks.append(task)

        all_results = await asyncio.gather(*tasks)

        # Analyze results
        total_successes = 0
        total_errors = 0
        error_types = {}

        for agent_results in all_results:
            for result_type, result_value in agent_results:
                if result_type == "success":
                    total_successes += 1
                else:
                    total_errors += 1
                    error_types[result_value] = error_types.get(result_value, 0) + 1

        # Should have some successes and some errors due to quotas/backpressure
        assert total_successes > 0, "Should have some successful intent submissions"

        # Print results for debugging
        print(f"Total successes: {total_successes}")
        print(f"Total errors: {total_errors}")
        print(f"Error types: {error_types}")

        # Verify error handling worked correctly
        assert total_successes + total_errors == 5 * 8  # 5 agents * 8 intents each
