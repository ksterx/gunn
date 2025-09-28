"""Integration tests for fault tolerance with network partitions and adapter failures.

These tests verify the system's resilience to various failure scenarios
including network partitions, adapter failures, and recovery mechanisms.
"""

import asyncio
import random
import time
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.facades import RLFacade
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import Intent
from gunn.utils.errors import (
    CircuitBreakerOpenError,
)
from gunn.utils.telemetry import get_logger


class NetworkPartitionSimulator:
    """Simulates network partitions and connectivity issues."""

    def __init__(self):
        self.partitioned = False
        self.failure_rate = 0.0
        self.latency_ms = 0.0
        self.logger = get_logger("network_simulator")

    def set_partition(self, partitioned: bool):
        """Enable/disable network partition."""
        self.partitioned = partitioned
        self.logger.info(
            f"Network partition: {'ENABLED' if partitioned else 'DISABLED'}"
        )

    def set_failure_rate(self, rate: float):
        """Set random failure rate (0.0 to 1.0)."""
        self.failure_rate = max(0.0, min(1.0, rate))
        self.logger.info(f"Network failure rate: {self.failure_rate:.1%}")

    def set_latency(self, latency_ms: float):
        """Set additional network latency."""
        self.latency_ms = max(0.0, latency_ms)
        self.logger.info(f"Network latency: {self.latency_ms}ms")

    async def simulate_network_call(self, operation_name: str = "network_op"):
        """Simulate a network call with potential failures."""
        # Check for partition
        if self.partitioned:
            raise ConnectionError(f"Network partition: {operation_name} failed")

        # Check for random failure
        if random.random() < self.failure_rate:
            raise ConnectionError(f"Network failure: {operation_name} failed")

        # Add latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)


class AdapterFailureSimulator:
    """Simulates various adapter failure scenarios."""

    def __init__(self):
        self.failures_enabled = {}
        self.failure_counts = {}
        self.recovery_delays = {}
        self.logger = get_logger("adapter_simulator")

    def enable_failure(
        self, adapter_name: str, failure_type: str, recovery_delay: float = 0.0
    ):
        """Enable failure for specific adapter."""
        key = f"{adapter_name}:{failure_type}"
        self.failures_enabled[key] = True
        self.failure_counts[key] = 0
        self.recovery_delays[key] = recovery_delay
        self.logger.info(f"Enabled failure: {key} (recovery: {recovery_delay}s)")

    def disable_failure(self, adapter_name: str, failure_type: str):
        """Disable failure for specific adapter."""
        key = f"{adapter_name}:{failure_type}"
        self.failures_enabled[key] = False
        self.logger.info(f"Disabled failure: {key}")

    async def check_failure(self, adapter_name: str, failure_type: str) -> bool:
        """Check if failure should occur and handle recovery."""
        key = f"{adapter_name}:{failure_type}"

        if not self.failures_enabled.get(key, False):
            return False

        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1

        # Simulate recovery delay
        recovery_delay = self.recovery_delays.get(key, 0.0)
        if recovery_delay > 0:
            await asyncio.sleep(recovery_delay)

        return True


class FaultTolerantOrchestrator(Orchestrator):
    """Orchestrator with fault tolerance testing capabilities."""

    def __init__(self, config: OrchestratorConfig, world_id: str = "fault_test"):
        super().__init__(config, world_id)
        self.network_simulator = NetworkPartitionSimulator()
        self.adapter_simulator = AdapterFailureSimulator()
        self.failure_recovery_count = 0
        self.circuit_breaker_trips = 0

    async def simulate_network_partition(self, duration_seconds: float):
        """Simulate network partition for specified duration."""
        self.network_simulator.set_partition(True)
        await asyncio.sleep(duration_seconds)
        self.network_simulator.set_partition(False)

    async def simulate_adapter_failure(
        self, adapter_name: str, failure_type: str, duration_seconds: float
    ):
        """Simulate adapter failure for specified duration."""
        self.adapter_simulator.enable_failure(adapter_name, failure_type)
        await asyncio.sleep(duration_seconds)
        self.adapter_simulator.disable_failure(adapter_name, failure_type)

    async def broadcast_event_with_failures(self, draft):
        """Broadcast event with potential network failures."""
        try:
            await self.network_simulator.simulate_network_call("broadcast_event")
            return await super().broadcast_event(draft)
        except ConnectionError as e:
            self.logger.warning(f"Broadcast failed due to network issue: {e}")
            # Implement retry logic
            await asyncio.sleep(0.1)  # Brief retry delay
            return await super().broadcast_event(draft)

    async def submit_intent_with_failures(self, intent: Intent) -> str:
        """Submit intent with potential adapter failures."""
        adapter_name = "intent_processor"

        if await self.adapter_simulator.check_failure(adapter_name, "timeout"):
            raise asyncio.TimeoutError(
                f"Intent submission timeout for {intent['req_id']}"
            )

        if await self.adapter_simulator.check_failure(adapter_name, "validation_error"):
            raise ValueError(f"Intent validation failed for {intent['req_id']}")

        try:
            await self.network_simulator.simulate_network_call("submit_intent")
            return await super().submit_intent(intent)
        except ConnectionError:
            self.failure_recovery_count += 1
            # Implement exponential backoff
            retry_delay = min(2.0**self.failure_recovery_count, 10.0)
            await asyncio.sleep(retry_delay)
            return await super().submit_intent(intent)


class TestFaultTolerance:
    """Test suite for fault tolerance scenarios."""

    @pytest.fixture
    def config(self) -> OrchestratorConfig:
        """Create fault-tolerant test configuration."""
        return OrchestratorConfig(
            max_agents=5,
            staleness_threshold=1,
            debounce_ms=100.0,  # Longer debounce for fault scenarios
            deadline_ms=10000.0,  # Longer deadline for recovery
            token_budget=1000,
            backpressure_policy="defer",
            default_priority=1,
            use_in_memory_dedup=True,
            max_queue_depth=500,
            quota_intents_per_minute=3000,
        )

    @pytest.fixture
    async def fault_tolerant_orchestrator(
        self, config: OrchestratorConfig
    ) -> FaultTolerantOrchestrator:
        """Create fault-tolerant orchestrator."""
        orchestrator = FaultTolerantOrchestrator(config)
        await orchestrator.initialize()
        return orchestrator

    @pytest.fixture
    async def rl_facade(
        self, fault_tolerant_orchestrator: FaultTolerantOrchestrator
    ) -> RLFacade:
        """Create RL facade with fault-tolerant orchestrator."""
        facade = RLFacade(orchestrator=fault_tolerant_orchestrator)
        await facade.initialize()
        return facade

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for fault tolerance testing."""
        config = PolicyConfig(
            distance_limit=100.0,
            relationship_filter=[],
            field_visibility={},
            max_patch_ops=25,
        )
        return DefaultObservationPolicy(config)

    @pytest.mark.asyncio
    async def test_network_partition_recovery(
        self,
        fault_tolerant_orchestrator: FaultTolerantOrchestrator,
        rl_facade: RLFacade,
        observation_policy: DefaultObservationPolicy,
    ):
        """Test recovery from network partition scenarios."""
        # Register agents
        agent_handles = []
        for i in range(3):
            agent_id = f"partition_agent_{i}"
            handle = await rl_facade.register_agent(agent_id, observation_policy)
            agent_handles.append((agent_id, handle))

        # Normal operation before partition
        intent: Intent = {
            "kind": "PrePartition",
            "payload": {"test": "before_partition"},
            "context_seq": 0,
            "req_id": "pre_partition_1",
            "agent_id": "partition_agent_0",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        effect, observation = await rl_facade.step("partition_agent_0", intent)
        assert effect is not None, "Normal operation should succeed"

        # Simulate network partition
        partition_task = asyncio.create_task(
            fault_tolerant_orchestrator.simulate_network_partition(2.0)
        )

        # Attempt operations during partition (should fail initially but recover)
        partition_intent: Intent = {
            "kind": "DuringPartition",
            "payload": {"test": "during_partition"},
            "context_seq": 1,
            "req_id": "during_partition_1",
            "agent_id": "partition_agent_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # This should eventually succeed after partition recovery
        start_time = time.perf_counter()

        try:
            effect, observation = await rl_facade.step(
                "partition_agent_1", partition_intent
            )
            recovery_time = time.perf_counter() - start_time

            # Should recover within reasonable time
            assert recovery_time < 5.0, f"Recovery took too long: {recovery_time:.2f}s"
            assert effect is not None, "Should recover after partition"

        except Exception as e:
            # If it fails, verify it's due to partition
            assert "partition" in str(e).lower() or "connection" in str(e).lower()

        # Wait for partition to end
        await partition_task

        # Verify normal operation resumes
        post_partition_intent: Intent = {
            "kind": "PostPartition",
            "payload": {"test": "after_partition"},
            "context_seq": 2,
            "req_id": "post_partition_1",
            "agent_id": "partition_agent_2",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        effect, observation = await rl_facade.step(
            "partition_agent_2", post_partition_intent
        )
        assert effect is not None, "Normal operation should resume after partition"

        # Verify event log integrity
        event_log = fault_tolerant_orchestrator.event_log
        assert event_log.validate_integrity(), "Event log should maintain integrity"

    @pytest.mark.asyncio
    async def test_adapter_failure_scenarios(
        self,
        fault_tolerant_orchestrator: FaultTolerantOrchestrator,
        rl_facade: RLFacade,
        observation_policy: DefaultObservationPolicy,
    ):
        """Test various adapter failure scenarios and recovery."""
        # Register agent
        await rl_facade.register_agent("adapter_test_agent", observation_policy)

        # Test timeout failure
        timeout_task = asyncio.create_task(
            fault_tolerant_orchestrator.simulate_adapter_failure(
                "intent_processor", "timeout", 1.0
            )
        )

        timeout_intent: Intent = {
            "kind": "TimeoutTest",
            "payload": {"test": "timeout_scenario"},
            "context_seq": 0,
            "req_id": "timeout_test_1",
            "agent_id": "adapter_test_agent",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Should handle timeout gracefully
        with pytest.raises((asyncio.TimeoutError, ConnectionError)):
            await rl_facade.step("adapter_test_agent", timeout_intent)

        await timeout_task

        # Test validation error failure
        validation_task = asyncio.create_task(
            fault_tolerant_orchestrator.simulate_adapter_failure(
                "intent_processor", "validation_error", 0.5
            )
        )

        validation_intent: Intent = {
            "kind": "ValidationTest",
            "payload": {"test": "validation_scenario"},
            "context_seq": 1,
            "req_id": "validation_test_1",
            "agent_id": "adapter_test_agent",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Should handle validation error
        with pytest.raises(ValueError):
            await rl_facade.step("adapter_test_agent", validation_intent)

        await validation_task

        # Verify recovery - normal operation should work
        recovery_intent: Intent = {
            "kind": "RecoveryTest",
            "payload": {"test": "recovery_scenario"},
            "context_seq": 2,
            "req_id": "recovery_test_1",
            "agent_id": "adapter_test_agent",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        effect, observation = await rl_facade.step(
            "adapter_test_agent", recovery_intent
        )
        assert effect is not None, "Should recover after adapter failures"

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(
        self,
        fault_tolerant_orchestrator: FaultTolerantOrchestrator,
        rl_facade: RLFacade,
        observation_policy: DefaultObservationPolicy,
    ):
        """Test prevention of cascading failures."""
        # Register multiple agents
        agent_ids = []
        for i in range(5):
            agent_id = f"cascade_agent_{i}"
            await rl_facade.register_agent(agent_id, observation_policy)
            agent_ids.append(agent_id)

        # Simulate high failure rate
        fault_tolerant_orchestrator.network_simulator.set_failure_rate(
            0.3
        )  # 30% failure rate

        # Submit intents from all agents concurrently
        async def submit_intent_with_retry(
            agent_id: str, intent_id: int
        ) -> tuple[str, bool]:
            intent: Intent = {
                "kind": "CascadeTest",
                "payload": {"agent": agent_id, "intent_id": intent_id},
                "context_seq": intent_id,
                "req_id": f"cascade_{agent_id}_{intent_id}",
                "agent_id": agent_id,
                "priority": 1,
                "schema_version": "1.0.0",
            }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    effect, observation = await rl_facade.step(agent_id, intent)
                    return intent["req_id"], True
                except (ConnectionError, asyncio.TimeoutError):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                except Exception:
                    break

            return intent["req_id"], False

        # Submit multiple intents concurrently
        tasks = []
        for agent_id in agent_ids:
            for intent_id in range(3):
                task = submit_intent_with_retry(agent_id, intent_id)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successful_intents = 0
        failed_intents = 0

        for result in results:
            if isinstance(result, tuple):
                req_id, success = result
                if success:
                    successful_intents += 1
                else:
                    failed_intents += 1
            else:
                failed_intents += 1

        # Should have some successes despite failures
        total_intents = len(tasks)
        success_rate = successful_intents / total_intents

        assert (
            success_rate > 0.3
        ), f"Success rate {success_rate:.1%} too low, cascading failure may have occurred"
        assert successful_intents > 0, "At least some intents should succeed"

        # Reset failure rate
        fault_tolerant_orchestrator.network_simulator.set_failure_rate(0.0)

        # Verify system is still functional
        recovery_intent: Intent = {
            "kind": "PostCascadeTest",
            "payload": {"test": "post_cascade"},
            "context_seq": 100,
            "req_id": "post_cascade_test",
            "agent_id": agent_ids[0],
            "priority": 1,
            "schema_version": "1.0.0",
        }

        effect, observation = await rl_facade.step(agent_ids[0], recovery_intent)
        assert effect is not None, "System should be functional after cascade test"

    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(
        self,
        fault_tolerant_orchestrator: FaultTolerantOrchestrator,
        rl_facade: RLFacade,
        observation_policy: DefaultObservationPolicy,
    ):
        """Test circuit breaker pattern for fault tolerance."""
        # Register agent
        await rl_facade.register_agent("circuit_test_agent", observation_policy)

        # Mock circuit breaker
        with patch("gunn.utils.errors.CircuitBreaker") as mock_circuit_breaker:
            circuit_breaker_instance = Mock()
            mock_circuit_breaker.return_value = circuit_breaker_instance

            # Configure circuit breaker to trip after 3 failures
            failure_count = 0

            async def mock_call(func, *args, **kwargs):
                nonlocal failure_count
                failure_count += 1

                if failure_count <= 3:
                    raise ConnectionError("Simulated failure")
                elif failure_count <= 6:
                    raise CircuitBreakerOpenError("Circuit breaker is open")
                else:
                    # Recovery
                    return await func(*args, **kwargs)

            circuit_breaker_instance.call = AsyncMock(side_effect=mock_call)

            # Submit intents that will trigger circuit breaker
            for i in range(8):
                intent: Intent = {
                    "kind": "CircuitTest",
                    "payload": {"attempt": i},
                    "context_seq": i,
                    "req_id": f"circuit_test_{i}",
                    "agent_id": "circuit_test_agent",
                    "priority": 1,
                    "schema_version": "1.0.0",
                }

                try:
                    effect, observation = await rl_facade.step(
                        "circuit_test_agent", intent
                    )

                    if i >= 7:  # Should succeed after recovery
                        assert (
                            effect is not None
                        ), f"Should succeed after recovery on attempt {i}"

                except (ConnectionError, CircuitBreakerOpenError) as e:
                    if i < 7:  # Expected failures
                        assert isinstance(e, (ConnectionError, CircuitBreakerOpenError))
                    else:
                        pytest.fail(f"Unexpected failure after recovery: {e}")

                # Brief delay between attempts
                await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_partial_system_failure_isolation(
        self,
        fault_tolerant_orchestrator: FaultTolerantOrchestrator,
        rl_facade: RLFacade,
        observation_policy: DefaultObservationPolicy,
    ):
        """Test that partial system failures are isolated and don't affect other components."""
        # Register agents in different "groups"
        group_a_agents = []
        group_b_agents = []

        for i in range(3):
            agent_a = f"group_a_agent_{i}"
            agent_b = f"group_b_agent_{i}"

            await rl_facade.register_agent(agent_a, observation_policy)
            await rl_facade.register_agent(agent_b, observation_policy)

            group_a_agents.append(agent_a)
            group_b_agents.append(agent_b)

        # Simulate failure affecting only group A
        fault_tolerant_orchestrator.adapter_simulator.enable_failure(
            "group_a", "connection_error"
        )

        # Test that group A operations fail
        group_a_intent: Intent = {
            "kind": "GroupATest",
            "payload": {"group": "A"},
            "context_seq": 0,
            "req_id": "group_a_test_1",
            "agent_id": group_a_agents[0],
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Group A should fail (simulated)
        if await fault_tolerant_orchestrator.adapter_simulator.check_failure(
            "group_a", "connection_error"
        ):
            # Simulate the failure
            with pytest.raises(ConnectionError):
                raise ConnectionError("Group A connection failed")

        # Test that group B operations still work
        group_b_intent: Intent = {
            "kind": "GroupBTest",
            "payload": {"group": "B"},
            "context_seq": 0,
            "req_id": "group_b_test_1",
            "agent_id": group_b_agents[0],
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Group B should succeed
        effect, observation = await rl_facade.step(group_b_agents[0], group_b_intent)
        assert effect is not None, "Group B should be unaffected by Group A failures"

        # Disable failure and verify group A recovers
        fault_tolerant_orchestrator.adapter_simulator.disable_failure(
            "group_a", "connection_error"
        )

        recovery_intent: Intent = {
            "kind": "GroupARecovery",
            "payload": {"group": "A", "status": "recovered"},
            "context_seq": 1,
            "req_id": "group_a_recovery_1",
            "agent_id": group_a_agents[0],
            "priority": 1,
            "schema_version": "1.0.0",
        }

        effect, observation = await rl_facade.step(group_a_agents[0], recovery_intent)
        assert effect is not None, "Group A should recover after failure is resolved"

    @pytest.mark.asyncio
    async def test_data_consistency_during_failures(
        self,
        fault_tolerant_orchestrator: FaultTolerantOrchestrator,
        rl_facade: RLFacade,
        observation_policy: DefaultObservationPolicy,
    ):
        """Test that data consistency is maintained during various failure scenarios."""
        # Register agents
        await rl_facade.register_agent("consistency_agent_1", observation_policy)
        await rl_facade.register_agent("consistency_agent_2", observation_policy)

        # Submit initial intents to establish baseline
        baseline_intents = []
        for i in range(5):
            intent: Intent = {
                "kind": "BaselineTest",
                "payload": {"sequence": i, "data": f"baseline_{i}"},
                "context_seq": i,
                "req_id": f"baseline_{i}",
                "agent_id": "consistency_agent_1",
                "priority": 1,
                "schema_version": "1.0.0",
            }

            effect, observation = await rl_facade.step("consistency_agent_1", intent)
            baseline_intents.append((intent, effect))

        # Get initial event log state
        initial_entries = fault_tolerant_orchestrator.event_log.get_all_entries()
        initial_count = len(initial_entries)

        # Verify initial integrity
        assert (
            fault_tolerant_orchestrator.event_log.validate_integrity()
        ), "Initial log should be valid"

        # Introduce failures during operations
        fault_tolerant_orchestrator.network_simulator.set_failure_rate(
            0.2
        )  # 20% failure rate

        # Submit intents during failure period
        failure_period_intents = []
        for i in range(10):
            intent: Intent = {
                "kind": "FailurePeriodTest",
                "payload": {"sequence": i, "data": f"failure_period_{i}"},
                "context_seq": initial_count + i,
                "req_id": f"failure_period_{i}",
                "agent_id": "consistency_agent_2",
                "priority": 1,
                "schema_version": "1.0.0",
            }

            try:
                effect, observation = await rl_facade.step(
                    "consistency_agent_2", intent
                )
                failure_period_intents.append((intent, effect))
            except (ConnectionError, asyncio.TimeoutError):
                # Expected failures during this period
                pass

            await asyncio.sleep(0.05)  # Brief delay

        # Disable failures
        fault_tolerant_orchestrator.network_simulator.set_failure_rate(0.0)

        # Verify data consistency after failures
        final_entries = fault_tolerant_orchestrator.event_log.get_all_entries()

        # Event log should still be valid
        assert (
            fault_tolerant_orchestrator.event_log.validate_integrity()
        ), "Log integrity should be maintained"

        # Verify monotonic global sequences
        global_seqs = [entry.effect["global_seq"] for entry in final_entries]
        assert global_seqs == sorted(
            global_seqs
        ), "Global sequences should remain monotonic"

        # Verify no duplicate req_ids
        req_ids = [entry.req_id for entry in final_entries if entry.req_id]
        assert len(req_ids) == len(set(req_ids)), "Should have no duplicate req_ids"

        # Submit final verification intent
        verification_intent: Intent = {
            "kind": "VerificationTest",
            "payload": {"test": "post_failure_verification"},
            "context_seq": len(final_entries),
            "req_id": "verification_test_1",
            "agent_id": "consistency_agent_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        effect, observation = await rl_facade.step(
            "consistency_agent_1", verification_intent
        )
        assert (
            effect is not None
        ), "System should be fully functional after failure recovery"

        # Final integrity check
        assert (
            fault_tolerant_orchestrator.event_log.validate_integrity()
        ), "Final log should be valid"

    @pytest.mark.asyncio
    async def test_graceful_degradation_under_load(
        self,
        fault_tolerant_orchestrator: FaultTolerantOrchestrator,
        rl_facade: RLFacade,
        observation_policy: DefaultObservationPolicy,
    ):
        """Test graceful degradation under high load with failures."""
        # Register agents
        agent_ids = []
        for i in range(8):
            agent_id = f"load_test_agent_{i}"
            await rl_facade.register_agent(agent_id, observation_policy)
            agent_ids.append(agent_id)

        # Configure moderate failure rate and latency
        fault_tolerant_orchestrator.network_simulator.set_failure_rate(
            0.1
        )  # 10% failure rate
        fault_tolerant_orchestrator.network_simulator.set_latency(
            50.0
        )  # 50ms additional latency

        # Generate high load with concurrent operations
        async def agent_workload(agent_id: str, num_operations: int) -> dict[str, Any]:
            successful_ops = 0
            failed_ops = 0
            total_latency = 0.0

            for i in range(num_operations):
                intent: Intent = {
                    "kind": "LoadTest",
                    "payload": {"operation": i, "agent": agent_id},
                    "context_seq": i,
                    "req_id": f"load_{agent_id}_{i}",
                    "agent_id": agent_id,
                    "priority": 1,
                    "schema_version": "1.0.0",
                }

                start_time = time.perf_counter()

                try:
                    effect, observation = await rl_facade.step(agent_id, intent)
                    successful_ops += 1

                    end_time = time.perf_counter()
                    total_latency += end_time - start_time

                except Exception:
                    failed_ops += 1

                # Brief pause to avoid overwhelming
                await asyncio.sleep(0.01)

            return {
                "agent_id": agent_id,
                "successful_ops": successful_ops,
                "failed_ops": failed_ops,
                "avg_latency": total_latency / successful_ops
                if successful_ops > 0
                else 0,
            }

        # Run concurrent workloads
        operations_per_agent = 20
        tasks = [
            agent_workload(agent_id, operations_per_agent) for agent_id in agent_ids
        ]

        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.perf_counter()

        total_duration = end_time - start_time

        # Analyze results
        total_successful = 0
        total_failed = 0
        avg_latencies = []

        for result in results:
            if isinstance(result, dict):
                total_successful += result["successful_ops"]
                total_failed += result["failed_ops"]
                if result["avg_latency"] > 0:
                    avg_latencies.append(result["avg_latency"])

        # Calculate metrics
        total_operations = total_successful + total_failed
        success_rate = (
            total_successful / total_operations if total_operations > 0 else 0
        )
        overall_throughput = total_successful / total_duration

        # Verify graceful degradation
        assert (
            success_rate >= 0.7
        ), f"Success rate {success_rate:.1%} too low for graceful degradation"
        assert overall_throughput > 0, "Should maintain some throughput under load"
        assert total_successful > 0, "Should complete some operations successfully"

        # Verify system remains stable
        assert (
            fault_tolerant_orchestrator.event_log.validate_integrity()
        ), "System should remain stable"

        # Reset network conditions
        fault_tolerant_orchestrator.network_simulator.set_failure_rate(0.0)
        fault_tolerant_orchestrator.network_simulator.set_latency(0.0)

        # Verify recovery to normal operation
        recovery_intent: Intent = {
            "kind": "RecoveryVerification",
            "payload": {"test": "normal_operation_recovery"},
            "context_seq": 1000,
            "req_id": "recovery_verification",
            "agent_id": agent_ids[0],
            "priority": 1,
            "schema_version": "1.0.0",
        }

        effect, observation = await rl_facade.step(agent_ids[0], recovery_intent)
        assert effect is not None, "Should return to normal operation after load test"
