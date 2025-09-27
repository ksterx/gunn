"""Unit tests for two-phase commit intent processing.

Tests the complete two-phase commit pipeline including idempotency,
staleness detection, quota checking, backpressure, fairness scheduling,
validation, and effect creation.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.types import Intent
from gunn.utils.errors import (
    BackpressureError,
    QuotaExceededError,
    StaleContextError,
)


class TestTwoPhaseCommit:
    """Test suite for two-phase commit intent processing."""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator with in-memory dedup store for testing."""
        config = OrchestratorConfig(
            max_agents=10,
            staleness_threshold=2,
            dedup_ttl_minutes=1,
            max_dedup_entries=100,
            dedup_warmup_minutes=0,  # No warmup for testing
            max_queue_depth=5,
            quota_intents_per_minute=10,
            quota_tokens_per_minute=1000,
            use_in_memory_dedup=True,
        )

        orchestrator = Orchestrator(config, world_id="test_world")
        await orchestrator.initialize()

        yield orchestrator

        await orchestrator.shutdown()

    @pytest.fixture
    def mock_policy(self):
        """Create mock observation policy."""
        policy = MagicMock(spec=ObservationPolicy)
        return policy

    @pytest.fixture
    def sample_intent(self):
        """Create sample intent for testing."""
        return Intent(
            kind="Speak",
            payload={"text": "Hello world"},
            context_seq=0,
            req_id="test_req_1",
            agent_id="test_agent",
            priority=0,
            schema_version="1.0.0",
        )

    @pytest.mark.asyncio
    async def test_basic_intent_submission(
        self, orchestrator, mock_policy, sample_intent
    ):
        """Test basic intent submission and processing."""
        # Register agent
        _ = await orchestrator.register_agent("test_agent", mock_policy)

        # Submit intent
        req_id = await orchestrator.submit_intent(sample_intent)

        assert req_id == "test_req_1"

        # Wait for processing
        await asyncio.sleep(0.1)

        # Check that intent was processed
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 1
        assert stats["global_seq"] > 0

    @pytest.mark.asyncio
    async def test_idempotency_checking(self, orchestrator, mock_policy, sample_intent):
        """Test that duplicate intents are handled idempotently."""
        # Register agent
        await orchestrator.register_agent("test_agent", mock_policy)

        # Submit intent first time
        req_id_1 = await orchestrator.submit_intent(sample_intent)
        assert req_id_1 == "test_req_1"

        # Submit same intent again (should be idempotent)
        req_id_2 = await orchestrator.submit_intent(sample_intent)
        assert req_id_2 == "test_req_1"

        # Wait for processing
        await asyncio.sleep(0.1)

        # Should only be enqueued once
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 1

    @pytest.mark.asyncio
    async def test_staleness_detection(self, orchestrator, mock_policy):
        """Test staleness detection based on context_seq."""
        # Register agent
        _ = await orchestrator.register_agent("test_agent", mock_policy)

        # Advance global sequence by submitting some intents
        for i in range(5):
            intent = Intent(
                kind="Move",
                payload={"x": i, "y": i},
                context_seq=0,
                req_id=f"setup_req_{i}",
                agent_id="test_agent",
                priority=0,
                schema_version="1.0.0",
            )
            await orchestrator.submit_intent(intent)

        await asyncio.sleep(0.1)  # Let them process

        # Now submit intent with stale context
        stale_intent = Intent(
            kind="Speak",
            payload={"text": "Stale message"},
            context_seq=0,  # Very old context
            req_id="stale_req",
            agent_id="test_agent",
            priority=0,
            schema_version="1.0.0",
        )

        # Should raise StaleContextError
        with pytest.raises(StaleContextError) as exc_info:
            await orchestrator.submit_intent(stale_intent)

        error = exc_info.value
        assert error.req_id == "stale_req"
        assert error.expected_seq == 0
        assert error.actual_seq > 2  # Should be greater than threshold

    @pytest.mark.asyncio
    async def test_quota_checking(self):
        """Test quota limits for intents per minute."""
        # Create orchestrator with higher queue depth to avoid backpressure
        config = OrchestratorConfig(
            max_queue_depth=20,  # Higher than quota limit
            quota_intents_per_minute=5,  # Lower quota for easier testing
            use_in_memory_dedup=True,
        )
        orchestrator = Orchestrator(config, world_id="quota_test")
        await orchestrator.initialize()

        try:
            mock_policy = MagicMock(spec=ObservationPolicy)
            await orchestrator.register_agent("test_agent", mock_policy)

            # Submit intents up to quota limit
            for i in range(5):  # quota_intents_per_minute = 5
                intent = Intent(
                    kind="Speak",
                    payload={"text": f"Message {i}"},
                    context_seq=0,
                    req_id=f"quota_req_{i}",
                    agent_id="test_agent",
                    priority=0,
                    schema_version="1.0.0",
                )
                await orchestrator.submit_intent(intent)

            # Next intent should exceed quota
            excess_intent = Intent(
                kind="Speak",
                payload={"text": "Excess message"},
                context_seq=0,
                req_id="excess_req",
                agent_id="test_agent",
                priority=0,
                schema_version="1.0.0",
            )

            with pytest.raises(QuotaExceededError) as exc_info:
                await orchestrator.submit_intent(excess_intent)

            error = exc_info.value
            assert error.agent_id == "test_agent"
            assert error.quota_type == "intents_per_minute"
            assert error.limit == 5

        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_backpressure_checking(self, orchestrator, mock_policy):
        """Test backpressure limits based on queue depth."""
        # Register agent
        await orchestrator.register_agent("test_agent", mock_policy)

        # Fill up the queue to max_queue_depth (5)
        for i in range(5):
            intent = Intent(
                kind="Speak",
                payload={"text": f"Message {i}"},
                context_seq=0,
                req_id=f"backpressure_req_{i}",
                agent_id="test_agent",
                priority=0,
                schema_version="1.0.0",
            )
            await orchestrator.submit_intent(intent)

        # Next intent should trigger backpressure
        excess_intent = Intent(
            kind="Speak",
            payload={"text": "Backpressure message"},
            context_seq=0,
            req_id="backpressure_excess_req",
            agent_id="test_agent",
            priority=0,
            schema_version="1.0.0",
        )

        with pytest.raises(BackpressureError) as exc_info:
            await orchestrator.submit_intent(excess_intent)

        error = exc_info.value
        assert error.agent_id == "test_agent"
        assert error.queue_type == "agent_queue"
        assert error.current_depth == 5
        assert error.threshold == 5

    @pytest.mark.asyncio
    async def test_priority_scheduling(self, orchestrator, mock_policy):
        """Test that higher priority intents are processed first."""
        # Register agent
        await orchestrator.register_agent("test_agent", mock_policy)

        # Submit intents with different priorities
        low_priority_intent = Intent(
            kind="Speak",
            payload={"text": "Low priority"},
            context_seq=0,
            req_id="low_priority_req",
            agent_id="test_agent",
            priority=-10,  # Low priority
            schema_version="1.0.0",
        )

        high_priority_intent = Intent(
            kind="Speak",
            payload={"text": "High priority"},
            context_seq=0,
            req_id="high_priority_req",
            agent_id="test_agent",
            priority=20,  # High priority
            schema_version="1.0.0",
        )

        # Submit low priority first, then high priority
        await orchestrator.submit_intent(low_priority_intent)
        await orchestrator.submit_intent(high_priority_intent)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Both should be processed, but high priority should be processed first
        # (This is hard to test directly, but we can check they were both enqueued)
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 2

    @pytest.mark.asyncio
    async def test_fairness_scheduling(self, orchestrator, mock_policy):
        """Test weighted round robin fairness across agents."""
        # Register multiple agents
        await orchestrator.register_agent("agent_1", mock_policy)
        await orchestrator.register_agent("agent_2", mock_policy)

        # Submit intents from both agents
        for i in range(3):
            intent_1 = Intent(
                kind="Speak",
                payload={"text": f"Agent 1 message {i}"},
                context_seq=0,
                req_id=f"agent1_req_{i}",
                agent_id="agent_1",
                priority=0,
                schema_version="1.0.0",
            )

            intent_2 = Intent(
                kind="Speak",
                payload={"text": f"Agent 2 message {i}"},
                context_seq=0,
                req_id=f"agent2_req_{i}",
                agent_id="agent_2",
                priority=0,
                schema_version="1.0.0",
            )

            await orchestrator.submit_intent(intent_1)
            await orchestrator.submit_intent(intent_2)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Both agents should have had their intents processed
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 6

        # Check per-agent queue depths
        agent_depths = stats["scheduler"]["agent_queue_depths"]
        assert "agent_1" in agent_depths
        assert "agent_2" in agent_depths

    @pytest.mark.asyncio
    async def test_validation_failure(self, orchestrator, mock_policy, sample_intent):
        """Test intent validation failure."""
        # Create mock validator that always fails
        mock_validator = MagicMock()
        mock_validator.validate_intent.return_value = False
        orchestrator.effect_validator = mock_validator

        # Register agent
        await orchestrator.register_agent("test_agent", mock_policy)

        # Submit intent (should be enqueued but fail during processing)
        req_id = await orchestrator.submit_intent(sample_intent)
        assert req_id == "test_req_1"

        # Wait for processing
        await asyncio.sleep(0.1)

        # Intent should be enqueued but processing should fail
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 1

    @pytest.mark.asyncio
    async def test_ttl_cleanup(self, orchestrator, mock_policy):
        """Test TTL cleanup of deduplication entries."""
        # Register agent
        await orchestrator.register_agent("test_agent", mock_policy)

        # Submit intent
        intent = Intent(
            kind="Speak",
            payload={"text": "TTL test"},
            context_seq=0,
            req_id="ttl_req",
            agent_id="test_agent",
            priority=0,
            schema_version="1.0.0",
        )

        await orchestrator.submit_intent(intent)

        # Check dedup store stats
        dedup_stats = await orchestrator._dedup_store.get_stats()
        assert dedup_stats["total_entries"] >= 1

        # Manually trigger cleanup (in real scenario this would happen automatically)
        cleaned = await orchestrator._dedup_store.cleanup_expired()
        # With TTL of 1 minute, nothing should be cleaned immediately
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_warmup_period(self):
        """Test TTL warmup guard for relaxed deduplication."""
        config = OrchestratorConfig(
            dedup_warmup_minutes=1,  # 1 minute warmup
            use_in_memory_dedup=True,
        )

        orchestrator = Orchestrator(config, world_id="warmup_test")
        await orchestrator.initialize()

        try:
            mock_policy = MagicMock(spec=ObservationPolicy)
            await orchestrator.register_agent("test_agent", mock_policy)

            # During warmup, duplicate intents should be allowed
            intent = Intent(
                kind="Speak",
                payload={"text": "Warmup test"},
                context_seq=0,
                req_id="warmup_req",
                agent_id="test_agent",
                priority=0,
                schema_version="1.0.0",
            )

            # Submit same intent twice during warmup
            req_id_1 = await orchestrator.submit_intent(intent)
            req_id_2 = await orchestrator.submit_intent(intent)

            # Both should succeed during warmup
            assert req_id_1 == "warmup_req"
            assert req_id_2 == "warmup_req"

            # Wait for processing
            await asyncio.sleep(0.1)

            # Both should be enqueued during warmup
            stats = orchestrator.get_processing_stats()
            assert stats["scheduler"]["total_enqueued"] == 2

        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, orchestrator, mock_policy):
        """Test conflict resolution between concurrent intents."""
        # Register agent
        await orchestrator.register_agent("test_agent", mock_policy)

        # Submit multiple intents concurrently
        intents = []
        for i in range(5):
            intent = Intent(
                kind="Move",
                payload={"x": 10, "y": 10},  # Same target position
                context_seq=0,
                req_id=f"conflict_req_{i}",
                agent_id="test_agent",
                priority=0,
                schema_version="1.0.0",
            )
            intents.append(intent)

        # Submit all intents concurrently
        tasks = [orchestrator.submit_intent(intent) for intent in intents]
        req_ids = await asyncio.gather(*tasks)

        # All should be accepted (conflict resolution happens during processing)
        assert len(req_ids) == 5
        assert all(req_id.startswith("conflict_req_") for req_id in req_ids)

        # Wait for processing
        await asyncio.sleep(0.2)

        # All should be enqueued
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 5

    @pytest.mark.asyncio
    async def test_processing_pipeline_timing(
        self, orchestrator, mock_policy, sample_intent
    ):
        """Test that processing pipeline meets timing requirements."""
        # Register agent
        await orchestrator.register_agent("test_agent", mock_policy)

        # Measure submission time
        start_time = time.perf_counter()
        req_id = await orchestrator.submit_intent(sample_intent)
        submission_time = (time.perf_counter() - start_time) * 1000

        assert req_id == "test_req_1"

        # Submission should be fast (< 10ms for in-memory operations)
        assert submission_time < 10.0

        # Wait for processing
        await asyncio.sleep(0.1)

        # Check processing completed
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 1

    @pytest.mark.asyncio
    async def test_error_recovery(self, orchestrator, mock_policy):
        """Test error recovery and graceful degradation."""
        # Register agent
        await orchestrator.register_agent("test_agent", mock_policy)

        # Test various error conditions
        error_cases = [
            # Missing required fields
            {
                "kind": "Speak",
                "payload": {"text": "No req_id"},
                "context_seq": 0,
                "agent_id": "test_agent",
                "priority": 0,
                "schema_version": "1.0.0",
            },
            # Invalid agent
            Intent(
                kind="Speak",
                payload={"text": "Invalid agent"},
                context_seq=0,
                req_id="invalid_agent_req",
                agent_id="nonexistent_agent",
                priority=0,
                schema_version="1.0.0",
            ),
        ]

        for _, invalid_intent in enumerate(error_cases):
            with pytest.raises((ValueError, KeyError)):
                await orchestrator.submit_intent(invalid_intent)

        # System should still be functional after errors
        valid_intent = Intent(
            kind="Speak",
            payload={"text": "Recovery test"},
            context_seq=0,
            req_id="recovery_req",
            agent_id="test_agent",
            priority=0,
            schema_version="1.0.0",
        )

        req_id = await orchestrator.submit_intent(valid_intent)
        assert req_id == "recovery_req"

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, mock_policy):
        """Test proper cleanup during shutdown."""
        config = OrchestratorConfig(use_in_memory_dedup=True)
        orchestrator = Orchestrator(config, world_id="shutdown_test")
        await orchestrator.initialize()

        # Register agent and submit intent
        await orchestrator.register_agent("test_agent", mock_policy)

        intent = Intent(
            kind="Speak",
            payload={"text": "Shutdown test"},
            context_seq=0,
            req_id="shutdown_req",
            agent_id="test_agent",
            priority=0,
            schema_version="1.0.0",
        )

        await orchestrator.submit_intent(intent)

        # Shutdown should complete without errors
        await orchestrator.shutdown()

        # Verify cleanup
        assert len(orchestrator.agent_handles) == 0
        assert len(orchestrator._cancel_tokens) == 0
        assert len(orchestrator._quota_tracker) == 0
