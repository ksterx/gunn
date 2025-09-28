"""Integration test for two-phase commit system.

This test demonstrates the complete two-phase commit pipeline working
end-to-end with real components and timing.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gunn.adapters.llm import DummyLLMAdapter
from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.types import Intent


class TestIntegrationTwoPhaseCommit:
    """Integration test for complete two-phase commit system."""

    @pytest.fixture
    async def system(self):
        """Set up complete system for integration testing."""
        config = OrchestratorConfig(
            max_agents=5,
            staleness_threshold=1,
            max_queue_depth=10,
            quota_intents_per_minute=20,
            use_in_memory_dedup=True,
            dedup_warmup_minutes=0,
        )

        orchestrator = Orchestrator(config, world_id="integration_test")
        await orchestrator.initialize()

        # Create LLM adapter for cancellation testing
        llm_adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=500.0,
        )

        yield {
            "orchestrator": orchestrator,
            "llm_adapter": llm_adapter,
            "config": config,
        }

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_complete_intent_processing_pipeline(self, system):
        """Test complete intent processing from submission to effect creation."""
        orchestrator = system["orchestrator"]

        # Register agents
        mock_policy = MagicMock(spec=ObservationPolicy)
        _ = await orchestrator.register_agent("agent_a", mock_policy)
        _ = await orchestrator.register_agent("agent_b", mock_policy)

        # Submit intents from both agents
        intent_a = Intent(
            kind="Speak",
            payload={"text": "Hello from Agent A"},
            context_seq=0,
            req_id="req_a_1",
            agent_id="agent_a",
            priority=10,  # High priority
            schema_version="1.0.0",
        )

        intent_b = Intent(
            kind="Move",
            payload={"x": 10, "y": 20},
            context_seq=0,
            req_id="req_b_1",
            agent_id="agent_b",
            priority=5,  # Lower priority
            schema_version="1.0.0",
        )

        # Submit intents
        req_id_a = await orchestrator.submit_intent(intent_a)
        req_id_b = await orchestrator.submit_intent(intent_b)

        assert req_id_a == "req_a_1"
        assert req_id_b == "req_b_1"

        # Wait for processing
        await asyncio.sleep(0.2)

        # Check processing stats
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 2
        assert stats["global_seq"] >= 2

        # Verify both agents have quota tracking
        assert "agent_a" in stats["quota_tracker"]
        assert "agent_b" in stats["quota_tracker"]

    @pytest.mark.asyncio
    async def test_cancellation_integration(self, system):
        """Test integration of cancellation with LLM adapter."""
        orchestrator = system["orchestrator"]
        llm_adapter = system["llm_adapter"]

        # Register agent
        mock_policy = MagicMock(spec=ObservationPolicy)
        await orchestrator.register_agent("agent_llm", mock_policy)

        # Issue cancel token
        cancel_token = orchestrator.issue_cancel_token("agent_llm", "llm_req_1")

        # Start LLM generation
        generation_task = asyncio.create_task(
            llm_adapter.generate_with_timing("Test prompt", cancel_token, max_tokens=50)
        )

        # Submit intent that might trigger cancellation
        intent = Intent(
            kind="Custom",
            payload={"reason": "new_information"},
            context_seq=0,
            req_id="interrupt_req",
            agent_id="agent_llm",
            priority=20,  # Very high priority
            schema_version="1.0.0",
        )

        # Wait a bit, then submit interrupt intent
        await asyncio.sleep(0.05)  # 50ms
        await orchestrator.submit_intent(intent)

        # Cancel the generation
        await asyncio.sleep(0.02)  # 20ms
        cancel_token.cancel("integration_test_cancellation")

        # Generation should be cancelled
        with pytest.raises(asyncio.CancelledError):
            await generation_task

        # Verify cancellation timing
        assert cancel_token.cancelled
        assert cancel_token.reason == "integration_test_cancellation"

    @pytest.mark.asyncio
    async def test_fairness_and_priority_integration(self, system):
        """Test fairness scheduling with priority across multiple agents."""
        orchestrator = system["orchestrator"]

        # Register multiple agents with different weights
        mock_policy = MagicMock(spec=ObservationPolicy)
        agents = []
        for i in range(3):
            agent_id = f"agent_{i}"
            await orchestrator.register_agent(agent_id, mock_policy)
            agents.append(agent_id)

            # Set different weights for fairness testing
            orchestrator._scheduler.set_agent_weight(agent_id, i + 1)

        # Submit intents with different priorities
        intents_submitted = 0
        for agent_id in agents:
            for priority in [20, 10, 0]:  # High, medium, low
                for i in range(2):  # 2 intents per priority per agent
                    intent = Intent(
                        kind="Custom",
                        payload={"agent": agent_id, "priority": priority, "index": i},
                        context_seq=0,
                        req_id=f"{agent_id}_p{priority}_i{i}",
                        agent_id=agent_id,
                        priority=priority,
                        schema_version="1.0.0",
                    )
                    await orchestrator.submit_intent(intent)
                    intents_submitted += 1

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check that all intents were processed fairly
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == intents_submitted

        # Verify fairness across agents
        agent_depths = stats["scheduler"]["agent_queue_depths"]
        for agent_id in agents:
            assert agent_id in agent_depths

    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, system):
        """Test error recovery across the complete system."""
        orchestrator = system["orchestrator"]

        # Register agent
        mock_policy = MagicMock(spec=ObservationPolicy)
        await orchestrator.register_agent("error_agent", mock_policy)

        # Test sequence: valid intent, invalid intent, valid intent
        intents = [
            Intent(
                kind="Speak",
                payload={"text": "Valid message 1"},
                context_seq=0,
                req_id="valid_1",
                agent_id="error_agent",
                priority=0,
                schema_version="1.0.0",
            ),
            # This will cause an error (missing req_id)
            {
                "kind": "Speak",
                "payload": {"text": "Invalid message"},
                "context_seq": 0,
                "agent_id": "error_agent",
                "priority": 0,
                "schema_version": "1.0.0",
            },
            Intent(
                kind="Speak",
                payload={"text": "Valid message 2"},
                context_seq=0,
                req_id="valid_2",
                agent_id="error_agent",
                priority=0,
                schema_version="1.0.0",
            ),
        ]

        results = []
        for intent in intents:
            try:
                req_id = await orchestrator.submit_intent(intent)
                results.append(("success", req_id))
            except Exception as e:
                results.append(("error", type(e).__name__))

        # Should have: success, error, success
        assert len(results) == 3
        assert results[0][0] == "success"
        assert results[0][1] == "valid_1"
        assert results[1][0] == "error"
        assert results[1][1] == "ValueError"
        assert results[2][0] == "success"
        assert results[2][1] == "valid_2"

        # System should still be functional
        await asyncio.sleep(0.1)
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 2  # Only valid intents

    @pytest.mark.asyncio
    async def test_concurrent_processing_integration(self, system):
        """Test concurrent processing with multiple agents and intents."""
        orchestrator = system["orchestrator"]

        # Register multiple agents
        mock_policy = MagicMock(spec=ObservationPolicy)
        num_agents = 3
        agents = []
        for i in range(num_agents):
            agent_id = f"concurrent_agent_{i}"
            await orchestrator.register_agent(agent_id, mock_policy)
            agents.append(agent_id)

        # Submit intents concurrently from all agents
        async def submit_intents_for_agent(agent_id: str, num_intents: int):
            tasks = []
            for i in range(num_intents):
                intent = Intent(
                    kind="Custom",
                    payload={"agent": agent_id, "index": i},
                    context_seq=0,
                    req_id=f"{agent_id}_concurrent_{i}",
                    agent_id=agent_id,
                    priority=0,
                    schema_version="1.0.0",
                )
                task = orchestrator.submit_intent(intent)
                tasks.append(task)

            return await asyncio.gather(*tasks)

        # Submit from all agents concurrently
        agent_tasks = [submit_intents_for_agent(agent_id, 5) for agent_id in agents]

        results = await asyncio.gather(*agent_tasks)

        # Verify all intents were accepted
        total_intents = 0
        for agent_results in results:
            assert len(agent_results) == 5
            total_intents += len(agent_results)

        assert total_intents == num_agents * 5

        # Wait for processing
        await asyncio.sleep(0.3)

        # Verify processing
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == total_intents

    @pytest.mark.asyncio
    async def test_performance_timing_integration(self, system):
        """Test performance timing of the complete system."""
        orchestrator = system["orchestrator"]

        # Register agent
        mock_policy = MagicMock(spec=ObservationPolicy)
        await orchestrator.register_agent("perf_agent", mock_policy)

        # Measure submission timing
        import time

        num_intents = 10
        submission_times = []

        for i in range(num_intents):
            intent = Intent(
                kind="Custom",
                payload={"index": i},
                context_seq=0,
                req_id=f"perf_req_{i}",
                agent_id="perf_agent",
                priority=0,
                schema_version="1.0.0",
            )

            start_time = time.perf_counter()
            await orchestrator.submit_intent(intent)
            end_time = time.perf_counter()

            submission_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Verify submission times are reasonable (< 10ms each)
        avg_submission_time = sum(submission_times) / len(submission_times)
        max_submission_time = max(submission_times)

        assert (
            avg_submission_time < 10.0
        ), f"Average submission time {avg_submission_time:.2f}ms too high"
        assert (
            max_submission_time < 20.0
        ), f"Max submission time {max_submission_time:.2f}ms too high"

        # Wait for processing and verify
        await asyncio.sleep(0.2)
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == num_intents

    @pytest.mark.asyncio
    async def test_deduplication_integration(self, system):
        """Test deduplication working in the complete system."""
        orchestrator = system["orchestrator"]

        # Register agent
        mock_policy = MagicMock(spec=ObservationPolicy)
        await orchestrator.register_agent("dedup_agent", mock_policy)

        # Submit same intent multiple times
        intent = Intent(
            kind="Custom",
            payload={"text": "This is a duplicate"},
            context_seq=0,
            req_id="duplicate_req",
            agent_id="dedup_agent",
            priority=0,
            schema_version="1.0.0",
        )

        # Submit multiple times
        req_ids = []
        for _ in range(5):
            req_id = await orchestrator.submit_intent(intent)
            req_ids.append(req_id)

        # All should return the same req_id
        assert all(req_id == "duplicate_req" for req_id in req_ids)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Should only be enqueued once due to deduplication
        stats = orchestrator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 1

        # Verify dedup store has the entry
        dedup_stats = await orchestrator._dedup_store.get_stats()
        assert dedup_stats["total_entries"] >= 1
