"""Unit tests for Orchestrator two-phase commit functionality."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import DefaultObservationPolicy
from gunn.schemas.types import Intent
from gunn.storage.dedup_store import DedupStore
from gunn.utils.errors import StaleContextError, ValidationError


class TestOrchestratorTwoPhaseCommit:
    """Test suite for Orchestrator two-phase commit functionality."""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator with test configuration."""
        config = OrchestratorConfig(
            staleness_threshold=1000,  # High threshold for most tests
            dedup_ttl_minutes=1,  # Short TTL for testing
            warmup_ttl_minutes=0.01,  # Very short warmup
        )

        # Use in-memory database for testing
        dedup_store = DedupStore(
            db_path=":memory:",
            dedup_ttl_minutes=1,
            warmup_ttl_minutes=0.01,
        )

        orchestrator = Orchestrator(
            config=config,
            world_id="test_world",
            dedup_store=dedup_store,
        )

        await orchestrator.initialize()

        yield orchestrator

        await orchestrator.shutdown()

    @pytest.fixture
    def basic_policy(self):
        """Create basic observation policy for testing."""
        from gunn.policies.observation import PolicyConfig

        config = PolicyConfig()
        return DefaultObservationPolicy(config)

    @pytest.fixture
    def sample_intent(self):
        """Create sample intent for testing."""
        return Intent(
            kind="Speak",
            payload={"text": "Hello, world!"},
            context_seq=0,
            req_id="test_req_1",
            agent_id="agent_1",
            priority=1,
            schema_version="1.0.0",
        )

    @pytest.mark.asyncio
    async def test_initialization_required(self):
        """Test that orchestrator must be initialized before use."""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config=config)

        intent = Intent(
            kind="Speak",
            payload={"text": "test"},
            context_seq=0,
            req_id="req_1",
            agent_id="agent_1",
            priority=1,
            schema_version="1.0.0",
        )

        with pytest.raises(RuntimeError, match="Orchestrator not initialized"):
            await orchestrator.submit_intent(intent)

    @pytest.mark.asyncio
    async def test_basic_intent_processing(
        self, orchestrator, basic_policy, sample_intent
    ):
        """Test basic intent processing through two-phase commit."""
        # Register agent
        await orchestrator.register_agent("agent_1", basic_policy)

        # Submit intent
        req_id = await orchestrator.submit_intent(sample_intent)
        assert req_id == "test_req_1"

        # Verify global sequence was incremented
        assert orchestrator.get_latest_seq() == 1

    @pytest.mark.asyncio
    async def test_idempotency_checking(
        self, orchestrator, basic_policy, sample_intent
    ):
        """Test idempotency checking with persistent store."""
        # Register agent
        await orchestrator.register_agent("agent_1", basic_policy)

        # Submit intent first time
        req_id1 = await orchestrator.submit_intent(sample_intent)
        assert req_id1 == "test_req_1"
        initial_seq = orchestrator.get_latest_seq()

        # Submit same intent again - should be idempotent
        req_id2 = await orchestrator.submit_intent(sample_intent)
        assert req_id2 == "test_req_1"

        # Global sequence should not have incremented
        assert orchestrator.get_latest_seq() == initial_seq

    @pytest.mark.asyncio
    async def test_staleness_detection(self, basic_policy):
        """Test staleness detection in two-phase commit."""
        # Create orchestrator with low staleness threshold for this test
        config = OrchestratorConfig(
            staleness_threshold=2,  # Low threshold for staleness testing
            dedup_ttl_minutes=1,
            warmup_ttl_minutes=0.01,
        )
        dedup_store = DedupStore(
            db_path=":memory:",
            dedup_ttl_minutes=1,
            warmup_ttl_minutes=0.01,
        )

        test_orchestrator = Orchestrator(
            config=config,
            world_id="staleness_test",
            dedup_store=dedup_store,
        )

        await test_orchestrator.initialize()

        try:
            # Register agent
            agent_handle = await test_orchestrator.register_agent(
                "agent_1", basic_policy
            )

            # Create some events to advance global sequence
            for i in range(5):
                intent = Intent(
                    kind="Speak",
                    payload={"text": f"Message {i}"},
                    context_seq=i,
                    req_id=f"req_{i}",
                    agent_id="agent_1",
                    priority=1,
                    schema_version="1.0.0",
                )
                await test_orchestrator.submit_intent(intent)

            # Now submit intent with stale context
            stale_intent = Intent(
                kind="Speak",
                payload={"text": "Stale message"},
                context_seq=1,  # Much older than current global_seq
                req_id="stale_req",
                agent_id="agent_1",
                priority=1,
                schema_version="1.0.0",
            )

            with pytest.raises(StaleContextError) as exc_info:
                await test_orchestrator.submit_intent(stale_intent)

            error = exc_info.value
            assert error.req_id == "stale_req"
            assert error.expected_seq == 1
            assert error.actual_seq > 1
        finally:
            await test_orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_validation_pipeline(self, orchestrator, basic_policy):
        """Test validation pipeline in two-phase commit."""

        # Create custom validator that rejects certain intents
        class TestValidator:
            def validate_intent(self, intent, world_state):
                return intent["payload"].get("text") != "INVALID"

        # Create orchestrator with custom validator
        config = OrchestratorConfig(warmup_ttl_minutes=0.01)
        dedup_store = DedupStore(db_path=":memory:", warmup_ttl_minutes=0.01)

        test_orchestrator = Orchestrator(
            config=config,
            world_id="test_world",
            effect_validator=TestValidator(),
            dedup_store=dedup_store,
        )

        await test_orchestrator.initialize()

        try:
            # Register agent
            await test_orchestrator.register_agent("agent_1", basic_policy)

            # Valid intent should pass
            valid_intent = Intent(
                kind="Speak",
                payload={"text": "Valid message"},
                context_seq=0,
                req_id="valid_req",
                agent_id="agent_1",
                priority=1,
                schema_version="1.0.0",
            )

            req_id = await test_orchestrator.submit_intent(valid_intent)
            assert req_id == "valid_req"

            # Invalid intent should fail validation (with correct context_seq)
            invalid_intent = Intent(
                kind="Speak",
                payload={"text": "INVALID"},
                context_seq=1,  # Updated context_seq after the first intent
                req_id="invalid_req",
                agent_id="agent_1",
                priority=1,
                schema_version="1.0.0",
            )

            with pytest.raises(ValidationError) as exc_info:
                await test_orchestrator.submit_intent(invalid_intent)

            error = exc_info.value
            assert error.intent["req_id"] == "invalid_req"

        finally:
            await test_orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_priority_completion(self, orchestrator, basic_policy):
        """Test priority completion for EffectDraft."""
        # Register agent
        await orchestrator.register_agent("agent_1", basic_policy)

        # Submit intent without priority
        intent_no_priority = Intent(
            kind="Speak",
            payload={"text": "No priority"},
            context_seq=0,
            req_id="no_priority_req",
            agent_id="agent_1",
            priority=5,  # This should be used for priority completion
            schema_version="1.0.0",
        )

        await orchestrator.submit_intent(intent_no_priority)

        # Verify effect was created (check through event log)
        entries = orchestrator.event_log.get_entries_since(0)
        assert len(entries) == 1

        effect = entries[0].effect
        assert effect["payload"]["priority"] == 5

    @pytest.mark.asyncio
    async def test_concurrent_intent_processing(self, orchestrator, basic_policy):
        """Test concurrent intent processing with two-phase commit."""
        # Register agent
        await orchestrator.register_agent("agent_1", basic_policy)

        # Submit multiple concurrent intents
        async def submit_intent(i):
            intent = Intent(
                kind="Speak",
                payload={"text": f"Message {i}"},
                context_seq=0,
                req_id=f"concurrent_req_{i}",
                agent_id="agent_1",
                priority=1,
                schema_version="1.0.0",
            )
            return await orchestrator.submit_intent(intent)

        # Run 10 concurrent submissions
        tasks = [submit_intent(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed with unique req_ids
        assert len(results) == 10
        assert len(set(results)) == 10  # All unique

        # Global sequence should have advanced by 10
        assert orchestrator.get_latest_seq() == 10

    @pytest.mark.asyncio
    async def test_ttl_cleanup_integration(self, basic_policy):
        """Test TTL cleanup integration with two-phase commit."""
        # Create orchestrator with very short TTL for testing
        config = OrchestratorConfig(
            staleness_threshold=1000,
            dedup_ttl_minutes=0.01,  # Very short TTL: 0.6 seconds
            warmup_ttl_minutes=0.001,  # Very short warmup: 0.06 seconds
        )

        dedup_store = DedupStore(
            db_path=":memory:",
            dedup_ttl_minutes=0.01,  # 0.6 seconds
            warmup_ttl_minutes=0.001,  # 0.06 seconds
        )

        orchestrator = Orchestrator(
            config=config,
            world_id="test_world",
            dedup_store=dedup_store,
        )

        await orchestrator.initialize()

        try:
            # Register agent
            await orchestrator.register_agent("agent_1", basic_policy)

            # Submit intent
            intent = Intent(
                kind="Speak",
                payload={"text": "TTL test"},
                context_seq=0,
                req_id="ttl_req",
                agent_id="agent_1",
                priority=1,
                schema_version="1.0.0",
            )

            await orchestrator.submit_intent(intent)

            # Should be idempotent immediately
            req_id = await orchestrator.submit_intent(intent)
            assert req_id == "ttl_req"

            # Wait for TTL to expire
            await asyncio.sleep(0.1)  # Wait for warmup to end
            await asyncio.sleep(1.0)  # Wait for TTL expiration

            # Should be able to submit again after TTL expiration
            await orchestrator.submit_intent(intent)

            # Global sequence should have advanced
            assert orchestrator.get_latest_seq() == 2

        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_persistent_deduplication(self):
        """Test persistent deduplication across orchestrator restarts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            # Create first orchestrator
            config1 = OrchestratorConfig(warmup_ttl_minutes=0.01)
            dedup_store1 = DedupStore(db_path=db_path, warmup_ttl_minutes=0.01)

            orchestrator1 = Orchestrator(
                config=config1,
                world_id="test_world",
                dedup_store=dedup_store1,
            )

            await orchestrator1.initialize()

            # Register agent and submit intent
            from gunn.policies.observation import PolicyConfig

            policy = DefaultObservationPolicy(PolicyConfig())
            await orchestrator1.register_agent("agent_1", policy)

            intent = Intent(
                kind="Speak",
                payload={"text": "Persistent test"},
                context_seq=0,
                req_id="persistent_req",
                agent_id="agent_1",
                priority=1,
                schema_version="1.0.0",
            )

            await orchestrator1.submit_intent(intent)
            await orchestrator1.shutdown()

            # Create second orchestrator with same database
            config2 = OrchestratorConfig(warmup_ttl_minutes=0.01)
            dedup_store2 = DedupStore(db_path=db_path, warmup_ttl_minutes=0.01)

            orchestrator2 = Orchestrator(
                config=config2,
                world_id="test_world",
                dedup_store=dedup_store2,
            )

            await orchestrator2.initialize()
            await orchestrator2.register_agent("agent_1", policy)

            # Wait for warmup to end
            await asyncio.sleep(0.1)

            # Same intent should be idempotent
            req_id = await orchestrator2.submit_intent(intent)
            assert req_id == "persistent_req"

            # Global sequence should not have advanced
            assert orchestrator2.get_latest_seq() == 0

            await orchestrator2.shutdown()

    @pytest.mark.asyncio
    async def test_fairness_round_robin(self, orchestrator, basic_policy):
        """Test fairness using round-robin ordering."""
        # Register multiple agents
        await orchestrator.register_agent("agent_1", basic_policy)
        await orchestrator.register_agent("agent_2", basic_policy)
        await orchestrator.register_agent("agent_3", basic_policy)

        # Submit intents from different agents
        intents = []
        for agent_num in [1, 2, 3, 1, 2, 3]:
            intent = Intent(
                kind="Speak",
                payload={"text": f"Message from agent {agent_num}"},
                context_seq=0,
                req_id=f"fairness_req_{agent_num}_{len(intents)}",
                agent_id=f"agent_{agent_num}",
                priority=1,
                schema_version="1.0.0",
            )
            intents.append(intent)

        # Submit all intents
        for intent in intents:
            await orchestrator.submit_intent(intent)

        # Verify round-robin order is maintained
        assert len(orchestrator._agent_processing_order) == 3
        # Order should be based on last processed
        expected_order = ["agent_1", "agent_2", "agent_3"]
        assert orchestrator._agent_processing_order == expected_order

    @pytest.mark.asyncio
    async def test_intent_structure_validation(self, orchestrator):
        """Test validation of intent structure."""
        # Missing req_id
        with pytest.raises(ValueError, match="Intent must have 'req_id' field"):
            await orchestrator.submit_intent(
                {
                    "kind": "Speak",
                    "payload": {"text": "test"},
                    "agent_id": "agent_1",
                    "priority": 1,
                    "schema_version": "1.0.0",
                }
            )

        # Missing agent_id
        with pytest.raises(ValueError, match="Intent must have 'agent_id' field"):
            await orchestrator.submit_intent(
                {
                    "kind": "Speak",
                    "payload": {"text": "test"},
                    "req_id": "req_1",
                    "priority": 1,
                    "schema_version": "1.0.0",
                }
            )

        # Missing kind
        with pytest.raises(ValueError, match="Intent must have 'kind' field"):
            await orchestrator.submit_intent(
                {
                    "payload": {"text": "test"},
                    "req_id": "req_1",
                    "agent_id": "agent_1",
                    "priority": 1,
                    "schema_version": "1.0.0",
                }
            )

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, orchestrator, basic_policy):
        """Test conflict resolution in two-phase commit."""
        # Register agent
        await orchestrator.register_agent("agent_1", basic_policy)

        # Submit multiple intents rapidly to test conflict resolution
        async def rapid_submit(req_suffix):
            intent = Intent(
                kind="Speak",
                payload={"text": f"Rapid message {req_suffix}"},
                context_seq=0,
                req_id=f"rapid_req_{req_suffix}",
                agent_id="agent_1",
                priority=1,
                schema_version="1.0.0",
            )
            return await orchestrator.submit_intent(intent)

        # Submit 5 intents concurrently
        tasks = [rapid_submit(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        assert all(req_id.startswith("rapid_req_") for req_id in results)

        # Global sequence should have advanced by 5
        assert orchestrator.get_latest_seq() == 5
