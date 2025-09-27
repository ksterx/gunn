"""Integration tests for Orchestrator functionality.

These tests demonstrate end-to-end orchestrator functionality including
agent registration, intent processing, and deterministic ordering.
"""

import asyncio
from typing import cast

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import Intent


class TestOrchestratorIntegration:
    """Integration tests for complete orchestrator workflows."""

    @pytest.fixture
    def config(self) -> OrchestratorConfig:
        """Create test configuration."""
        return OrchestratorConfig(
            max_agents=10,
            staleness_threshold=1000,  # High threshold for existing tests
            default_priority=5,
        )

    @pytest.fixture
    async def orchestrator(self, config: OrchestratorConfig) -> Orchestrator:
        """Create orchestrator for integration testing."""
        orchestrator = Orchestrator(config, world_id="integration_test")
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for testing."""
        return DefaultObservationPolicy(
            PolicyConfig(
                distance_limit=100.0,
                relationship_filter=["friend", "ally"],
                max_patch_ops=25,
            )
        )

    @pytest.mark.asyncio
    async def test_multi_agent_conversation_workflow(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test a complete multi-agent conversation workflow."""
        # Register three agents
        alice = await orchestrator.register_agent("alice", observation_policy)
        bob = await orchestrator.register_agent("bob", observation_policy)
        charlie = await orchestrator.register_agent("charlie", observation_policy)

        # Verify registration
        assert orchestrator.get_agent_count() == 3
        assert alice.agent_id == "alice"
        assert bob.agent_id == "bob"
        assert charlie.agent_id == "charlie"

        # Alice speaks first
        alice_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello everyone!"},
            "context_seq": 0,
            "req_id": "alice_msg_1",
            "agent_id": "alice",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        req_id_1 = await alice.submit_intent(alice_intent)
        assert req_id_1 == "alice_msg_1"

        # Bob responds
        bob_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hi Alice! How are you?"},
            "context_seq": 1,
            "req_id": "bob_msg_1",
            "agent_id": "bob",
            "priority": 2,
            "schema_version": "1.0.0",
        }

        req_id_2 = await bob.submit_intent(bob_intent)
        assert req_id_2 == "bob_msg_1"

        # Charlie joins the conversation
        charlie_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hey folks! What's going on?"},
            "context_seq": 2,
            "req_id": "charlie_msg_1",
            "agent_id": "charlie",
            "priority": 0,  # Lower priority
            "schema_version": "1.0.0",
        }

        req_id_3 = await charlie.submit_intent(charlie_intent)
        assert req_id_3 == "charlie_msg_1"

        # Verify all events were logged in correct order
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 3

        # Check ordering by global_seq
        assert entries[0].effect["source_id"] == "alice"
        assert entries[1].effect["source_id"] == "bob"
        assert entries[2].effect["source_id"] == "charlie"

        # Verify global_seq is monotonic
        assert entries[0].effect["global_seq"] == 1
        assert entries[1].effect["global_seq"] == 2
        assert entries[2].effect["global_seq"] == 3

        # Verify content
        assert entries[0].effect["payload"]["text"] == "Hello everyone!"
        assert entries[1].effect["payload"]["text"] == "Hi Alice! How are you?"
        assert entries[2].effect["payload"]["text"] == "Hey folks! What's going on?"

    @pytest.mark.asyncio
    async def test_cancellation_workflow(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test cancellation token workflow."""
        # Register agent
        agent = await orchestrator.register_agent("test_agent", observation_policy)

        # Issue cancel token
        token = orchestrator.issue_cancel_token("test_agent", "long_task_1")
        assert not token.cancelled

        # Simulate some work
        await asyncio.sleep(0.01)

        # Cancel the token
        await agent.cancel("long_task_1")
        assert token.cancelled
        assert token.reason == "user_requested"  # type: ignore[unreachable]

        # Test staleness-based cancellation
        agent.view_seq = 5
        stale_token = orchestrator.issue_cancel_token("test_agent", "stale_task_1")

        # Trigger staleness check (new_seq=2000, current=5, staleness=1995, threshold=1000)
        was_cancelled = await orchestrator.cancel_if_stale(
            "test_agent", "stale_task_1", 2000
        )
        assert was_cancelled
        assert stale_token.cancelled
        assert "stale_due_to_seq_gap" in stale_token.reason

    @pytest.mark.asyncio
    async def test_idempotency_workflow(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test idempotency across multiple intent submissions."""
        # Register agent
        agent = await orchestrator.register_agent("test_agent", observation_policy)

        # Create intent
        intent: Intent = {
            "kind": "Move",
            "payload": {"x": 10, "y": 20},
            "context_seq": 0,
            "req_id": "move_1",
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Submit same intent multiple times
        req_ids = []
        for _ in range(5):
            req_id = await agent.submit_intent(intent)
            req_ids.append(req_id)

        # All should return same req_id
        assert all(req_id == "move_1" for req_id in req_ids)

        # Should only have one effect in log
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1
        assert entries[0].effect["kind"] == "Move"
        assert entries[0].effect["payload"]["x"] == 10
        assert entries[0].effect["payload"]["y"] == 20
        assert entries[0].effect["payload"]["priority"] == 0  # Priority should be added

    @pytest.mark.asyncio
    async def test_deterministic_ordering_under_load(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test deterministic ordering under concurrent load."""
        # Register multiple agents
        agents = []
        for i in range(5):
            agent = await orchestrator.register_agent(f"agent_{i}", observation_policy)
            agents.append(agent)

        # Create intents with different priorities
        intents = []
        for i, agent in enumerate(agents):
            intent: Intent = {
                "kind": "Custom",
                "payload": {"action_id": i},
                "context_seq": 0,
                "req_id": f"action_{i}",
                "agent_id": agent.agent_id,
                "priority": i % 3,  # Priorities 0, 1, 2, 0, 1
                "schema_version": "1.0.0",
            }
            intents.append((agent, intent))

        # Submit all intents concurrently
        tasks = [agent.submit_intent(intent) for agent, intent in intents]
        await asyncio.gather(*tasks)

        # Verify all effects were logged
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 5

        # Verify deterministic ordering by global_seq
        for i, entry in enumerate(entries):
            assert entry.effect["global_seq"] == i + 1

        # Verify each effect has required ordering fields
        for entry in entries:
            effect = entry.effect
            assert "sim_time" in effect
            assert "global_seq" in effect
            assert "source_id" in effect
            assert "uuid" in effect
            assert isinstance(effect["sim_time"], float)
            assert isinstance(effect["global_seq"], int)

    @pytest.mark.asyncio
    async def test_sim_time_authority_workflow(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test sim_time authority management workflow."""
        # Initially no authority
        assert orchestrator._sim_time_authority == "none"

        # Set Unity as authority
        orchestrator.set_sim_time_authority("unity")
        assert orchestrator._sim_time_authority == "unity"

        # Register agent and submit intent
        agent = await orchestrator.register_agent("test_agent", observation_policy)

        intent: Intent = {
            "kind": "Custom",
            "payload": {"data": "test"},
            "context_seq": 0,
            "req_id": "test_1",
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        await agent.submit_intent(intent)

        # Verify effect has sim_time
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1
        assert "sim_time" in entries[0].effect
        assert isinstance(entries[0].effect["sim_time"], float)

        # Change authority
        orchestrator.set_sim_time_authority("unreal")
        assert orchestrator._sim_time_authority == "unreal"

        # Reset to none
        orchestrator.set_sim_time_authority("none")
        assert orchestrator._sim_time_authority == "none"

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown_workflow(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test complete orchestrator shutdown workflow."""
        # Set up complex state
        agents = []
        for i in range(3):
            agent = await orchestrator.register_agent(f"agent_{i}", observation_policy)
            agents.append(agent)

        # Issue some cancel tokens
        tokens = []
        for i, agent in enumerate(agents):
            token = orchestrator.issue_cancel_token(agent.agent_id, f"task_{i}")
            tokens.append(token)

        # Submit some intents
        for i, agent in enumerate(agents):
            intent: Intent = {
                "kind": "Custom",
                "payload": {"index": i},
                "context_seq": 0,
                "req_id": f"req_{i}",
                "agent_id": agent.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
            await agent.submit_intent(intent)

        # Verify state before shutdown
        assert orchestrator.get_agent_count() == 3
        assert len(orchestrator._cancel_tokens) == 3
        assert orchestrator.event_log.get_entry_count() == 3

        # Check dedup store has entries
        stats = await orchestrator.dedup_store.get_stats()
        assert stats["total_entries"] == 3

        # Shutdown
        await orchestrator.shutdown()

        # Verify cleanup (but event log should remain)
        assert orchestrator.get_agent_count() == 0
        assert len(orchestrator._cancel_tokens) == 0
        # Note: dedup_store entries may persist after shutdown for TTL-based cleanup
        assert orchestrator.event_log.get_entry_count() == 3  # Log persists

    @pytest.mark.asyncio
    async def test_error_handling_workflow(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test error handling in various scenarios."""
        # Test agent registration errors
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            await orchestrator.register_agent("", observation_policy)

        # Register valid agent
        agent = await orchestrator.register_agent("test_agent", observation_policy)

        # Test duplicate registration
        with pytest.raises(ValueError, match="already registered"):
            await orchestrator.register_agent("test_agent", observation_policy)

        # Test invalid intent submission
        from gunn.schemas.types import Intent as IntentType

        with pytest.raises(ValueError, match="Intent must have 'req_id' field"):
            invalid_intent = cast(
                IntentType,
                {
                    "kind": "Custom",
                    "payload": {},
                    "agent_id": "test_agent",
                    "context_seq": 0,
                    "priority": 0,
                    "schema_version": "1.0.0",
                    # Missing req_id
                },
            )
            await agent.submit_intent(invalid_intent)

        # Test invalid effect broadcasting
        from gunn.schemas.types import EffectDraft as EffectDraftType

        with pytest.raises(ValueError, match="EffectDraft must have 'kind' field"):
            invalid_effect = cast(
                EffectDraftType,
                {
                    "payload": {},
                    "source_id": "test",
                    "schema_version": "1.0.0",
                    # Missing kind
                },
            )
            await orchestrator.broadcast_event(invalid_effect)

        # Test unregistered agent operations
        unregistered_handle = orchestrator.agent_handles.get("nonexistent")
        assert unregistered_handle is None


if __name__ == "__main__":
    pytest.main([__file__])
