"""Integration tests for concurrent intent submission.

These tests verify that agents can submit multiple intents concurrently,
enabling simultaneous physical actions and communication.
"""

import asyncio

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import Intent


class TestConcurrentIntentSubmission:
    """Integration tests for concurrent intent submission functionality."""

    def setup_agent_permissions(
        self, orchestrator: Orchestrator, agent_ids: list[str]
    ) -> None:
        """Setup permissions and world state for test agents."""
        validator = orchestrator.effect_validator
        if hasattr(validator, "set_agent_permissions"):
            # Grant all necessary permissions for testing
            permissions = {
                "submit_intent",
                "intent:speak",
                "intent:move",
                "intent:attack",
                "intent:interact",
                "intent:custom",
            }
            for agent_id in agent_ids:
                validator.set_agent_permissions(agent_id, permissions)

        # Disable cooldowns for concurrent intent testing
        if hasattr(validator, "set_intent_kind_cooldown"):
            validator.set_intent_kind_cooldown("Move", 0.0)
            validator.set_intent_kind_cooldown("Speak", 0.0)
            validator.set_intent_kind_cooldown("Attack", 0.0)
            validator.set_intent_kind_cooldown("Interact", 0.0)

        # Add agents to world state so they are not "not_in_world"
        for agent_id in agent_ids:
            orchestrator.world_state.entities[agent_id] = {
                "id": agent_id,
                "type": "agent",
                "position": {"x": 0, "y": 0},
                "health": 100,
                "team": "red" if agent_id.startswith("red") else "blue",
            }
            # Also add to spatial_index for Move intent validation
            orchestrator.world_state.spatial_index[agent_id] = (0.0, 0.0, 0.0)

    @pytest.fixture
    def config(self) -> OrchestratorConfig:
        """Create test configuration."""
        return OrchestratorConfig(
            max_agents=10,
            staleness_threshold=1,
            default_priority=5,
            processing_idle_shutdown_ms=0.0,  # Disable idle shutdown for tests
            dedup_warmup_minutes=0,  # Disable warmup period for tests
            use_in_memory_dedup=True,  # Use in-memory dedup store for tests
        )

    @pytest.fixture
    def orchestrator(self, config: OrchestratorConfig, request) -> Orchestrator:
        """Create orchestrator for integration testing."""
        # Use test function name as world_id to ensure test isolation
        world_id = f"test_{request.function.__name__}"
        return Orchestrator(config, world_id=world_id)

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
    async def test_single_intent_backward_compatibility(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that single intent submission still works (backward compatibility)."""
        # Register agent
        alice = await orchestrator.register_agent("alice", observation_policy)
        self.setup_agent_permissions(orchestrator, ["alice"])

        # Submit single intent using the original method
        intent: Intent = {
            "kind": "Speak",
            "payload": {"message": "Hello world!"},
            "context_seq": 0,
            "req_id": "alice_speak_1",
            "agent_id": "alice",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        req_id = await alice.submit_intent(intent)
        assert req_id == "alice_speak_1"

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify event was logged
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1
        assert entries[0].effect["kind"] == "Speak"

    @pytest.mark.asyncio
    async def test_multiple_intents_concurrent_submission(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that multiple intents can be submitted concurrently."""
        # Register agent
        alice = await orchestrator.register_agent("alice", observation_policy)
        self.setup_agent_permissions(orchestrator, ["alice"])

        # Create multiple intents
        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [10.0, 20.0], "reason": "Moving to position"},
            "context_seq": 0,
            "req_id": "alice_move_1",
            "agent_id": "alice",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        speak_intent: Intent = {
            "kind": "Speak",
            "payload": {"message": "I'm moving to the objective!"},
            "context_seq": 0,
            "req_id": "alice_speak_1",
            "agent_id": "alice",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Submit multiple intents concurrently (non-atomic mode)
        req_ids = await alice.submit_intents([move_intent, speak_intent], atomic=False)

        assert len(req_ids) == 2
        assert "alice_move_1" in req_ids
        assert "alice_speak_1" in req_ids

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify both events were logged
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 2

        # Verify both intent types were processed
        effect_kinds = {entry.effect["kind"] for entry in entries}
        assert "Move" in effect_kinds
        assert "Speak" in effect_kinds

    @pytest.mark.asyncio
    async def test_atomic_mode_all_or_nothing(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that atomic mode ensures all-or-nothing submission."""
        # Register agent
        alice = await orchestrator.register_agent("alice", observation_policy)
        self.setup_agent_permissions(orchestrator, ["alice"])

        # Create valid and invalid intents
        valid_intent: Intent = {
            "kind": "Speak",
            "payload": {"message": "This should work"},
            "context_seq": 0,
            "req_id": "alice_speak_valid",
            "agent_id": "alice",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Create an invalid intent (missing required payload field for Move)
        invalid_intent: Intent = {
            "kind": "Move",
            "payload": {},  # Missing 'to' field
            "context_seq": 0,
            "req_id": "alice_move_invalid",
            "agent_id": "alice",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Submit in atomic mode - should fail entirely
        with pytest.raises(Exception):  # Should raise validation error
            await alice.submit_intents(
                [valid_intent, invalid_intent], atomic=True
            )

        # Wait for any potential processing
        await asyncio.sleep(0.1)

        # Verify NO events were logged (atomic rollback)
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 0, "Atomic mode should not log any effects if one fails"

    @pytest.mark.asyncio
    async def test_non_atomic_mode_partial_success(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that non-atomic mode allows partial success."""
        # Register agent
        alice = await orchestrator.register_agent("alice", observation_policy)
        self.setup_agent_permissions(orchestrator, ["alice"])

        # Create valid and invalid intents
        valid_intent: Intent = {
            "kind": "Speak",
            "payload": {"message": "This should work"},
            "context_seq": 0,
            "req_id": "alice_speak_valid",
            "agent_id": "alice",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Create an invalid intent
        invalid_intent: Intent = {
            "kind": "Move",
            "payload": {},  # Missing 'to' field
            "context_seq": 0,
            "req_id": "alice_move_invalid",
            "agent_id": "alice",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Submit in non-atomic mode - should succeed partially
        req_ids = await alice.submit_intents(
            [valid_intent, invalid_intent], atomic=False
        )

        # Only the valid intent should succeed
        assert len(req_ids) == 1
        assert "alice_speak_valid" in req_ids

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify only the valid intent was logged
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1
        assert entries[0].effect["kind"] == "Speak"

    @pytest.mark.asyncio
    async def test_battle_agent_move_and_communicate(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that battle agents can move and speak simultaneously."""
        # Register two agents on different teams
        red_agent = await orchestrator.register_agent("red_1", observation_policy)
        blue_agent = await orchestrator.register_agent("blue_1", observation_policy)

        self.setup_agent_permissions(orchestrator, ["red_1", "blue_1"])

        # Red agent moves and communicates simultaneously
        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [50.0, 50.0], "reason": "Advancing to objective"},
            "context_seq": 0,
            "req_id": "red_1_move",
            "agent_id": "red_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        speak_intent: Intent = {
            "kind": "Speak",
            "payload": {
                "message": "Moving to objective, need backup!",
                "urgency": "high",
            },
            "context_seq": 0,
            "req_id": "red_1_speak",
            "agent_id": "red_1",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        # Submit both intents concurrently
        req_ids = await red_agent.submit_intents(
            [move_intent, speak_intent], atomic=False
        )

        assert len(req_ids) == 2

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify both actions were executed
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 2

        # Verify we have both Move and Speak effects
        effect_kinds = {entry.effect["kind"] for entry in entries}
        assert "Move" in effect_kinds
        assert "Speak" in effect_kinds

        # Verify both effects are from the same agent (use source_id from Effect)
        source_ids = {entry.effect.get("source_id", entry.effect.get("agent_id")) for entry in entries}
        assert source_ids == {"red_1"}

    @pytest.mark.asyncio
    async def test_empty_intent_list(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that submitting an empty intent list is handled gracefully."""
        alice = await orchestrator.register_agent("alice", observation_policy)
        self.setup_agent_permissions(orchestrator, ["alice"])

        # Submit empty list
        req_ids = await alice.submit_intents([], atomic=False)
        assert req_ids == []

        # Verify no events were logged
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_concurrent_submission_from_multiple_agents(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that multiple agents can submit concurrent intents simultaneously."""
        # Register three agents
        alice = await orchestrator.register_agent("alice", observation_policy)
        bob = await orchestrator.register_agent("bob", observation_policy)
        charlie = await orchestrator.register_agent("charlie", observation_policy)

        self.setup_agent_permissions(orchestrator, ["alice", "bob", "charlie"])

        # Each agent submits multiple intents concurrently
        # Use different target positions to avoid collision detection failures
        agent_positions = {
            "alice": [10.0, 10.0],
            "bob": [20.0, 20.0],
            "charlie": [30.0, 30.0],
        }

        async def agent_submit(agent_handle, agent_id: str) -> list[str]:
            move = {
                "kind": "Move",
                "payload": {"to": agent_positions[agent_id], "reason": "Moving"},
                "context_seq": 0,
                "req_id": f"{agent_id}_move",
                "agent_id": agent_id,
                "priority": 1,
                "schema_version": "1.0.0",
            }
            speak = {
                "kind": "Speak",
                "payload": {"message": f"{agent_id} is moving"},
                "context_seq": 0,
                "req_id": f"{agent_id}_speak",
                "agent_id": agent_id,
                "priority": 1,
                "schema_version": "1.0.0",
            }
            return await agent_handle.submit_intents([move, speak], atomic=False)

        # Submit from all agents concurrently
        results = await asyncio.gather(
            agent_submit(alice, "alice"),
            agent_submit(bob, "bob"),
            agent_submit(charlie, "charlie"),
        )

        # Verify all submissions succeeded
        assert len(results) == 3
        for result in results:
            assert len(result) == 2

        # Wait for processing
        await asyncio.sleep(0.3)

        # Verify all 6 events were logged (3 agents × 2 intents each)
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 6

        # Verify we have the right distribution of effect types
        effect_kinds = [entry.effect["kind"] for entry in entries]
        assert effect_kinds.count("Move") == 3
        assert effect_kinds.count("Speak") == 3

        # Verify all three agents submitted their intents
        source_ids = {entry.effect.get("source_id", entry.effect.get("agent_id")) for entry in entries}
        assert source_ids == {"alice", "bob", "charlie"}
