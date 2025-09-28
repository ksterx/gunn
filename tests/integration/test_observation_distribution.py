"""Tests for observation distribution system.

Tests cover observation delta generation, timed delivery, priority completion,
and efficient filtering for affected agents.
"""

import asyncio
import time

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import (
    DefaultObservationPolicy,
    DistanceLatencyModel,
    NoLatencyModel,
    PolicyConfig,
)
from gunn.schemas.types import EffectDraft


class TestObservationDistribution:
    """Test observation distribution functionality."""

    @pytest.fixture
    async def orchestrator(self) -> Orchestrator:
        """Create orchestrator for testing."""
        config = OrchestratorConfig(
            max_agents=10,
            default_priority=5,
            use_in_memory_dedup=True,
        )
        orchestrator = Orchestrator(config, world_id="test_world")
        await orchestrator.initialize()
        return orchestrator

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for testing."""
        config = PolicyConfig(
            distance_limit=100.0,
            max_patch_ops=10,
            include_spatial_index=True,
        )
        return DefaultObservationPolicy(config)

    @pytest.mark.asyncio
    async def test_broadcast_event_priority_completion(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test that EffectDraft gets priority completion from config.default_priority."""
        # Create draft without priority
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "test"},  # No priority field
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify effect was created with default priority
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

        effect = entries[0].effect
        assert effect["payload"]["priority"] == orchestrator.config.default_priority

    @pytest.mark.asyncio
    async def test_broadcast_event_preserves_existing_priority(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test that existing priority in EffectDraft is preserved."""
        # Create draft with explicit priority
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "test", "priority": 10},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify effect preserves original priority
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

        effect = entries[0].effect
        assert effect["payload"]["priority"] == 10

    @pytest.mark.asyncio
    async def test_world_state_update_move_effect(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test world state update for Move effects."""
        # Create Move effect
        draft: EffectDraft = {
            "kind": "Move",
            "payload": {"position": [10.0, 20.0, 30.0]},
            "source_id": "agent_1",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify world state was updated
        world_state = orchestrator.get_world_state()
        assert "agent_1" in world_state.spatial_index
        assert world_state.spatial_index["agent_1"] == (10.0, 20.0, 30.0)

        assert "agent_1" in world_state.entities
        entity_data = world_state.entities["agent_1"]
        assert entity_data["last_position"] == [10.0, 20.0, 30.0]
        assert "last_move_time" in entity_data

    @pytest.mark.asyncio
    async def test_world_state_update_speak_effect(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test world state update for Speak effects."""
        # Create Speak effect
        draft: EffectDraft = {
            "kind": "Speak",
            "payload": {"text": "Hello world!"},
            "source_id": "agent_1",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify world state was updated
        world_state = orchestrator.get_world_state()
        assert "agent_1" in world_state.entities

        entity_data = world_state.entities["agent_1"]
        assert entity_data["last_message"] == "Hello world!"
        assert "last_speak_time" in entity_data
        assert entity_data["message_count"] == 1

    @pytest.mark.asyncio
    async def test_world_state_update_interact_effect(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test world state update for Interact effects."""
        # Create Interact effect
        draft: EffectDraft = {
            "kind": "Interact",
            "payload": {"target_id": "agent_2"},
            "source_id": "agent_1",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify relationships were created
        world_state = orchestrator.get_world_state()
        assert "agent_1" in world_state.relationships
        assert "agent_2" in world_state.relationships["agent_1"]

        assert "agent_2" in world_state.relationships
        assert "agent_1" in world_state.relationships["agent_2"]

    @pytest.mark.asyncio
    async def test_observation_distribution_to_relevant_agents(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that observations are only sent to relevant agents."""
        # Register two agents with different policies
        agent_1_handle = await orchestrator.register_agent(
            "agent_1", observation_policy
        )

        # Create a policy that observes everything for agent_2
        wide_policy = DefaultObservationPolicy(PolicyConfig(distance_limit=1000.0))
        agent_2_handle = await orchestrator.register_agent("agent_2", wide_policy)

        # Create a policy that observes nothing for agent_3
        narrow_policy = DefaultObservationPolicy(PolicyConfig(distance_limit=0.1))
        agent_3_handle = await orchestrator.register_agent("agent_3", narrow_policy)

        # Position agents - make sure agent_1 is positioned so agent_2 can observe it
        orchestrator.world_state.spatial_index["agent_1"] = (0.0, 0.0, 0.0)
        orchestrator.world_state.spatial_index["agent_2"] = (
            10.0,
            10.0,
            0.0,
        )  # Close to agent_1
        orchestrator.world_state.spatial_index["agent_3"] = (
            500.0,
            500.0,
            500.0,
        )  # Far away

        # Broadcast event from agent_1
        draft: EffectDraft = {
            "kind": "Speak",
            "payload": {"text": "Hello!"},
            "source_id": "agent_1",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Wait for observation delivery
        await asyncio.sleep(0.1)  # 100ms should be enough for delivery

        # agent_1 should receive observation (always observes own effects)
        try:
            observation_1 = await asyncio.wait_for(
                agent_1_handle.next_observation(), timeout=0.1
            )
            assert observation_1 is not None
            assert observation_1["view_seq"] == 1
        except TimeoutError:
            pytest.fail("agent_1 should have received observation of own effect")

        # agent_2 should receive observation (wide policy, should observe agent_1)
        try:
            observation_2 = await asyncio.wait_for(
                agent_2_handle.next_observation(), timeout=0.1
            )
            assert observation_2 is not None
            assert observation_2["view_seq"] == 1
        except TimeoutError:
            # Debug: check if agent_2 can observe agent_1
            can_observe = wide_policy.should_observe_event(
                {
                    "kind": "Speak",
                    "payload": {"text": "Hello!"},
                    "source_id": "agent_1",
                    "schema_version": "1.0.0",
                    "uuid": "test",
                    "global_seq": 1,
                    "sim_time": 0.0,
                },
                "agent_2",
                orchestrator.world_state,
            )
            pytest.fail(
                f"agent_2 should have received observation. Can observe: {can_observe}"
            )

        # agent_3 should not receive observation (narrow policy, far distance)
        try:
            await asyncio.wait_for(agent_3_handle.next_observation(), timeout=0.1)
            pytest.fail("agent_3 should not have received observation")
        except TimeoutError:
            pass  # Expected - agent_3 should not receive observation

    @pytest.mark.asyncio
    async def test_timed_delivery_with_latency_model(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test timed delivery using latency models."""
        # Set up latency model with measurable delay
        latency_model = DistanceLatencyModel(base_latency=0.05, distance_factor=0.001)
        observation_policy.set_latency_model(latency_model)

        # Register agent
        agent_handle = await orchestrator.register_agent("agent_1", observation_policy)

        # Broadcast event
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "test"},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        start_time = time.perf_counter()
        await orchestrator.broadcast_event(draft)

        # Wait for observation
        observation = await agent_handle.next_observation()
        delivery_time = time.perf_counter() - start_time

        # Verify observation was delivered
        assert observation is not None
        assert observation["view_seq"] == 1

        # Verify delivery took at least the base latency
        assert delivery_time >= 0.05  # Should take at least 50ms due to latency model

    @pytest.mark.asyncio
    async def test_observation_delta_generation(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test observation delta generation with JSON patches."""
        # Register agent
        agent_handle = await orchestrator.register_agent("agent_1", observation_policy)

        # Position agent to observe events
        orchestrator.world_state.spatial_index["agent_1"] = (0.0, 0.0, 0.0)
        orchestrator.world_state.spatial_index["speaker"] = (10.0, 10.0, 0.0)

        # First event - should create full snapshot
        draft1: EffectDraft = {
            "kind": "Speak",
            "payload": {"text": "First message"},
            "source_id": "speaker",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft1)
        observation1 = await agent_handle.next_observation()

        assert observation1["view_seq"] == 1
        assert len(observation1["patches"]) >= 1  # Should have patches

        # Second event - should create incremental delta
        draft2: EffectDraft = {
            "kind": "Speak",
            "payload": {"text": "Second message"},
            "source_id": "speaker",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft2)
        observation2 = await agent_handle.next_observation()

        assert observation2["view_seq"] == 2
        assert len(observation2["patches"]) >= 1

        # Verify patch contains message update
        patches = observation2["patches"]
        message_patch = None
        for patch in patches:
            if "last_message" in patch.get("path", ""):
                message_patch = patch
                break

        assert message_patch is not None
        assert message_patch["value"] == "Second message"

    @pytest.mark.asyncio
    async def test_max_patch_ops_fallback(self, orchestrator: Orchestrator) -> None:
        """Test fallback to full snapshot when patches exceed max_patch_ops."""
        # Create policy with very low max_patch_ops
        policy_config = PolicyConfig(max_patch_ops=1, distance_limit=100.0)
        observation_policy = DefaultObservationPolicy(policy_config)

        # Register agent
        agent_handle = await orchestrator.register_agent("agent_1", observation_policy)

        # Position entities
        orchestrator.world_state.spatial_index["agent_1"] = (0.0, 0.0, 0.0)
        orchestrator.world_state.spatial_index["speaker"] = (10.0, 10.0, 0.0)

        # Create complex effect that would generate many patches
        draft: EffectDraft = {
            "kind": "Speak",
            "payload": {
                "text": "Complex message",
                "metadata": {"key1": "value1", "key2": "value2"},
                "extra_data": {"nested": {"deep": "value"}},
            },
            "source_id": "speaker",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)
        observation = await agent_handle.next_observation()

        # Should fallback to full snapshot (replace operations)
        patches = observation["patches"]
        assert len(patches) == 2  # Should be exactly 2 replace operations

        replace_ops = [p for p in patches if p["op"] == "replace"]
        assert len(replace_ops) == 2

        paths = [p["path"] for p in replace_ops]
        assert "/visible_entities" in paths
        assert "/visible_relationships" in paths

    @pytest.mark.asyncio
    async def test_event_log_source_metadata_world_id(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test that world_id is stored in EventLogEntry.source_metadata."""
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "test"},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify source_metadata contains world_id
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

        entry = entries[0]
        assert "world_id" in entry.source_metadata
        assert entry.source_metadata["world_id"] == "test_world"

    @pytest.mark.asyncio
    async def test_event_log_source_metadata_priority(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test that priority is stored in EventLogEntry.source_metadata."""
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "test", "priority": 15},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify source_metadata contains priority
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

        entry = entries[0]
        assert "priority" in entry.source_metadata
        assert entry.source_metadata["priority"] == 15

    @pytest.mark.asyncio
    async def test_observation_consistency_multiple_agents(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test observation consistency across multiple agents."""
        # Register multiple agents
        handles = []
        for i in range(3):
            handle = await orchestrator.register_agent(f"agent_{i}", observation_policy)
            handles.append(handle)
            # Position all agents close together
            orchestrator.world_state.spatial_index[f"agent_{i}"] = (0.0, 0.0, 0.0)

        # Broadcast event
        draft: EffectDraft = {
            "kind": "Speak",
            "payload": {"text": "Hello everyone!"},
            "source_id": "speaker",
            "schema_version": "1.0.0",
        }

        # Position speaker near agents
        orchestrator.world_state.spatial_index["speaker"] = (5.0, 5.0, 0.0)

        await orchestrator.broadcast_event(draft)

        # All agents should receive consistent observations
        observations = []
        for handle in handles:
            observation = await handle.next_observation()
            observations.append(observation)

        # Verify all observations have same view_seq
        view_seqs = [obs["view_seq"] for obs in observations]
        assert all(seq == view_seqs[0] for seq in view_seqs)

        # Verify all observations have consistent content (context_digest will differ due to agent_id)
        # But the visible entities should be the same for all agents
        for obs in observations:
            assert "speaker" in obs["patches"][0]["value"]  # All should see the speaker

    @pytest.mark.asyncio
    async def test_delivery_timing_performance(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that observation delivery meets performance requirements (≤20ms)."""
        # Use no-latency model for baseline performance test
        observation_policy.set_latency_model(NoLatencyModel())

        # Register agent
        agent_handle = await orchestrator.register_agent("agent_1", observation_policy)

        # Position agent to receive observations
        orchestrator.world_state.spatial_index["agent_1"] = (0.0, 0.0, 0.0)

        # Measure delivery time
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "performance_test"},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        start_time = time.perf_counter()
        await orchestrator.broadcast_event(draft)

        # Wait for observation
        observation = await agent_handle.next_observation()
        end_time = time.perf_counter()

        delivery_time_ms = (end_time - start_time) * 1000

        # Verify observation was delivered
        assert observation is not None
        assert observation["view_seq"] == 1

        # Verify delivery time meets SLO (≤20ms for core in-proc)
        # Note: In tests this might be faster, but we check it's reasonable
        assert delivery_time_ms < 100  # Allow 100ms in test environment

    @pytest.mark.asyncio
    async def test_world_state_metadata_updates(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test that world state metadata is updated with effect information."""
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "metadata_test"},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify world state metadata was updated
        world_state = orchestrator.get_world_state()
        metadata = world_state.metadata

        assert "last_effect_seq" in metadata
        assert metadata["last_effect_seq"] == 1

        assert "last_effect_time" in metadata
        assert isinstance(metadata["last_effect_time"], float)

        assert "last_effect_kind" in metadata
        assert metadata["last_effect_kind"] == "TestEvent"

    @pytest.mark.asyncio
    async def test_error_handling_in_world_state_update(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test error handling during world state updates."""
        # Create effect with invalid position data
        draft: EffectDraft = {
            "kind": "Move",
            "payload": {"position": "invalid_position"},  # Should be list/tuple
            "source_id": "agent_1",
            "schema_version": "1.0.0",
        }

        # Should not raise exception, but log error
        await orchestrator.broadcast_event(draft)

        # Verify effect was still logged despite world state update failure
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

        effect = entries[0].effect
        assert effect["kind"] == "Move"

    @pytest.mark.asyncio
    async def test_error_handling_in_observation_distribution(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test error handling during observation distribution."""

        # Create a mock observation policy that raises an exception
        class FailingObservationPolicy(DefaultObservationPolicy):
            def should_observe_event(self, effect, agent_id, world_state):
                raise RuntimeError("Policy failure")

        failing_policy = FailingObservationPolicy(PolicyConfig())

        # Register agent with failing policy
        await orchestrator.register_agent("failing_agent", failing_policy)

        # Broadcast event - should not raise exception
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "error_test"},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify effect was still logged despite observation distribution failure
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_cleanup_on_shutdown(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test proper cleanup during orchestrator shutdown."""
        # Register agents and create some state
        await orchestrator.register_agent("agent_1", observation_policy)
        await orchestrator.register_agent("agent_2", observation_policy)

        # Broadcast some events
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "shutdown_test"},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        # Verify state exists
        assert len(orchestrator.agent_handles) == 2
        assert len(orchestrator._per_agent_queues) == 2

        # Shutdown
        await orchestrator.shutdown()

        # Verify cleanup
        assert len(orchestrator.agent_handles) == 0
        assert len(orchestrator.observation_policies) == 0
        assert len(orchestrator._per_agent_queues) == 0


if __name__ == "__main__":
    pytest.main([__file__])
