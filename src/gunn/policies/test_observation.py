"""Unit tests for observation policies."""


import pytest

from gunn.policies.observation import (
    ConversationObservationPolicy,
    DefaultObservationPolicy,
    DistanceLatencyModel,
    PolicyConfig,
    create_observation_policy,
)
from gunn.schemas.messages import View, WorldState
from gunn.schemas.types import Effect


class TestPolicyConfig:
    """Test PolicyConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PolicyConfig()

        assert config.distance_limit == 100.0
        assert config.relationship_filter == []
        assert config.field_visibility == {}
        assert config.max_patch_ops == 50
        assert config.include_spatial_index is True
        assert config.relationship_depth == 2

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = PolicyConfig(
            distance_limit=50.0,
            relationship_filter=["friend", "ally"],
            max_patch_ops=25,
            relationship_depth=3,
        )

        assert config.distance_limit == 50.0
        assert config.relationship_filter == ["friend", "ally"]
        assert config.max_patch_ops == 25
        assert config.relationship_depth == 3

    def test_invalid_config(self):
        """Test invalid configuration values."""
        # Negative distance limit
        with pytest.raises(ValueError):
            PolicyConfig(distance_limit=-1.0)

        # Zero max_patch_ops
        with pytest.raises(ValueError):
            PolicyConfig(max_patch_ops=0)

        # Negative relationship depth
        with pytest.raises(ValueError):
            PolicyConfig(relationship_depth=-1)


class TestDistanceLatencyModel:
    """Test distance-based latency model."""

    def test_default_latency(self):
        """Test default latency calculation."""
        model = DistanceLatencyModel()

        effect: Effect = {
            "uuid": "test",
            "kind": "test",
            "payload": {},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }

        latency = model.calculate_delay("agent1", "agent2", effect)
        assert latency == 0.01  # base_latency

    def test_custom_latency(self):
        """Test custom latency parameters."""
        model = DistanceLatencyModel(base_latency=0.05, distance_factor=0.002)

        effect: Effect = {
            "uuid": "test",
            "kind": "test",
            "payload": {},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }

        latency = model.calculate_delay("agent1", "agent2", effect)
        assert latency == 0.05


class TestDefaultObservationPolicy:
    """Test default observation policy implementation."""

    @pytest.fixture
    def world_state(self) -> WorldState:
        """Create test world state."""
        return WorldState(
            entities={
                "agent1": {"type": "agent", "name": "Alice", "health": 100},
                "agent2": {"type": "agent", "name": "Bob", "health": 80},
                "agent3": {"type": "agent", "name": "Charlie", "health": 90},
                "item1": {"type": "item", "name": "Sword", "damage": 10},
                "wall1": {"type": "obstacle", "name": "Wall", "solid": True},
            },
            relationships={
                "agent1": ["agent2", "item1"],
                "agent2": ["agent1", "agent3"],
                "agent3": ["agent2"],
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (10.0, 0.0, 0.0),
                "agent3": (200.0, 0.0, 0.0),  # Far away
                "item1": (5.0, 0.0, 0.0),
                "wall1": (50.0, 0.0, 0.0),
            },
        )

    @pytest.fixture
    def policy(self) -> DefaultObservationPolicy:
        """Create test observation policy."""
        config = PolicyConfig(distance_limit=50.0, relationship_depth=2)
        return DefaultObservationPolicy(config)

    def test_filter_world_state_distance(
        self, policy: DefaultObservationPolicy, world_state: WorldState
    ):
        """Test distance-based filtering."""
        view = policy.filter_world_state(world_state, "agent1")

        # Should see self, agent2, item1, wall1 (all within 50 units)
        # Should NOT see agent3 (200 units away)
        assert "agent1" in view.visible_entities
        assert "agent2" in view.visible_entities
        assert "item1" in view.visible_entities
        assert "wall1" in view.visible_entities
        assert "agent3" not in view.visible_entities

        assert view.agent_id == "agent1"
        assert view.view_seq == 0
        assert len(view.context_digest) == 64  # SHA-256 hex digest

    def test_filter_world_state_relationships(self, world_state: WorldState):
        """Test relationship-based filtering."""
        config = PolicyConfig(
            distance_limit=1000.0,  # Large distance to test relationships only
            relationship_filter=["friend"],  # Require specific relationship
        )
        policy = DefaultObservationPolicy(config)

        view = policy.filter_world_state(world_state, "agent1")

        # Should see self and directly related entities
        assert "agent1" in view.visible_entities
        assert "agent2" in view.visible_entities  # Direct relationship
        assert "item1" in view.visible_entities  # Direct relationship

        # Relationships should be filtered to visible entities
        assert "agent1" in view.visible_relationships
        assert set(view.visible_relationships["agent1"]) == {"agent2", "item1"}

    def test_filter_world_state_field_visibility(self, world_state: WorldState):
        """Test field-level visibility filtering."""
        config = PolicyConfig(
            distance_limit=100.0,
            field_visibility={"health": False, "name": True, "type": True},
        )
        policy = DefaultObservationPolicy(config)

        view = policy.filter_world_state(world_state, "agent1")

        # Check that health field is filtered out
        agent2_data = view.visible_entities["agent2"]
        assert "name" in agent2_data
        assert "type" in agent2_data
        assert "health" not in agent2_data

    def test_should_observe_event_self(
        self, policy: DefaultObservationPolicy, world_state: WorldState
    ):
        """Test that agents always observe their own effects."""
        effect: Effect = {
            "uuid": "test",
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }

        assert policy.should_observe_event(effect, "agent1", world_state) is True

    def test_should_observe_event_spatial(
        self, policy: DefaultObservationPolicy, world_state: WorldState
    ):
        """Test spatial event observation."""
        # Effect at position (5, 0, 0) - close to agent1 at (0, 0, 0)
        effect: Effect = {
            "uuid": "test",
            "kind": "Explosion",
            "payload": {"position": [5.0, 0.0, 0.0], "damage": 20},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "environment",
            "schema_version": "1.0.0",
        }

        # agent1 should observe (within 50 unit limit)
        assert policy.should_observe_event(effect, "agent1", world_state) is True

        # agent3 should not observe (200 units away)
        assert policy.should_observe_event(effect, "agent3", world_state) is False

    def test_should_observe_event_entity(
        self, policy: DefaultObservationPolicy, world_state: WorldState
    ):
        """Test entity-specific event observation."""
        effect: Effect = {
            "uuid": "test",
            "kind": "EntityDamaged",
            "payload": {"entity_id": "agent2", "damage": 10},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "environment",
            "schema_version": "1.0.0",
        }

        # agent1 should observe (agent2 is visible)
        assert policy.should_observe_event(effect, "agent1", world_state) is True

        # agent3 should not observe (agent2 not visible due to distance)
        assert policy.should_observe_event(effect, "agent3", world_state) is False

    def test_calculate_observation_delta_small_change(
        self, policy: DefaultObservationPolicy
    ):
        """Test observation delta calculation for small changes."""
        old_view = View(
            agent_id="agent1",
            view_seq=1,
            visible_entities={"agent2": {"health": 100, "name": "Bob"}},
            visible_relationships={"agent2": ["agent1"]},
            context_digest="old_digest",
        )

        new_view = View(
            agent_id="agent1",
            view_seq=2,
            visible_entities={
                "agent2": {"health": 90, "name": "Bob"}
            },  # Health changed
            visible_relationships={"agent2": ["agent1"]},
            context_digest="new_digest",
        )

        delta = policy.calculate_observation_delta(old_view, new_view)

        assert delta["view_seq"] == 2
        assert delta["context_digest"] == "new_digest"
        assert delta["schema_version"] == "1.0.0"
        assert len(delta["patches"]) == 1

        # Should have a replace operation for the health field
        patch = delta["patches"][0]
        assert patch["op"] == "replace"
        assert patch["path"] == "/visible_entities/agent2/health"
        assert patch["value"] == 90

    def test_calculate_observation_delta_large_change(self, world_state: WorldState):
        """Test observation delta fallback for large changes."""
        config = PolicyConfig(max_patch_ops=2)  # Low threshold for testing
        policy = DefaultObservationPolicy(config)

        old_view = View(
            agent_id="agent1",
            view_seq=1,
            visible_entities={"agent2": {"health": 100, "name": "Bob"}},
            visible_relationships={},
            context_digest="old_digest",
        )

        # Large change - add many new entities
        new_view = View(
            agent_id="agent1",
            view_seq=2,
            visible_entities={
                "agent2": {"health": 90, "name": "Bob"},
                "agent3": {"health": 80, "name": "Charlie"},
                "item1": {"type": "sword", "damage": 10},
                "item2": {"type": "shield", "defense": 5},
            },
            visible_relationships={"agent2": ["agent3"]},
            context_digest="new_digest",
        )

        delta = policy.calculate_observation_delta(old_view, new_view)

        # Should fallback to full snapshot (replace operations)
        assert (
            len(delta["patches"]) == 2
        )  # Replace visible_entities and visible_relationships

        replace_entities = next(
            p for p in delta["patches"] if p["path"] == "/visible_entities"
        )
        assert replace_entities["op"] == "replace"
        assert len(replace_entities["value"]) == 4

    def test_calculate_observation_delta_different_agents(
        self, policy: DefaultObservationPolicy
    ):
        """Test error when calculating delta between different agents."""
        old_view = View(
            agent_id="agent1",
            view_seq=1,
            visible_entities={},
            visible_relationships={},
            context_digest="digest1",
        )

        new_view = View(
            agent_id="agent2",  # Different agent
            view_seq=2,
            visible_entities={},
            visible_relationships={},
            context_digest="digest2",
        )

        with pytest.raises(
            ValueError, match="Cannot generate delta between views for different agents"
        ):
            policy.calculate_observation_delta(old_view, new_view)

    def test_relationship_path_detection(
        self, policy: DefaultObservationPolicy, world_state: WorldState
    ):
        """Test relationship path detection within depth limits."""
        # agent1 -> agent2 -> agent3 (depth 2)
        assert policy._has_relationship_path("agent1", "agent3", world_state, 2) is True
        assert (
            policy._has_relationship_path("agent1", "agent3", world_state, 1) is False
        )

        # Direct relationship
        assert policy._has_relationship_path("agent1", "agent2", world_state, 1) is True

        # No path
        assert policy._has_relationship_path("agent1", "wall1", world_state, 3) is False

    def test_distance_calculation(self, policy: DefaultObservationPolicy):
        """Test distance calculation between positions."""
        pos1 = (0.0, 0.0, 0.0)
        pos2 = (3.0, 4.0, 0.0)  # 3-4-5 triangle

        distance = policy._calculate_distance(pos1, pos2)
        assert abs(distance - 5.0) < 0.001  # Should be 5.0

    def test_context_digest_stability(self, policy: DefaultObservationPolicy):
        """Test that context digest is stable for same content."""
        view_data1 = {
            "agent_id": "agent1",
            "visible_entities": {"agent2": {"health": 100}},
            "visible_relationships": {},
        }

        view_data2 = {
            "agent_id": "agent1",
            "visible_entities": {"agent2": {"health": 100}},
            "visible_relationships": {},
        }

        digest1 = policy._generate_context_digest(view_data1)
        digest2 = policy._generate_context_digest(view_data2)

        assert digest1 == digest2
        assert len(digest1) == 64  # SHA-256 hex


class TestConversationObservationPolicy:
    """Test conversation-specific observation policy."""

    @pytest.fixture
    def world_state(self) -> WorldState:
        """Create conversation-focused world state."""
        return WorldState(
            entities={
                "agent1": {"type": "agent", "name": "Alice", "role": "participant"},
                "agent2": {"type": "agent", "name": "Bob", "role": "participant"},
                "agent3": {"type": "agent", "name": "Charlie", "role": "observer"},
                "item1": {"type": "item", "name": "Table"},
            },
            relationships={
                "agent1": ["agent2"],  # Conversation participants
                "agent2": ["agent1", "agent3"],
                "agent3": ["agent2"],
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (1.0, 0.0, 0.0),
                "agent3": (2.0, 0.0, 0.0),
                "item1": (5.0, 0.0, 0.0),
            },
        )

    @pytest.fixture
    def policy(self) -> ConversationObservationPolicy:
        """Create conversation observation policy."""
        config = PolicyConfig()
        return ConversationObservationPolicy(config)

    def test_filter_world_state_conversation(
        self, policy: ConversationObservationPolicy, world_state: WorldState
    ):
        """Test conversation-focused world state filtering."""
        view = policy.filter_world_state(world_state, "agent1")

        # Should see self and conversation participants (other agents)
        assert "agent1" in view.visible_entities
        assert "agent2" in view.visible_entities

        # Should not see non-agent entities
        assert "item1" not in view.visible_entities

        # Should see relationships between visible agents
        assert (
            "agent1" in view.visible_relationships
            or "agent2" in view.visible_relationships
        )

    def test_should_observe_speaking_events(
        self, policy: ConversationObservationPolicy, world_state: WorldState
    ):
        """Test observation of speaking events."""
        speak_effect: Effect = {
            "uuid": "test",
            "kind": "Speak",
            "payload": {"text": "Hello everyone!"},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "agent2",
            "schema_version": "1.0.0",
        }

        # agent1 should observe (agent2 is a conversation participant)
        assert policy.should_observe_event(speak_effect, "agent1", world_state) is True

        # agent3 should not observe (not directly related to agent2 in this test)
        # Note: This depends on the relationship structure in the test data

    def test_should_observe_participant_events(
        self, policy: ConversationObservationPolicy, world_state: WorldState
    ):
        """Test observation of participant join/leave events."""
        join_effect: Effect = {
            "uuid": "test",
            "kind": "ParticipantJoined",
            "payload": {"participant": "agent4"},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "system",
            "schema_version": "1.0.0",
        }

        # All agents should observe participant events
        assert policy.should_observe_event(join_effect, "agent1", world_state) is True
        assert policy.should_observe_event(join_effect, "agent2", world_state) is True
        assert policy.should_observe_event(join_effect, "agent3", world_state) is True

    def test_should_not_observe_non_conversation_events(
        self, policy: ConversationObservationPolicy, world_state: WorldState
    ):
        """Test that non-conversation events are not observed."""
        move_effect: Effect = {
            "uuid": "test",
            "kind": "Move",
            "payload": {"entity_id": "item1", "position": [10, 0, 0]},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "environment",
            "schema_version": "1.0.0",
        }

        # Should not observe item movement
        assert policy.should_observe_event(move_effect, "agent1", world_state) is False


class TestObservationPolicyFactory:
    """Test observation policy factory function."""

    def test_create_default_policy(self):
        """Test creating default observation policy."""
        config = PolicyConfig()
        policy = create_observation_policy("default", config)

        assert isinstance(policy, DefaultObservationPolicy)
        assert policy.config == config

    def test_create_conversation_policy(self):
        """Test creating conversation observation policy."""
        config = PolicyConfig()
        policy = create_observation_policy("conversation", config)

        assert isinstance(policy, ConversationObservationPolicy)
        assert policy.config == config

    def test_create_unknown_policy(self):
        """Test error for unknown policy type."""
        config = PolicyConfig()

        with pytest.raises(
            ValueError, match="Unknown observation policy type: unknown"
        ):
            create_observation_policy("unknown", config)


class TestLatencyModelIntegration:
    """Test latency model integration with observation policies."""

    def test_set_latency_model(self):
        """Test setting custom latency model."""
        config = PolicyConfig()
        policy = DefaultObservationPolicy(config)

        latency_model = DistanceLatencyModel(base_latency=0.1)
        policy.set_latency_model(latency_model)

        assert policy.latency_model == latency_model

    def test_latency_calculation(self):
        """Test latency calculation through policy."""
        config = PolicyConfig()
        policy = DefaultObservationPolicy(config)

        latency_model = DistanceLatencyModel(base_latency=0.05)
        policy.set_latency_model(latency_model)

        effect: Effect = {
            "uuid": "test",
            "kind": "test",
            "payload": {},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }

        delay = policy.latency_model.calculate_delay("agent1", "agent2", effect)
        assert delay == 0.05


if __name__ == "__main__":
    pytest.main([__file__])
