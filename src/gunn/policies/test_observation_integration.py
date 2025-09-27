"""Integration tests for observation policies focusing on path stability and edge cases."""

import pytest

from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.messages import View, WorldState
from gunn.schemas.types import Effect


class TestPathStability:
    """Test JSON Patch path stability across entity reordering and changes."""

    @pytest.fixture
    def policy(self) -> DefaultObservationPolicy:
        """Create policy for path stability testing."""
        config = PolicyConfig(
            distance_limit=1000.0
        )  # Large distance to see all entities
        return DefaultObservationPolicy(config)

    def test_stable_paths_with_entity_reordering(
        self, policy: DefaultObservationPolicy
    ) -> None:
        """Test that JSON patch paths remain stable when entities are reordered."""
        # Create two world states with same entities but different ordering
        world_state_1 = WorldState(
            entities={
                "agent1": {"health": 100, "name": "Alice"},
                "agent2": {"health": 90, "name": "Bob"},
                "agent3": {"health": 80, "name": "Charlie"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (10.0, 0.0, 0.0),
                "agent3": (20.0, 0.0, 0.0),
            },
        )

        world_state_2 = WorldState(
            entities={
                "agent3": {"health": 80, "name": "Charlie"},  # Reordered
                "agent1": {"health": 100, "name": "Alice"},
                "agent2": {"health": 85, "name": "Bob"},  # Health changed
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (10.0, 0.0, 0.0),
                "agent3": (20.0, 0.0, 0.0),
            },
        )

        view1 = policy.filter_world_state(world_state_1, "agent1")
        view1.view_seq = 1

        view2 = policy.filter_world_state(world_state_2, "agent1")
        view2.view_seq = 2

        delta = policy.calculate_observation_delta(view1, view2)

        # Should have exactly one patch for the health change
        assert len(delta["patches"]) == 1

        patch = delta["patches"][0]
        assert patch["op"] == "replace"
        assert patch["path"] == "/visible_entities/agent2/health"
        assert patch["value"] == 85

    def test_stable_paths_with_entity_addition(
        self, policy: DefaultObservationPolicy
    ) -> None:
        """Test stable paths when entities are added."""
        world_state_1 = WorldState(
            entities={
                "agent1": {"health": 100, "name": "Alice"},
                "agent2": {"health": 90, "name": "Bob"},
            },
            spatial_index={"agent1": (0.0, 0.0, 0.0), "agent2": (10.0, 0.0, 0.0)},
        )

        world_state_2 = WorldState(
            entities={
                "agent1": {"health": 100, "name": "Alice"},
                "agent2": {"health": 90, "name": "Bob"},
                "agent3": {"health": 80, "name": "Charlie"},  # Added
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (10.0, 0.0, 0.0),
                "agent3": (20.0, 0.0, 0.0),
            },
        )

        view1 = policy.filter_world_state(world_state_1, "agent1")
        view1.view_seq = 1

        view2 = policy.filter_world_state(world_state_2, "agent1")
        view2.view_seq = 2

        delta = policy.calculate_observation_delta(view1, view2)

        # Should have one add operation for the new entity
        assert len(delta["patches"]) == 1

        patch = delta["patches"][0]
        assert patch["op"] == "add"
        assert patch["path"] == "/visible_entities/agent3"
        assert patch["value"] == {"health": 80, "name": "Charlie"}

    def test_stable_paths_with_entity_removal(
        self, policy: DefaultObservationPolicy
    ) -> None:
        """Test stable paths when entities are removed."""
        world_state_1 = WorldState(
            entities={
                "agent1": {"health": 100, "name": "Alice"},
                "agent2": {"health": 90, "name": "Bob"},
                "agent3": {"health": 80, "name": "Charlie"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (10.0, 0.0, 0.0),
                "agent3": (20.0, 0.0, 0.0),
            },
        )

        world_state_2 = WorldState(
            entities={
                "agent1": {"health": 100, "name": "Alice"},
                "agent3": {"health": 80, "name": "Charlie"},
                # agent2 removed
            },
            spatial_index={"agent1": (0.0, 0.0, 0.0), "agent3": (20.0, 0.0, 0.0)},
        )

        view1 = policy.filter_world_state(world_state_1, "agent1")
        view1.view_seq = 1

        view2 = policy.filter_world_state(world_state_2, "agent1")
        view2.view_seq = 2

        delta = policy.calculate_observation_delta(view1, view2)

        # Should have one remove operation
        assert len(delta["patches"]) == 1

        patch = delta["patches"][0]
        assert patch["op"] == "remove"
        assert patch["path"] == "/visible_entities/agent2"


class TestLargePatchFallback:
    """Test fallback to full snapshot for large changes."""

    def test_max_patch_ops_threshold(self) -> None:
        """Test that large changes trigger fallback to full snapshot."""
        config = PolicyConfig(distance_limit=1000.0, max_patch_ops=3)
        policy = DefaultObservationPolicy(config)

        # Create initial view with few entities
        old_view = View(
            agent_id="agent1",
            view_seq=1,
            visible_entities={"agent2": {"health": 100, "name": "Bob"}},
            visible_relationships={},
            context_digest="old_digest",
        )

        # Create new view with many new entities (will exceed max_patch_ops)
        new_view = View(
            agent_id="agent1",
            view_seq=2,
            visible_entities={
                "agent2": {"health": 90, "name": "Bob"},  # Changed
                "agent3": {"health": 80, "name": "Charlie"},  # Added
                "agent4": {"health": 70, "name": "David"},  # Added
                "agent5": {"health": 60, "name": "Eve"},  # Added
                "item1": {"type": "sword", "damage": 10},  # Added
                "item2": {"type": "shield", "defense": 5},  # Added
            },
            visible_relationships={
                "agent2": ["agent3", "agent4"],  # Added relationships
                "agent3": ["agent2"],
            },
            context_digest="new_digest",
        )

        delta = policy.calculate_observation_delta(old_view, new_view)

        # Should fallback to full snapshot (2 replace operations)
        assert len(delta["patches"]) == 2

        # Find the replace operations
        entity_replace = next(
            p for p in delta["patches"] if p["path"] == "/visible_entities"
        )
        relationship_replace = next(
            p for p in delta["patches"] if p["path"] == "/visible_relationships"
        )

        assert entity_replace["op"] == "replace"
        assert relationship_replace["op"] == "replace"

        # Verify full data is included
        assert len(entity_replace["value"]) == 6  # All entities
        assert len(relationship_replace["value"]) == 2  # All relationships

    def test_small_changes_no_fallback(self) -> None:
        """Test that small changes don't trigger fallback."""
        config = PolicyConfig(distance_limit=1000.0, max_patch_ops=10)
        policy = DefaultObservationPolicy(config)

        old_view = View(
            agent_id="agent1",
            view_seq=1,
            visible_entities={
                "agent2": {"health": 100, "name": "Bob"},
                "agent3": {"health": 90, "name": "Charlie"},
            },
            visible_relationships={"agent2": ["agent3"]},
            context_digest="old_digest",
        )

        new_view = View(
            agent_id="agent1",
            view_seq=2,
            visible_entities={
                "agent2": {"health": 85, "name": "Bob"},  # Health changed
                "agent3": {"health": 90, "name": "Charlie"},  # Unchanged
            },
            visible_relationships={
                "agent2": ["agent3"]  # Unchanged
            },
            context_digest="new_digest",
        )

        delta = policy.calculate_observation_delta(old_view, new_view)

        # Should have individual patch operations, not fallback
        assert len(delta["patches"]) == 1

        patch = delta["patches"][0]
        assert patch["op"] == "replace"
        assert patch["path"] == "/visible_entities/agent2/health"
        assert patch["value"] == 85


class TestFilteringAccuracy:
    """Test accuracy of observation filtering under various conditions."""

    @pytest.fixture
    def complex_world_state(self) -> WorldState:
        """Create complex world state for filtering tests."""
        return WorldState(
            entities={
                "agent1": {
                    "type": "agent",
                    "name": "Alice",
                    "health": 100,
                    "secret": "classified",
                },
                "agent2": {
                    "type": "agent",
                    "name": "Bob",
                    "health": 90,
                    "secret": "top_secret",
                },
                "agent3": {
                    "type": "agent",
                    "name": "Charlie",
                    "health": 80,
                    "secret": "public",
                },
                "npc1": {"type": "npc", "name": "Guard", "health": 120},
                "item1": {"type": "item", "name": "Sword", "damage": 10},
                "item2": {"type": "item", "name": "Potion", "healing": 25},
                "obstacle1": {"type": "obstacle", "name": "Wall", "solid": True},
            },
            relationships={
                "agent1": ["agent2", "item1"],
                "agent2": ["agent1", "agent3", "npc1"],
                "agent3": ["agent2"],
                "npc1": ["agent2"],
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (15.0, 0.0, 0.0),
                "agent3": (30.0, 0.0, 0.0),
                "npc1": (20.0, 0.0, 0.0),
                "item1": (5.0, 0.0, 0.0),
                "item2": (100.0, 0.0, 0.0),  # Far away
                "obstacle1": (25.0, 0.0, 0.0),
            },
        )

    def test_distance_filtering_accuracy(self, complex_world_state: WorldState) -> None:
        """Test accurate distance-based filtering."""
        config = PolicyConfig(distance_limit=25.0)
        policy = DefaultObservationPolicy(config)

        view = policy.filter_world_state(complex_world_state, "agent1")

        # Should see entities within 25 units
        expected_entities = {"agent1", "agent2", "npc1", "item1", "obstacle1"}
        actual_entities = set(view.visible_entities.keys())

        assert actual_entities == expected_entities

        # Should NOT see distant entities
        assert "agent3" not in view.visible_entities  # 30 units away
        assert "item2" not in view.visible_entities  # 100 units away

    def test_field_visibility_filtering_accuracy(
        self, complex_world_state: WorldState
    ) -> None:
        """Test accurate field-level visibility filtering."""
        config = PolicyConfig(
            distance_limit=100.0,
            field_visibility={
                "secret": False,  # Hide secret fields
                "health": True,  # Show health
                "name": True,  # Show name
                "type": True,  # Show type
            },
        )
        policy = DefaultObservationPolicy(config)

        view = policy.filter_world_state(complex_world_state, "agent1")

        # Check that secret fields are filtered out
        for entity_id, entity_data in view.visible_entities.items():
            if isinstance(entity_data, dict):
                assert "secret" not in entity_data, f"Secret field found in {entity_id}"

                # But other fields should be present
                if entity_id.startswith("agent"):
                    assert "health" in entity_data
                    assert "name" in entity_data
                    assert "type" in entity_data

    def test_relationship_filtering_accuracy(
        self, complex_world_state: WorldState
    ) -> None:
        """Test accurate relationship filtering."""
        config = PolicyConfig(distance_limit=100.0)
        policy = DefaultObservationPolicy(config)

        view = policy.filter_world_state(complex_world_state, "agent1")

        # Check that relationships are filtered to only visible entities
        for entity_id, relationships in view.visible_relationships.items():
            assert (
                entity_id in view.visible_entities
            ), f"Relationship source {entity_id} not visible"

            for target_id in relationships:
                assert (
                    target_id in view.visible_entities
                ), f"Relationship target {target_id} not visible"

    def test_relationship_depth_filtering(
        self, complex_world_state: WorldState
    ) -> None:
        """Test relationship depth filtering."""
        # Test with depth 1 (direct relationships only)
        config = PolicyConfig(
            distance_limit=1000.0,  # Large distance to focus on relationships
            relationship_filter=["friend"],  # Require relationships
            relationship_depth=1,
        )
        policy = DefaultObservationPolicy(config)

        view = policy.filter_world_state(complex_world_state, "agent1")

        # Should see self and directly related entities
        assert "agent1" in view.visible_entities
        assert "agent2" in view.visible_entities  # Direct relationship
        assert "item1" in view.visible_entities  # Direct relationship

        # Should NOT see indirectly related entities with depth 1
        # (This test assumes the relationship filtering is strict)


class TestEventObservationAccuracy:
    """Test accuracy of event observation decisions."""

    @pytest.fixture
    def policy(self) -> DefaultObservationPolicy:
        """Create policy for event observation testing."""
        config = PolicyConfig(distance_limit=50.0)
        return DefaultObservationPolicy(config)

    @pytest.fixture
    def world_state(self) -> WorldState:
        """Create world state for event testing."""
        return WorldState(
            entities={
                "agent1": {"type": "agent", "name": "Alice"},
                "agent2": {"type": "agent", "name": "Bob"},
                "agent3": {"type": "agent", "name": "Charlie"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (25.0, 0.0, 0.0),  # Within range
                "agent3": (100.0, 0.0, 0.0),  # Out of range
            },
        )

    def test_spatial_event_observation_accuracy(
        self, policy: DefaultObservationPolicy, world_state: WorldState
    ) -> None:
        """Test accurate spatial event observation."""
        # Event close to agent1
        close_effect: Effect = {
            "uuid": "test1",
            "kind": "Explosion",
            "payload": {"position": [10.0, 0.0, 0.0], "damage": 20},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "environment",
            "schema_version": "1.0.0",
        }

        # Event far from agent1
        far_effect: Effect = {
            "uuid": "test2",
            "kind": "Explosion",
            "payload": {"position": [200.0, 0.0, 0.0], "damage": 20},
            "global_seq": 2,
            "sim_time": 0.0,
            "source_id": "environment",
            "schema_version": "1.0.0",
        }

        # agent1 should observe close event but not far event
        assert policy.should_observe_event(close_effect, "agent1", world_state) is True
        assert policy.should_observe_event(far_effect, "agent1", world_state) is False

    def test_entity_specific_event_observation_accuracy(
        self, policy: DefaultObservationPolicy, world_state: WorldState
    ) -> None:
        """Test accurate entity-specific event observation."""
        # Event affecting agent2 (visible to agent1)
        visible_effect: Effect = {
            "uuid": "test1",
            "kind": "EntityDamaged",
            "payload": {"entity_id": "agent2", "damage": 10},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "environment",
            "schema_version": "1.0.0",
        }

        # Event affecting agent3 (not visible to agent1 due to distance)
        invisible_effect: Effect = {
            "uuid": "test2",
            "kind": "EntityDamaged",
            "payload": {"entity_id": "agent3", "damage": 10},
            "global_seq": 2,
            "sim_time": 0.0,
            "source_id": "environment",
            "schema_version": "1.0.0",
        }

        # agent1 should observe effect on agent2 but not agent3
        assert (
            policy.should_observe_event(visible_effect, "agent1", world_state) is True
        )
        assert (
            policy.should_observe_event(invisible_effect, "agent1", world_state)
            is False
        )

    def test_self_event_observation_accuracy(
        self, policy: DefaultObservationPolicy, world_state: WorldState
    ) -> None:
        """Test that agents always observe their own events."""
        self_effect: Effect = {
            "uuid": "test1",
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "global_seq": 1,
            "sim_time": 0.0,
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }

        # Agent should always observe own effects
        assert policy.should_observe_event(self_effect, "agent1", world_state) is True


if __name__ == "__main__":
    pytest.main([__file__])
