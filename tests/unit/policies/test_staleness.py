"""Unit tests for intelligent staleness detection in SpatialObservationPolicy.

This module tests the intent-specific staleness logic that prevents false positives
by only triggering cancellation when relevant preconditions change.
"""

import pytest

from gunn.policies.observation import (
    PolicyConfig,
    SpatialObservationPolicy,
    StalenessConfig,
)
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Intent


class TestSpatialObservationPolicyStaleness:
    """Test suite for SpatialObservationPolicy staleness detection."""

    def test_move_intent_not_stale_when_position_unchanged(self):
        """Move intent should not be stale if agent position hasn't changed."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(move_position_threshold=1.0),
        )

        # Create world states with same agent position
        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )

        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [10.0, 5.0, 0.0]},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert not policy.is_intent_stale(move_intent, old_state, new_state)

    def test_move_intent_stale_when_position_changed_significantly(self):
        """Move intent should be stale if agent position changed beyond threshold."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(move_position_threshold=1.0),
        )

        # Agent moved 2 units (beyond 1.0 threshold)
        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (2.0, 0.0, 0.0)},
        )

        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [10.0, 5.0, 0.0]},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert policy.is_intent_stale(move_intent, old_state, new_state)

    def test_move_intent_not_stale_when_position_changed_within_threshold(self):
        """Move intent should not be stale if position change is within threshold."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(move_position_threshold=2.0),
        )

        # Agent moved 1 unit (within 2.0 threshold)
        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (1.0, 0.0, 0.0)},
        )

        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [10.0, 5.0, 0.0]},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert not policy.is_intent_stale(move_intent, old_state, new_state)

    def test_move_intent_stale_when_new_obstacle_appears_near_target(self):
        """Move intent should be stale if new entity appears near target."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(move_position_threshold=1.0),
        )

        target_position = [10.0, 5.0, 0.0]

        # No obstacle near target initially
        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )

        # New obstacle appeared near target (within threshold)
        new_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "obstacle1": {"type": "obstacle"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "obstacle1": (10.5, 5.0, 0.0),  # 0.5 units from target
            },
        )

        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": target_position},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert policy.is_intent_stale(move_intent, old_state, new_state)

    def test_move_intent_supports_2d_coordinates(self):
        """Move intent should handle 2D coordinates correctly."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(move_position_threshold=1.0),
        )

        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )

        # Use 2D coordinates in payload
        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [10.0, 5.0]},  # 2D coordinates
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Should not crash and should work correctly
        assert not policy.is_intent_stale(move_intent, old_state, new_state)

    def test_speak_intent_not_stale_when_nearby_agents_unchanged(self):
        """Speak intent should not be stale if nearby agents haven't changed."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(speak_proximity_threshold=5.0),
        )

        # Same nearby agents in both states
        old_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (3.0, 0.0, 0.0),  # Within 5.0 threshold
            },
        )
        new_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (3.0, 0.0, 0.0),
            },
        )

        speak_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello everyone!"},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert not policy.is_intent_stale(speak_intent, old_state, new_state)

    def test_speak_intent_stale_when_agent_joins_conversation_range(self):
        """Speak intent should be stale if new agent enters conversation range."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(speak_proximity_threshold=5.0),
        )

        # Only agent2 nearby initially
        old_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (3.0, 0.0, 0.0),
            },
        )

        # agent3 joined conversation range
        new_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
                "agent3": {"type": "agent"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (3.0, 0.0, 0.0),
                "agent3": (4.0, 0.0, 0.0),  # Within 5.0 threshold
            },
        )

        speak_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello everyone!"},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert policy.is_intent_stale(speak_intent, old_state, new_state)

    def test_speak_intent_stale_when_agent_leaves_conversation_range(self):
        """Speak intent should be stale if agent leaves conversation range."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(speak_proximity_threshold=5.0),
        )

        # agent2 and agent3 nearby initially
        old_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
                "agent3": {"type": "agent"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (3.0, 0.0, 0.0),
                "agent3": (4.0, 0.0, 0.0),
            },
        )

        # agent3 left conversation range
        new_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
                "agent3": {"type": "agent"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (3.0, 0.0, 0.0),
                "agent3": (10.0, 0.0, 0.0),  # Beyond 5.0 threshold
            },
        )

        speak_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello everyone!"},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert policy.is_intent_stale(speak_intent, old_state, new_state)

    def test_speak_intent_not_stale_when_non_agent_entity_changes(self):
        """Speak intent should not be stale if non-agent entities change."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(speak_proximity_threshold=5.0),
        )

        old_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (3.0, 0.0, 0.0),
            },
        )

        # Non-agent entity appeared nearby (should not affect staleness)
        new_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
                "tree1": {"type": "tree"},  # Not an agent
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (3.0, 0.0, 0.0),
                "tree1": (2.0, 0.0, 0.0),
            },
        )

        speak_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello everyone!"},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert not policy.is_intent_stale(speak_intent, old_state, new_state)

    def test_agent_specific_thresholds(self):
        """Test that agent-specific thresholds override defaults."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(
                move_position_threshold=1.0,
                agent_specific_thresholds={
                    "agent1": {"Move": 5.0}  # Higher threshold for agent1
                },
            ),
        )

        # Agent moved 2 units (beyond default 1.0, within agent1's 5.0)
        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (2.0, 0.0, 0.0)},
        )

        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [10.0, 5.0, 0.0]},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Should not be stale due to agent-specific threshold
        assert not policy.is_intent_stale(move_intent, old_state, new_state)

    def test_custom_intent_uses_default_staleness(self):
        """Custom intent types should use default staleness behavior."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(default_staleness_enabled=True),
        )

        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )

        custom_intent: Intent = {
            "kind": "Custom",
            "payload": {"action": "dance"},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Should use default staleness (True)
        assert policy.is_intent_stale(custom_intent, old_state, new_state)

    def test_custom_intent_can_disable_default_staleness(self):
        """Custom intent types can disable default staleness."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(default_staleness_enabled=False),
        )

        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )

        custom_intent: Intent = {
            "kind": "Custom",
            "payload": {"action": "dance"},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Should not be stale when default is disabled
        assert not policy.is_intent_stale(custom_intent, old_state, new_state)

    def test_speak_intent_not_stale_when_agent_has_no_position(self):
        """Speak intent should handle agents without positions gracefully."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(speak_proximity_threshold=5.0),
        )

        # Agent has no position in spatial index
        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={},
        )
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={},
        )

        speak_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello!"},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Should not be stale (can't do spatial check)
        assert not policy.is_intent_stale(speak_intent, old_state, new_state)


class TestStalenessTimingAndAccuracy:
    """Test staleness detection timing and accuracy requirements."""

    def test_staleness_prevents_false_positives_for_unrelated_changes(self):
        """Staleness should not trigger for unrelated world changes."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(
                move_position_threshold=1.0,
                speak_proximity_threshold=5.0,
            ),
        )

        # World state with multiple agents
        old_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
                "agent3": {"type": "agent"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),
                "agent2": (20.0, 0.0, 0.0),  # Far away
                "agent3": (30.0, 0.0, 0.0),  # Far away
            },
        )

        # agent2 and agent3 moved (but far from agent1)
        new_state = WorldState(
            entities={
                "agent1": {"type": "agent"},
                "agent2": {"type": "agent"},
                "agent3": {"type": "agent"},
            },
            spatial_index={
                "agent1": (0.0, 0.0, 0.0),  # agent1 unchanged
                "agent2": (25.0, 0.0, 0.0),  # Moved
                "agent3": (35.0, 0.0, 0.0),  # Moved
            },
        )

        # agent1's Move intent should not be stale
        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [5.0, 0.0, 0.0]},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert not policy.is_intent_stale(move_intent, old_state, new_state)

        # agent1's Speak intent should not be stale
        speak_intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello!"},
            "context_seq": 1,
            "req_id": "req2",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        assert not policy.is_intent_stale(speak_intent, old_state, new_state)

    def test_staleness_detects_relevant_changes_quickly(self):
        """Staleness should detect relevant changes without delay."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(move_position_threshold=0.5),
        )

        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )

        # Small but significant position change
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.6, 0.0, 0.0)},  # 0.6 > 0.5 threshold
        )

        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [10.0, 0.0, 0.0]},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Should immediately detect staleness
        assert policy.is_intent_stale(move_intent, old_state, new_state)

    def test_multiple_staleness_checks_are_consistent(self):
        """Multiple staleness checks should return consistent results."""
        policy = SpatialObservationPolicy(
            PolicyConfig(distance_limit=10.0),
            StalenessConfig(move_position_threshold=1.0),
        )

        old_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (0.0, 0.0, 0.0)},
        )
        new_state = WorldState(
            entities={"agent1": {"type": "agent"}},
            spatial_index={"agent1": (2.0, 0.0, 0.0)},
        )

        move_intent: Intent = {
            "kind": "Move",
            "payload": {"to": [10.0, 0.0, 0.0]},
            "context_seq": 1,
            "req_id": "req1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Multiple checks should return same result
        result1 = policy.is_intent_stale(move_intent, old_state, new_state)
        result2 = policy.is_intent_stale(move_intent, old_state, new_state)
        result3 = policy.is_intent_stale(move_intent, old_state, new_state)

        assert result1 is True
        assert result1 == result2 == result3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
