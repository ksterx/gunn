"""
Tests for BattleObservationPolicy implementation.

This module tests the team-based observation policy with fog of war,
team communication visibility, and vision range constraints.
"""

import pytest

from demo.backend.gunn_integration import BattleObservationPolicy
from demo.shared.enums import AgentStatus, LocationType, WeaponCondition
from demo.shared.models import Agent, MapLocation
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Effect


class TestBattleObservationPolicy:
    """Test suite for BattleObservationPolicy."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test agents
        self.team_a_agent_1 = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(10.0, 10.0),
            health=100,
            status=AgentStatus.ALIVE,
            weapon_condition=WeaponCondition.EXCELLENT,
            vision_range=30.0,
            attack_range=15.0,
            communication_range=50.0,
        )

        self.team_a_agent_2 = Agent(
            agent_id="team_a_agent_2",
            team="team_a",
            position=(15.0, 15.0),
            health=80,
            status=AgentStatus.ALIVE,
            weapon_condition=WeaponCondition.GOOD,
            vision_range=30.0,
            attack_range=15.0,
            communication_range=50.0,
        )

        self.team_b_agent_1 = Agent(
            agent_id="team_b_agent_1",
            team="team_b",
            position=(100.0, 100.0),  # Far away
            health=90,
            status=AgentStatus.ALIVE,
            weapon_condition=WeaponCondition.DAMAGED,
            vision_range=30.0,
            attack_range=15.0,
            communication_range=50.0,
        )

        self.team_b_agent_2 = Agent(
            agent_id="team_b_agent_2",
            team="team_b",
            position=(25.0, 25.0),  # Close to team A
            health=60,
            status=AgentStatus.ALIVE,
            weapon_condition=WeaponCondition.BROKEN,
            vision_range=30.0,
            attack_range=15.0,
            communication_range=50.0,
        )

        # Create map locations
        self.forge_a = MapLocation(
            position=(5.0, 5.0), location_type=LocationType.FORGE, radius=3.0
        )

        self.forge_b = MapLocation(
            position=(200.0, 200.0), location_type=LocationType.FORGE, radius=3.0
        )

        # Create world state
        self.world_state = WorldState(
            entities={
                "team_a_agent_1": self.team_a_agent_1.model_dump(),
                "team_a_agent_2": self.team_a_agent_2.model_dump(),
                "team_b_agent_1": self.team_b_agent_1.model_dump(),
                "team_b_agent_2": self.team_b_agent_2.model_dump(),
                "forge_a": {"type": "map_location", **self.forge_a.model_dump()},
                "forge_b": {"type": "map_location", **self.forge_b.model_dump()},
            },
            spatial_index={
                "team_a_agent_1": (10.0, 10.0, 0.0),
                "team_a_agent_2": (15.0, 15.0, 0.0),
                "team_b_agent_1": (100.0, 100.0, 0.0),
                "team_b_agent_2": (25.0, 25.0, 0.0),
                "forge_a": (5.0, 5.0, 0.0),
                "forge_b": (200.0, 200.0, 0.0),
            },
            relationships={},
            metadata={
                "team_communications": {
                    "team_a": [
                        {
                            "sender_id": "team_a_agent_1",
                            "team": "team_a",
                            "message": "Enemy spotted at coordinates 25,25",
                            "urgency": "high",
                            "timestamp": 10.0,
                        },
                        {
                            "sender_id": "team_a_agent_2",
                            "team": "team_a",
                            "message": "Moving to support",
                            "urgency": "medium",
                            "timestamp": 11.0,
                        },
                    ],
                    "team_b": [
                        {
                            "sender_id": "team_b_agent_1",
                            "team": "team_b",
                            "message": "Team A agents detected",
                            "urgency": "high",
                            "timestamp": 12.0,
                        }
                    ],
                }
            },
        )

        # Create observation policies
        self.team_a_policy = BattleObservationPolicy("team_a", vision_range=30.0)
        self.team_b_policy = BattleObservationPolicy("team_b", vision_range=30.0)

    def test_policy_initialization(self):
        """Test observation policy initialization."""
        policy = BattleObservationPolicy(
            "team_a", vision_range=25.0, communication_range=40.0
        )

        assert policy.team == "team_a"
        assert policy.vision_range == 25.0
        assert policy.communication_range == 40.0
        assert policy.config.distance_limit == 25.0

    def test_invalid_team_initialization(self):
        """Test that invalid team names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid team"):
            BattleObservationPolicy("invalid_team")

    def test_self_visibility(self):
        """Test that agents can always see themselves."""
        view = self.team_a_policy.filter_world_state(self.world_state, "team_a_agent_1")

        assert "team_a_agent_1" in view.visible_entities
        # Should see full data for self
        self_data = view.visible_entities["team_a_agent_1"]
        assert self_data["health"] == 100
        assert self_data["weapon_condition"] == "excellent"

    def test_teammate_visibility(self):
        """Test that agents can always see teammates with full information."""
        view = self.team_a_policy.filter_world_state(self.world_state, "team_a_agent_1")

        # Should see teammate
        assert "team_a_agent_2" in view.visible_entities

        # Should see full teammate data
        teammate_data = view.visible_entities["team_a_agent_2"]
        assert teammate_data["health"] == 80
        assert teammate_data["weapon_condition"] == "good"
        assert teammate_data["team"] == "team_a"

    def test_enemy_visibility_within_range(self):
        """Test that enemies within vision range are visible with limited information."""
        view = self.team_a_policy.filter_world_state(self.world_state, "team_a_agent_1")

        # Should see nearby enemy (team_b_agent_2 at 25,25 is within 30 units of 10,10)
        assert "team_b_agent_2" in view.visible_entities

        # Should see limited enemy data (fog of war)
        enemy_data = view.visible_entities["team_b_agent_2"]
        assert enemy_data["team"] == "team_b"
        assert enemy_data["position"] == (25.0, 25.0)
        assert enemy_data["status"] == "alive"
        assert enemy_data["health"] == 60  # Health visible for tactical decisions

        # Should NOT see detailed info like weapon condition
        assert "weapon_condition" not in enemy_data
        assert "attack_range" not in enemy_data
        assert "communication_range" not in enemy_data

    def test_enemy_visibility_out_of_range(self):
        """Test that enemies outside vision range are not visible."""
        view = self.team_a_policy.filter_world_state(self.world_state, "team_a_agent_1")

        # Should NOT see distant enemy (team_b_agent_1 at 100,100 is >30 units away)
        assert "team_b_agent_1" not in view.visible_entities

    def test_map_location_visibility(self):
        """Test that map locations are always visible."""
        view = self.team_a_policy.filter_world_state(self.world_state, "team_a_agent_1")

        # Should see both forges regardless of distance
        assert "forge_a" in view.visible_entities
        assert "forge_b" in view.visible_entities

        # Should see full map location data
        forge_data = view.visible_entities["forge_a"]
        assert forge_data["type"] == "map_location"
        assert forge_data["location_type"] == "forge"
        assert forge_data["position"] == (5.0, 5.0)

    def test_team_communication_visibility(self):
        """Test that only team communications are visible."""
        view = self.team_a_policy.filter_world_state(self.world_state, "team_a_agent_1")

        # Should see team communications
        assert "team_communications" in view.visible_entities

        comm_data = view.visible_entities["team_communications"]
        assert comm_data["type"] == "communications"
        assert comm_data["team"] == "team_a"

        # Should see team A messages
        messages = comm_data["messages"]
        assert len(messages) == 2
        assert messages[0]["sender_id"] == "team_a_agent_1"
        assert messages[0]["message"] == "Enemy spotted at coordinates 25,25"
        assert messages[1]["sender_id"] == "team_a_agent_2"

        # Test team B policy sees different messages
        view_b = self.team_b_policy.filter_world_state(
            self.world_state, "team_b_agent_1"
        )
        comm_data_b = view_b.visible_entities["team_communications"]
        messages_b = comm_data_b["messages"]
        assert len(messages_b) == 1
        assert messages_b[0]["sender_id"] == "team_b_agent_1"

    def test_should_observe_own_effects(self):
        """Test that agents always observe their own effects."""
        effect: Effect = {
            "uuid": "test-uuid",
            "kind": "Move",
            "payload": {"position": (20.0, 20.0)},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "team_a_agent_1",
            "schema_version": "1.0.0",
        }

        assert self.team_a_policy.should_observe_event(
            effect, "team_a_agent_1", self.world_state
        )

    def test_should_observe_teammate_effects(self):
        """Test that agents observe effects from visible teammates."""
        effect: Effect = {
            "uuid": "test-uuid",
            "kind": "Attack",
            "payload": {"target_id": "team_b_agent_2"},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "team_a_agent_2",  # Teammate
            "schema_version": "1.0.0",
        }

        assert self.team_a_policy.should_observe_event(
            effect, "team_a_agent_1", self.world_state
        )

    def test_should_observe_nearby_enemy_effects(self):
        """Test that agents observe effects from enemies within vision range."""
        effect: Effect = {
            "uuid": "test-uuid",
            "kind": "Attack",
            "payload": {"target_id": "team_a_agent_1"},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "team_b_agent_2",  # Nearby enemy
            "schema_version": "1.0.0",
        }

        assert self.team_a_policy.should_observe_event(
            effect, "team_a_agent_1", self.world_state
        )

    def test_should_not_observe_distant_enemy_effects(self):
        """Test that agents don't observe effects from enemies outside vision range."""
        effect: Effect = {
            "uuid": "test-uuid",
            "kind": "Move",
            "payload": {"position": (110.0, 110.0)},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "team_b_agent_1",  # Distant enemy
            "schema_version": "1.0.0",
        }

        assert not self.team_a_policy.should_observe_event(
            effect, "team_a_agent_1", self.world_state
        )

    def test_should_observe_effects_with_position(self):
        """Test observation of effects based on position proximity."""
        # Effect near the agent
        near_effect: Effect = {
            "uuid": "test-uuid",
            "kind": "Explosion",
            "payload": {"position": (15.0, 15.0), "damage": 50},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "environment",
            "schema_version": "1.0.0",
        }

        assert self.team_a_policy.should_observe_event(
            near_effect, "team_a_agent_1", self.world_state
        )

        # Effect far from the agent
        far_effect: Effect = {
            "uuid": "test-uuid",
            "kind": "Explosion",
            "payload": {"position": (200.0, 200.0), "damage": 50},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "environment",
            "schema_version": "1.0.0",
        }

        assert not self.team_a_policy.should_observe_event(
            far_effect, "team_a_agent_1", self.world_state
        )

    def test_team_communication_filtering(self):
        """Test that communication effects are filtered by team."""
        # Team A communication
        team_a_comm: Effect = {
            "uuid": "test-uuid",
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_a_agent_2",
                "sender_team": "team_a",
                "message": "Need backup",
                "urgency": "high",
            },
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "team_a_agent_2",
            "schema_version": "1.0.0",
        }

        # Team A agent should see team A communication
        assert self.team_a_policy.should_observe_communication(
            team_a_comm, "team_a_agent_1"
        )

        # Team B agent should NOT see team A communication
        assert not self.team_b_policy.should_observe_communication(
            team_a_comm, "team_b_agent_1"
        )

        # Team B communication
        team_b_comm: Effect = {
            "uuid": "test-uuid",
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_b_agent_1",
                "sender_team": "team_b",
                "message": "Flanking maneuver",
                "urgency": "medium",
            },
            "global_seq": 2,
            "sim_time": 2.0,
            "source_id": "team_b_agent_1",
            "schema_version": "1.0.0",
        }

        # Team B agent should see team B communication
        assert self.team_b_policy.should_observe_communication(
            team_b_comm, "team_b_agent_1"
        )

        # Team A agent should NOT see team B communication
        assert not self.team_a_policy.should_observe_communication(
            team_b_comm, "team_a_agent_1"
        )

    def test_communication_filtering_with_agent_id_inference(self):
        """Test communication filtering when team is inferred from agent ID."""
        # Communication without explicit sender_team
        comm_effect: Effect = {
            "uuid": "test-uuid",
            "kind": "CommunicateAction",
            "payload": {"sender_id": "team_a_agent_1", "message": "Moving to position"},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "team_a_agent_1",
            "schema_version": "1.0.0",
        }

        # Should infer team from agent ID
        assert self.team_a_policy.should_observe_communication(
            comm_effect, "team_a_agent_2"
        )
        assert not self.team_b_policy.should_observe_communication(
            comm_effect, "team_b_agent_1"
        )

    def test_non_communication_effects_use_normal_rules(self):
        """Test that non-communication effects use normal visibility rules."""
        move_effect: Effect = {
            "uuid": "test-uuid",
            "kind": "Move",
            "payload": {"position": (20.0, 20.0)},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "team_a_agent_1",
            "schema_version": "1.0.0",
        }

        # Non-communication effects should return True (use normal visibility rules)
        assert self.team_a_policy.should_observe_communication(
            move_effect, "team_a_agent_1"
        )
        assert self.team_b_policy.should_observe_communication(
            move_effect, "team_b_agent_1"
        )

    def test_view_context_digest_generation(self):
        """Test that view context digest is generated correctly."""
        view = self.team_a_policy.filter_world_state(self.world_state, "team_a_agent_1")

        assert view.context_digest is not None
        assert len(view.context_digest) == 64  # SHA-256 hex digest length
        assert isinstance(view.context_digest, str)

    def test_view_consistency(self):
        """Test that identical world states produce identical views."""
        view1 = self.team_a_policy.filter_world_state(
            self.world_state, "team_a_agent_1"
        )
        view2 = self.team_a_policy.filter_world_state(
            self.world_state, "team_a_agent_1"
        )

        # Views should be identical (excluding view_seq which may differ)
        assert view1.agent_id == view2.agent_id
        assert view1.visible_entities == view2.visible_entities
        assert view1.visible_relationships == view2.visible_relationships
        assert view1.context_digest == view2.context_digest

    def test_vision_range_boundary(self):
        """Test vision range boundary conditions."""
        # Create agent exactly at vision range boundary
        boundary_agent = Agent(
            agent_id="team_b_boundary",
            team="team_b",
            position=(40.0, 10.0),  # Exactly 30 units from (10,10)
            health=100,
            status=AgentStatus.ALIVE,
            weapon_condition=WeaponCondition.EXCELLENT,
        )

        # Add to world state
        test_world_state = WorldState(
            entities={
                **self.world_state.entities,
                "team_b_boundary": boundary_agent.model_dump(),
            },
            spatial_index={
                **self.world_state.spatial_index,
                "team_b_boundary": (40.0, 10.0, 0.0),
            },
            relationships=self.world_state.relationships,
            metadata=self.world_state.metadata,
        )

        view = self.team_a_policy.filter_world_state(test_world_state, "team_a_agent_1")

        # Agent at exactly vision range should be visible
        assert "team_b_boundary" in view.visible_entities

    def test_empty_team_communications(self):
        """Test handling of empty team communications."""
        empty_world_state = WorldState(
            entities=self.world_state.entities,
            spatial_index=self.world_state.spatial_index,
            relationships=self.world_state.relationships,
            metadata={},  # No team communications
        )

        view = self.team_a_policy.filter_world_state(
            empty_world_state, "team_a_agent_1"
        )

        # Should still have team_communications entry but with empty messages
        assert "team_communications" in view.visible_entities
        comm_data = view.visible_entities["team_communications"]
        assert comm_data["messages"] == []

    def test_agent_without_position(self):
        """Test handling of agents without spatial positions."""
        no_pos_world_state = WorldState(
            entities=self.world_state.entities,
            spatial_index={},  # No spatial positions
            relationships=self.world_state.relationships,
            metadata=self.world_state.metadata,
        )

        # Should not crash and should use default position
        view = self.team_a_policy.filter_world_state(
            no_pos_world_state, "team_a_agent_1"
        )

        # Should still see self and teammates
        assert "team_a_agent_1" in view.visible_entities
        assert "team_a_agent_2" in view.visible_entities
