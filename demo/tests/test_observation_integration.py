"""
Integration tests for BattleObservationPolicy with existing models and schemas.

This module tests the integration between the observation policy and the
battle demo's data models, ensuring proper filtering and visibility rules.
"""

from demo.backend.gunn_integration import BattleObservationPolicy
from demo.shared.enums import AgentStatus, LocationType, WeaponCondition
from demo.shared.models import Agent, BattleWorldState, MapLocation
from demo.shared.utils import calculate_distance, serialize_world_state_for_api
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Effect


class TestObservationIntegration:
    """Integration tests for observation policy with battle models."""

    def setup_method(self):
        """Set up test fixtures with realistic battle scenario."""
        # Create a realistic battle world state
        self.battle_world = BattleWorldState(
            agents={
                "team_a_agent_1": Agent(
                    agent_id="team_a_agent_1",
                    team="team_a",
                    position=(50.0, 50.0),
                    health=85,
                    status=AgentStatus.ALIVE,
                    weapon_condition=WeaponCondition.GOOD,
                    vision_range=35.0,
                    attack_range=20.0,
                ),
                "team_a_agent_2": Agent(
                    agent_id="team_a_agent_2",
                    team="team_a",
                    position=(60.0, 45.0),
                    health=70,
                    status=AgentStatus.HEALING,
                    weapon_condition=WeaponCondition.DAMAGED,
                    vision_range=30.0,
                    attack_range=15.0,
                ),
                "team_b_agent_1": Agent(
                    agent_id="team_b_agent_1",
                    team="team_b",
                    position=(75.0, 55.0),  # Within vision range
                    health=95,
                    status=AgentStatus.ALIVE,
                    weapon_condition=WeaponCondition.EXCELLENT,
                    vision_range=40.0,
                    attack_range=25.0,
                ),
                "team_b_agent_2": Agent(
                    agent_id="team_b_agent_2",
                    team="team_b",
                    position=(150.0, 150.0),  # Far away
                    health=100,
                    status=AgentStatus.ALIVE,
                    weapon_condition=WeaponCondition.EXCELLENT,
                    vision_range=30.0,
                    attack_range=15.0,
                ),
            },
            map_locations={
                "forge_a": MapLocation(
                    position=(20.0, 20.0), location_type=LocationType.FORGE, radius=5.0
                ),
                "forge_b": MapLocation(
                    position=(180.0, 180.0),
                    location_type=LocationType.FORGE,
                    radius=5.0,
                ),
                "cover_1": MapLocation(
                    position=(65.0, 50.0), location_type=LocationType.COVER, radius=3.0
                ),
            },
            game_time=45.5,
            game_status="active",
        )

        # Add some team communications
        self.battle_world.add_team_message("team_a_agent_1", "Enemy at 75,55", "high")
        self.battle_world.add_team_message("team_a_agent_2", "Taking cover", "medium")
        self.battle_world.add_team_message("team_b_agent_1", "Engaging team A", "high")

        # Convert to Gunn WorldState format
        self.gunn_world_state = self._convert_to_gunn_world_state(self.battle_world)

        # Create observation policies
        self.team_a_policy = BattleObservationPolicy("team_a", vision_range=35.0)
        self.team_b_policy = BattleObservationPolicy("team_b", vision_range=40.0)

    def _convert_to_gunn_world_state(
        self, battle_world: BattleWorldState
    ) -> WorldState:
        """Convert BattleWorldState to Gunn WorldState format."""
        entities = {}
        spatial_index = {}

        # Add agents
        for agent_id, agent in battle_world.agents.items():
            entities[agent_id] = agent.model_dump()
            spatial_index[agent_id] = (*agent.position, 0.0)

        # Add map locations
        for loc_id, location in battle_world.map_locations.items():
            entities[loc_id] = {"type": "map_location", **location.model_dump()}
            spatial_index[loc_id] = (*location.position, 0.0)

        # Prepare metadata
        metadata = {
            "team_communications": battle_world.team_communications,
            "game_time": battle_world.game_time,
            "game_status": battle_world.game_status,
            "team_scores": battle_world.team_scores,
        }

        return WorldState(
            entities=entities,
            spatial_index=spatial_index,
            relationships={},
            metadata=metadata,
        )

    def test_realistic_team_a_visibility(self):
        """Test realistic visibility scenario for team A agent."""
        view = self.team_a_policy.filter_world_state(
            self.gunn_world_state, "team_a_agent_1"
        )

        # Should see self with full information
        assert "team_a_agent_1" in view.visible_entities
        self_data = view.visible_entities["team_a_agent_1"]
        assert self_data["health"] == 85
        assert self_data["weapon_condition"] == "good"
        assert self_data["vision_range"] == 35.0

        # Should see teammate with full information
        assert "team_a_agent_2" in view.visible_entities
        teammate_data = view.visible_entities["team_a_agent_2"]
        assert teammate_data["health"] == 70
        assert teammate_data["status"] == "healing"
        assert teammate_data["weapon_condition"] == "damaged"

        # Should see nearby enemy with limited information
        assert "team_b_agent_1" in view.visible_entities
        enemy_data = view.visible_entities["team_b_agent_1"]
        assert enemy_data["team"] == "team_b"
        assert enemy_data["position"] == (75.0, 55.0)
        assert enemy_data["health"] == 95  # Health visible for tactical decisions
        assert "weapon_condition" not in enemy_data  # Hidden due to fog of war
        assert "vision_range" not in enemy_data

        # Should NOT see distant enemy
        assert "team_b_agent_2" not in view.visible_entities

        # Should see all map locations
        assert "forge_a" in view.visible_entities
        assert "forge_b" in view.visible_entities
        assert "cover_1" in view.visible_entities

        # Should see team communications
        assert "team_communications" in view.visible_entities
        comm_data = view.visible_entities["team_communications"]
        assert comm_data["team"] == "team_a"
        assert len(comm_data["messages"]) == 2

    def test_realistic_team_b_visibility(self):
        """Test realistic visibility scenario for team B agent."""
        view = self.team_b_policy.filter_world_state(
            self.gunn_world_state, "team_b_agent_1"
        )

        # Should see self and distant teammate
        assert "team_b_agent_1" in view.visible_entities
        assert "team_b_agent_2" in view.visible_entities

        # Should see both team A agents (within 40 unit vision range)
        distance_to_a1 = calculate_distance((75.0, 55.0), (50.0, 50.0))
        distance_to_a2 = calculate_distance((75.0, 55.0), (60.0, 45.0))

        assert distance_to_a1 <= 40.0  # Verify test assumption
        assert distance_to_a2 <= 40.0  # Verify test assumption

        assert "team_a_agent_1" in view.visible_entities
        assert "team_a_agent_2" in view.visible_entities

        # Enemy data should be limited
        enemy_data = view.visible_entities["team_a_agent_1"]
        assert enemy_data["team"] == "team_a"
        assert "weapon_condition" not in enemy_data

        # Should see team B communications only
        comm_data = view.visible_entities["team_communications"]
        assert comm_data["team"] == "team_b"
        assert len(comm_data["messages"]) == 1
        assert comm_data["messages"][0]["sender_id"] == "team_b_agent_1"

    def test_vision_range_accuracy(self):
        """Test that vision range calculations are accurate."""
        # Test agent with 35.0 vision range
        view = self.team_a_policy.filter_world_state(
            self.gunn_world_state, "team_a_agent_1"
        )

        # Calculate actual distance to team_b_agent_1
        distance = calculate_distance((50.0, 50.0), (75.0, 55.0))

        if distance <= 35.0:
            assert "team_b_agent_1" in view.visible_entities
        else:
            assert "team_b_agent_1" not in view.visible_entities

        # Distant enemy should definitely not be visible
        distant_distance = calculate_distance((50.0, 50.0), (150.0, 150.0))
        assert distant_distance > 35.0
        assert "team_b_agent_2" not in view.visible_entities

    def test_combat_effect_observation(self):
        """Test observation of combat-related effects."""
        # Attack effect from visible enemy
        attack_effect: Effect = {
            "uuid": "attack-1",
            "kind": "Attack",
            "payload": {
                "attacker_id": "team_b_agent_1",
                "target_id": "team_a_agent_1",
                "damage": 25,
                "position": (75.0, 55.0),
            },
            "global_seq": 1,
            "sim_time": 46.0,
            "source_id": "team_b_agent_1",
            "schema_version": "1.0.0",
        }

        # Team A agent should observe attack from visible enemy
        assert self.team_a_policy.should_observe_event(
            attack_effect, "team_a_agent_1", self.gunn_world_state
        )

        # Teammate should also observe the attack
        assert self.team_a_policy.should_observe_event(
            attack_effect, "team_a_agent_2", self.gunn_world_state
        )

        # Attack from distant enemy targeting the agent
        distant_attack_on_self: Effect = {
            "uuid": "attack-2",
            "kind": "Attack",
            "payload": {
                "attacker_id": "team_b_agent_2",
                "target_id": "team_a_agent_1",
                "damage": 20,
            },
            "global_seq": 2,
            "sim_time": 46.5,
            "source_id": "team_b_agent_2",
            "schema_version": "1.0.0",
        }

        # SHOULD observe attack targeting self, even from distant enemy
        # (You know when you're being attacked, even if you can't see the attacker)
        assert self.team_a_policy.should_observe_event(
            distant_attack_on_self, "team_a_agent_1", self.gunn_world_state
        )

        # Attack from distant enemy targeting someone else
        distant_attack_on_other: Effect = {
            "uuid": "attack-3",
            "kind": "Attack",
            "payload": {
                "attacker_id": "team_b_agent_2",
                "target_id": "team_b_agent_1",  # Attacking their own teammate
                "damage": 20,
            },
            "global_seq": 3,
            "sim_time": 47.0,
            "source_id": "team_b_agent_2",
            "schema_version": "1.0.0",
        }

        # Should NOT observe attack from distant enemy on someone else
        assert not self.team_a_policy.should_observe_event(
            distant_attack_on_other, "team_a_agent_1", self.gunn_world_state
        )

    def test_healing_effect_observation(self):
        """Test observation of healing effects."""
        # Teammate healing effect
        heal_effect: Effect = {
            "uuid": "heal-1",
            "kind": "Heal",
            "payload": {
                "healer_id": "team_a_agent_1",
                "target_id": "team_a_agent_2",
                "heal_amount": 30,
                "position": (60.0, 45.0),
            },
            "global_seq": 3,
            "sim_time": 47.0,
            "source_id": "team_a_agent_1",
            "schema_version": "1.0.0",
        }

        # Both team members should observe healing
        assert self.team_a_policy.should_observe_event(
            heal_effect, "team_a_agent_1", self.gunn_world_state
        )
        assert self.team_a_policy.should_observe_event(
            heal_effect, "team_a_agent_2", self.gunn_world_state
        )

        # Nearby enemy should also observe healing (within vision)
        assert self.team_b_policy.should_observe_event(
            heal_effect, "team_b_agent_1", self.gunn_world_state
        )

        # Distant enemy should NOT observe healing
        assert not self.team_b_policy.should_observe_event(
            heal_effect, "team_b_agent_2", self.gunn_world_state
        )

    def test_communication_effect_isolation(self):
        """Test that communication effects are properly isolated by team."""
        # Team A communication
        team_a_comm: Effect = {
            "uuid": "comm-1",
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_a_agent_1",
                "sender_team": "team_a",
                "message": "Flanking maneuver",
                "urgency": "high",
            },
            "global_seq": 4,
            "sim_time": 48.0,
            "source_id": "team_a_agent_1",
            "schema_version": "1.0.0",
        }

        # Team A members should see their communication
        assert self.team_a_policy.should_observe_event(
            team_a_comm, "team_a_agent_1", self.gunn_world_state
        )
        assert self.team_a_policy.should_observe_event(
            team_a_comm, "team_a_agent_2", self.gunn_world_state
        )

        # Team B members should NOT see team A communication
        assert not self.team_b_policy.should_observe_event(
            team_a_comm, "team_b_agent_1", self.gunn_world_state
        )
        assert not self.team_b_policy.should_observe_event(
            team_a_comm, "team_b_agent_2", self.gunn_world_state
        )

    def test_world_state_serialization_compatibility(self):
        """Test that observation policy works with serialized world state."""
        # Serialize and deserialize world state (simulating API transfer)
        serialized = serialize_world_state_for_api(self.battle_world)

        # Create new world state from serialized data
        reconstructed_entities = {}
        reconstructed_spatial = {}

        for agent_id, agent_data in serialized["agents"].items():
            reconstructed_entities[agent_id] = agent_data
            reconstructed_spatial[agent_id] = (*agent_data["position"], 0.0)

        for loc_id, loc_data in serialized["map_locations"].items():
            reconstructed_entities[loc_id] = {"type": "map_location", **loc_data}
            reconstructed_spatial[loc_id] = (*loc_data["position"], 0.0)

        reconstructed_world = WorldState(
            entities=reconstructed_entities,
            spatial_index=reconstructed_spatial,
            relationships={},
            metadata={
                "team_communications": serialized["team_communications"],
                "game_time": serialized["game_time"],
                "game_status": serialized["game_status"],
            },
        )

        # Observation policy should work with reconstructed world state
        view = self.team_a_policy.filter_world_state(
            reconstructed_world, "team_a_agent_1"
        )

        assert "team_a_agent_1" in view.visible_entities
        assert "team_a_agent_2" in view.visible_entities
        assert "team_communications" in view.visible_entities

    def test_edge_case_agent_positions(self):
        """Test observation policy with edge case agent positions."""
        # Create world state with agents at exact vision range boundary
        boundary_world = WorldState(
            entities={
                "observer": {
                    "agent_id": "observer",
                    "team": "team_a",
                    "position": (0.0, 0.0),
                    "health": 100,
                    "status": "alive",
                },
                "boundary_agent": {
                    "agent_id": "boundary_agent",
                    "team": "team_b",
                    "position": (35.0, 0.0),  # Exactly at 35.0 vision range
                    "health": 100,
                    "status": "alive",
                },
                "just_outside": {
                    "agent_id": "just_outside",
                    "team": "team_b",
                    "position": (35.1, 0.0),  # Just outside vision range
                    "health": 100,
                    "status": "alive",
                },
            },
            spatial_index={
                "observer": (0.0, 0.0, 0.0),
                "boundary_agent": (35.0, 0.0, 0.0),
                "just_outside": (35.1, 0.0, 0.0),
            },
            relationships={},
            metadata={"team_communications": {"team_a": [], "team_b": []}},
        )

        view = self.team_a_policy.filter_world_state(boundary_world, "observer")

        # Agent at exactly vision range should be visible
        assert "boundary_agent" in view.visible_entities

        # Agent just outside vision range should not be visible
        assert "just_outside" not in view.visible_entities

    def test_multiple_policy_consistency(self):
        """Test that multiple observation policies work consistently."""
        # Create policies with different vision ranges
        short_range_policy = BattleObservationPolicy("team_a", vision_range=20.0)
        long_range_policy = BattleObservationPolicy("team_a", vision_range=50.0)

        short_view = short_range_policy.filter_world_state(
            self.gunn_world_state, "team_a_agent_1"
        )
        long_view = long_range_policy.filter_world_state(
            self.gunn_world_state, "team_a_agent_1"
        )

        # Short range view should be subset of long range view
        for entity_id in short_view.visible_entities:
            if entity_id != "team_communications":  # Communications are always included
                assert entity_id in long_view.visible_entities

        # Long range view should see more entities
        assert len(long_view.visible_entities) >= len(short_view.visible_entities)
