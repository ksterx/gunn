"""
Tests for battle demo utility functions.

This module contains comprehensive tests for coordinate calculations, distance measurements,
team management utilities, and data conversion functions.
"""

import math

import pytest

from demo.shared.enums import AgentStatus, LocationType, WeaponCondition
from demo.shared.models import Agent, BattleWorldState, MapLocation
from demo.shared.utils import (
    calculate_angle,
    calculate_direction_vector,
    calculate_distance,
    calculate_manhattan_distance,
    calculate_movement_time,
    calculate_team_stats,
    convert_to_screen_coords,
    convert_to_world_coords,
    format_game_time,
    format_health_percentage,
    get_agents_in_range,
    get_enemy_team,
    get_nearest_agent,
    get_nearest_location,
    interpolate_position,
    is_within_range,
    normalize_position,
    serialize_agent_for_api,
    serialize_world_state_for_api,
    validate_team_name,
)


class TestDistanceCalculations:
    """Test distance and coordinate calculation functions."""

    def test_calculate_distance(self):
        """Test Euclidean distance calculation."""
        # Same point
        assert calculate_distance((0, 0), (0, 0)) == 0.0

        # Horizontal distance
        assert calculate_distance((0, 0), (3, 0)) == 3.0

        # Vertical distance
        assert calculate_distance((0, 0), (0, 4)) == 4.0

        # Diagonal distance (3-4-5 triangle)
        assert calculate_distance((0, 0), (3, 4)) == 5.0

        # Negative coordinates
        assert calculate_distance((-1, -1), (2, 3)) == 5.0

    def test_calculate_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        # Same point
        assert calculate_manhattan_distance((0, 0), (0, 0)) == 0.0

        # Horizontal distance
        assert calculate_manhattan_distance((0, 0), (3, 0)) == 3.0

        # Vertical distance
        assert calculate_manhattan_distance((0, 0), (0, 4)) == 4.0

        # Diagonal distance
        assert calculate_manhattan_distance((0, 0), (3, 4)) == 7.0

        # Negative coordinates
        assert calculate_manhattan_distance((-1, -1), (2, 3)) == 7.0

    def test_calculate_angle(self):
        """Test angle calculation between positions."""
        # Right (0 degrees)
        angle = calculate_angle((0, 0), (1, 0))
        assert abs(angle - 0.0) < 1e-10

        # Up (90 degrees)
        angle = calculate_angle((0, 0), (0, 1))
        assert abs(angle - math.pi / 2) < 1e-10

        # Left (180 degrees)
        angle = calculate_angle((0, 0), (-1, 0))
        assert abs(abs(angle) - math.pi) < 1e-10

        # Down (-90 degrees)
        angle = calculate_angle((0, 0), (0, -1))
        assert abs(angle + math.pi / 2) < 1e-10

    def test_calculate_direction_vector(self):
        """Test direction vector calculation."""
        # Same point
        assert calculate_direction_vector((0, 0), (0, 0)) == (0.0, 0.0)

        # Right
        assert calculate_direction_vector((0, 0), (5, 0)) == (1.0, 0.0)

        # Up
        assert calculate_direction_vector((0, 0), (0, 3)) == (0.0, 1.0)

        # Diagonal (normalized)
        dx, dy = calculate_direction_vector((0, 0), (3, 4))
        assert abs(dx - 0.6) < 1e-10
        assert abs(dy - 0.8) < 1e-10

        # Verify normalization
        magnitude = math.sqrt(dx * dx + dy * dy)
        assert abs(magnitude - 1.0) < 1e-10


class TestPositionUtilities:
    """Test position manipulation and validation functions."""

    def test_normalize_position(self):
        """Test position normalization within map boundaries."""
        # Position within bounds
        assert normalize_position((50, 30), 100, 60) == (50, 30)

        # Position at boundaries
        assert normalize_position((0, 0), 100, 60) == (0, 0)
        assert normalize_position((100, 60), 100, 60) == (100, 60)

        # Position outside bounds
        assert normalize_position((-10, 70), 100, 60) == (0, 60)
        assert normalize_position((150, -5), 100, 60) == (100, 0)

    def test_is_within_range(self):
        """Test range checking between positions."""
        # Within range
        assert is_within_range((0, 0), (3, 4), 5.0)
        assert is_within_range((0, 0), (3, 4), 5.1)

        # Exactly at range
        assert is_within_range((0, 0), (3, 4), 5.0)

        # Outside range
        assert not is_within_range((0, 0), (3, 4), 4.9)
        assert not is_within_range((0, 0), (10, 0), 5.0)

    def test_interpolate_position(self):
        """Test position interpolation."""
        start = (0.0, 0.0)
        end = (10.0, 20.0)

        # At start
        assert interpolate_position(start, end, 0.0) == (0.0, 0.0)

        # At end
        assert interpolate_position(start, end, 1.0) == (10.0, 20.0)

        # Halfway
        assert interpolate_position(start, end, 0.5) == (5.0, 10.0)

        # Quarter way
        assert interpolate_position(start, end, 0.25) == (2.5, 5.0)

        # Clamping - values outside [0, 1]
        assert interpolate_position(start, end, -0.5) == (0.0, 0.0)
        assert interpolate_position(start, end, 1.5) == (10.0, 20.0)

    def test_calculate_movement_time(self):
        """Test movement time calculation."""
        # Normal movement
        time = calculate_movement_time((0, 0), (3, 4), 5.0)
        assert time == 1.0  # Distance 5, speed 5 = 1 second

        # Zero distance
        time = calculate_movement_time((5, 5), (5, 5), 10.0)
        assert time == 0.0

        # Zero speed
        time = calculate_movement_time((0, 0), (10, 0), 0.0)
        assert time == float("inf")

        # Negative speed
        time = calculate_movement_time((0, 0), (10, 0), -5.0)
        assert time == float("inf")


class TestCoordinateConversion:
    """Test coordinate conversion functions."""

    def test_convert_to_screen_coords(self):
        """Test world to screen coordinate conversion."""
        # Simple 1:1 mapping
        screen_coords = convert_to_screen_coords((50, 30), 100, 60, 100, 60)
        assert screen_coords == (50, 30)

        # 2:1 scaling
        screen_coords = convert_to_screen_coords((50, 30), 100, 60, 200, 120)
        assert screen_coords == (100, 60)

        # Different aspect ratio
        screen_coords = convert_to_screen_coords((100, 60), 200, 120, 400, 180)
        assert screen_coords == (200, 90)

    def test_convert_to_world_coords(self):
        """Test screen to world coordinate conversion."""
        # Simple 1:1 mapping
        world_coords = convert_to_world_coords((50, 30), 100, 60, 100, 60)
        assert world_coords == (50.0, 30.0)

        # 2:1 scaling
        world_coords = convert_to_world_coords((100, 60), 100, 60, 200, 120)
        assert world_coords == (50.0, 30.0)

        # Different aspect ratio
        world_coords = convert_to_world_coords((200, 90), 200, 120, 400, 180)
        assert world_coords == (100.0, 60.0)


class TestAgentUtilities:
    """Test agent-related utility functions."""

    def setup_method(self):
        """Set up test agents."""
        self.agents = {
            "team_a_agent_1": Agent(
                agent_id="team_a_agent_1", team="team_a", position=(0, 0)
            ),
            "team_a_agent_2": Agent(
                agent_id="team_a_agent_2", team="team_a", position=(10, 0)
            ),
            "team_b_agent_1": Agent(
                agent_id="team_b_agent_1", team="team_b", position=(20, 0)
            ),
            "team_b_agent_2": Agent(
                agent_id="team_b_agent_2", team="team_b", position=(30, 0)
            ),
        }

    def test_get_agents_in_range(self):
        """Test getting agents within range."""
        agent = self.agents["team_a_agent_1"]

        # All agents within range
        in_range = get_agents_in_range(agent, self.agents, 50.0)
        assert set(in_range) == {"team_a_agent_2", "team_b_agent_1", "team_b_agent_2"}

        # Only close agents
        in_range = get_agents_in_range(agent, self.agents, 15.0)
        assert set(in_range) == {"team_a_agent_2"}

        # No agents in range
        in_range = get_agents_in_range(agent, self.agents, 5.0)
        assert in_range == []

        # Only teammates
        in_range = get_agents_in_range(
            agent, self.agents, 50.0, include_teammates=True, include_enemies=False
        )
        assert in_range == ["team_a_agent_2"]

        # Only enemies
        in_range = get_agents_in_range(
            agent, self.agents, 50.0, include_teammates=False, include_enemies=True
        )
        assert set(in_range) == {"team_b_agent_1", "team_b_agent_2"}

    def test_get_nearest_agent(self):
        """Test getting nearest agent."""
        agent = self.agents["team_a_agent_1"]

        # Nearest overall
        nearest = get_nearest_agent(agent, self.agents)
        assert nearest == "team_a_agent_2"  # Distance 10

        # Nearest teammate
        nearest = get_nearest_agent(
            agent, self.agents, include_teammates=True, include_enemies=False
        )
        assert nearest == "team_a_agent_2"

        # Nearest enemy
        nearest = get_nearest_agent(
            agent, self.agents, include_teammates=False, include_enemies=True
        )
        assert nearest == "team_b_agent_1"  # Distance 20

        # No agents match criteria
        single_agent = {"team_a_agent_1": agent}
        nearest = get_nearest_agent(agent, single_agent)
        assert nearest is None


class TestMapUtilities:
    """Test map-related utility functions."""

    def setup_method(self):
        """Set up test map locations."""
        self.locations = {
            "forge_a": MapLocation(position=(10, 10), location_type=LocationType.FORGE),
            "forge_b": MapLocation(position=(90, 90), location_type=LocationType.FORGE),
            "cover_1": MapLocation(position=(50, 50), location_type=LocationType.COVER),
        }

    def test_get_nearest_location(self):
        """Test getting nearest map location."""
        # Nearest overall
        nearest = get_nearest_location((0, 0), self.locations)
        assert nearest == "forge_a"

        # Nearest of specific type
        nearest = get_nearest_location((80, 80), self.locations, "forge")
        assert nearest == "forge_b"

        # No locations of specified type
        nearest = get_nearest_location((0, 0), self.locations, "spawn_point")
        assert nearest is None


class TestFormattingUtilities:
    """Test formatting and display utility functions."""

    def test_format_game_time(self):
        """Test game time formatting."""
        assert format_game_time(0) == "00:00"
        assert format_game_time(30) == "00:30"
        assert format_game_time(60) == "01:00"
        assert format_game_time(125) == "02:05"
        assert format_game_time(3661) == "61:01"

    def test_format_health_percentage(self):
        """Test health percentage formatting."""
        assert format_health_percentage(100) == "100%"
        assert format_health_percentage(75) == "75%"
        assert format_health_percentage(0) == "0%"
        assert format_health_percentage(50, 200) == "25%"


class TestSerializationUtilities:
    """Test data serialization functions."""

    def test_serialize_agent_for_api(self):
        """Test agent serialization for API responses."""
        agent = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(50.0, 30.0),
            health=75,
            status=AgentStatus.ALIVE,
            weapon_condition=WeaponCondition.GOOD,
        )

        serialized = serialize_agent_for_api(agent)

        expected_keys = {
            "agent_id",
            "team",
            "position",
            "health",
            "status",
            "weapon_condition",
            "last_action_time",
            "communication_range",
            "vision_range",
            "attack_range",
            "is_alive",
            "can_attack",
        }
        assert set(serialized.keys()) == expected_keys
        assert serialized["agent_id"] == "team_a_agent_1"
        assert serialized["team"] == "team_a"
        assert serialized["position"] == (50.0, 30.0)
        assert serialized["health"] == 75
        assert serialized["status"] == "alive"
        assert serialized["weapon_condition"] == "good"
        assert serialized["is_alive"] is True
        assert serialized["can_attack"] is True

    def test_serialize_world_state_for_api(self):
        """Test world state serialization for API responses."""
        agent = Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0))
        location = MapLocation(position=(50, 50), location_type=LocationType.FORGE)

        world_state = BattleWorldState(
            agents={"team_a_agent_1": agent},
            map_locations={"forge_a": location},
            game_time=123.45,
        )
        world_state.add_team_message("team_a_agent_1", "Test message")

        serialized = serialize_world_state_for_api(world_state)

        expected_keys = {
            "agents",
            "map_locations",
            "team_scores",
            "game_time",
            "game_status",
            "team_communications",
        }
        assert set(serialized.keys()) == expected_keys
        assert "team_a_agent_1" in serialized["agents"]
        assert "forge_a" in serialized["map_locations"]
        assert serialized["game_time"] == 123.45
        assert len(serialized["team_communications"]["team_a"]) == 1


class TestTeamUtilities:
    """Test team management utility functions."""

    def test_validate_team_name(self):
        """Test team name validation."""
        assert validate_team_name("team_a") is True
        assert validate_team_name("team_b") is True
        assert validate_team_name("team_c") is False
        assert validate_team_name("invalid") is False
        assert validate_team_name("") is False

    def test_get_enemy_team(self):
        """Test getting enemy team name."""
        assert get_enemy_team("team_a") == "team_b"
        assert get_enemy_team("team_b") == "team_a"

        with pytest.raises(ValueError):
            get_enemy_team("invalid_team")

    def test_calculate_team_stats(self):
        """Test team statistics calculation."""
        agents = {
            "team_a_agent_1": Agent(
                agent_id="team_a_agent_1",
                team="team_a",
                position=(0, 0),
                health=100,
                status=AgentStatus.ALIVE,
            ),
            "team_a_agent_2": Agent(
                agent_id="team_a_agent_2",
                team="team_a",
                position=(10, 0),
                health=50,
                status=AgentStatus.ALIVE,
            ),
            "team_a_agent_3": Agent(
                agent_id="team_a_agent_3",
                team="team_a",
                position=(20, 0),
                health=0,
                status=AgentStatus.DEAD,
            ),
        }

        world_state = BattleWorldState(agents=agents)
        world_state.team_scores["team_a"] = 5
        world_state.add_team_message("team_a_agent_1", "Message 1")
        world_state.add_team_message("team_a_agent_1", "Message 2")

        stats = calculate_team_stats(world_state, "team_a")

        assert stats["total_agents"] == 3
        assert stats["alive_agents"] == 2
        assert stats["dead_agents"] == 1
        assert stats["total_health"] == 150
        assert stats["average_health"] == 50.0
        assert stats["team_score"] == 5
        assert stats["recent_messages"] == 2
        assert len(stats["weapon_conditions"]) == 3
