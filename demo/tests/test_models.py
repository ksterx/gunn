"""
Tests for battle demo data models.

This module contains comprehensive tests for Agent, MapLocation, BattleWorldState,
and TeamCommunication models, including validation, serialization, and team management.
"""

import pytest
from pydantic import ValidationError

from demo.shared.enums import AgentStatus, LocationType, WeaponCondition
from demo.shared.models import Agent, BattleWorldState, MapLocation, TeamCommunication


class TestAgent:
    """Test cases for the Agent model."""

    def test_agent_creation_valid(self):
        """Test creating a valid agent."""
        agent = Agent(agent_id="team_a_agent_1", team="team_a", position=(50.0, 50.0))

        assert agent.agent_id == "team_a_agent_1"
        assert agent.team == "team_a"
        assert agent.position == (50.0, 50.0)
        assert agent.health == 100
        assert agent.status == AgentStatus.ALIVE
        assert agent.weapon_condition == WeaponCondition.EXCELLENT
        assert agent.is_alive()
        assert agent.can_attack()

    def test_agent_id_validation(self):
        """Test agent ID validation."""
        # Valid agent IDs
        Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0))
        Agent(agent_id="team_b_agent_2", team="team_b", position=(0, 0))

        # Invalid agent IDs
        with pytest.raises(ValidationError):
            Agent(agent_id="", team="team_a", position=(0, 0))

        with pytest.raises(ValidationError):
            Agent(agent_id="invalid_agent", team="team_a", position=(0, 0))

        with pytest.raises(ValidationError):
            Agent(agent_id="team_c_agent_1", team="team_a", position=(0, 0))

    def test_team_consistency_validation(self):
        """Test that agent ID is consistent with team assignment."""
        # Valid combinations
        Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0))
        Agent(agent_id="team_b_agent_1", team="team_b", position=(0, 0))

        # Invalid combinations
        with pytest.raises(ValidationError):
            Agent(agent_id="team_a_agent_1", team="team_b", position=(0, 0))

        with pytest.raises(ValidationError):
            Agent(agent_id="team_b_agent_1", team="team_a", position=(0, 0))

    def test_position_validation(self):
        """Test position coordinate validation."""
        # Valid positions
        Agent(agent_id="team_a_agent_1", team="team_a", position=(0.0, 0.0))
        Agent(agent_id="team_a_agent_1", team="team_a", position=(100.0, 200.0))

        # Invalid positions
        with pytest.raises(ValidationError):
            Agent(agent_id="team_a_agent_1", team="team_a", position=(-1.0, 0.0))

        with pytest.raises(ValidationError):
            Agent(agent_id="team_a_agent_1", team="team_a", position=(0.0, -1.0))

        with pytest.raises(ValidationError):
            Agent(agent_id="team_a_agent_1", team="team_a", position=(1001.0, 0.0))

    def test_health_validation(self):
        """Test health value validation."""
        # Valid health values
        Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0), health=0)
        Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0), health=50)
        Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0), health=100)

        # Invalid health values
        with pytest.raises(ValidationError):
            Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0), health=-1)

        with pytest.raises(ValidationError):
            Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0), health=101)

    def test_agent_status_methods(self):
        """Test agent status checking methods."""
        # Alive agent
        agent = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(0, 0),
            health=50,
            status=AgentStatus.ALIVE,
        )
        assert agent.is_alive()
        assert agent.can_attack()

        # Dead agent
        agent.status = AgentStatus.DEAD
        assert not agent.is_alive()
        assert not agent.can_attack()

        # Agent with broken weapon
        agent.status = AgentStatus.ALIVE
        agent.weapon_condition = WeaponCondition.BROKEN
        assert agent.is_alive()
        assert not agent.can_attack()

        # Agent with zero health
        agent.health = 0
        agent.weapon_condition = WeaponCondition.EXCELLENT
        assert not agent.is_alive()
        assert not agent.can_attack()

    def test_get_teammates(self):
        """Test getting teammate agent IDs."""
        agents = {
            "team_a_agent_1": Agent(
                agent_id="team_a_agent_1", team="team_a", position=(0, 0)
            ),
            "team_a_agent_2": Agent(
                agent_id="team_a_agent_2", team="team_a", position=(10, 0)
            ),
            "team_b_agent_1": Agent(
                agent_id="team_b_agent_1", team="team_b", position=(20, 0)
            ),
        }

        agent = agents["team_a_agent_1"]
        teammates = agent.get_teammates(agents)

        assert teammates == ["team_a_agent_2"]
        assert "team_a_agent_1" not in teammates  # Should not include self
        assert "team_b_agent_1" not in teammates  # Should not include enemies

    def test_get_enemies(self):
        """Test getting enemy agent IDs."""
        agents = {
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

        agent = agents["team_a_agent_1"]
        enemies = agent.get_enemies(agents)

        assert set(enemies) == {"team_b_agent_1", "team_b_agent_2"}
        assert "team_a_agent_1" not in enemies  # Should not include self
        assert "team_a_agent_2" not in enemies  # Should not include teammates


class TestMapLocation:
    """Test cases for the MapLocation model."""

    def test_map_location_creation(self):
        """Test creating a valid map location."""
        location = MapLocation(
            position=(100.0, 50.0),
            location_type=LocationType.FORGE,
            radius=10.0,
            metadata={"team": "team_a"},
        )

        assert location.position == (100.0, 50.0)
        assert location.location_type == LocationType.FORGE
        assert location.radius == 10.0
        assert location.metadata["team"] == "team_a"

    def test_position_validation(self):
        """Test map location position validation."""
        # Valid positions
        MapLocation(position=(0.0, 0.0), location_type=LocationType.FORGE)
        MapLocation(position=(500.0, 300.0), location_type=LocationType.COVER)

        # Invalid positions
        with pytest.raises(ValidationError):
            MapLocation(position=(-1.0, 0.0), location_type=LocationType.FORGE)

        with pytest.raises(ValidationError):
            MapLocation(position=(1001.0, 0.0), location_type=LocationType.FORGE)

    def test_radius_validation(self):
        """Test radius validation."""
        # Valid radius
        MapLocation(position=(0, 0), location_type=LocationType.FORGE, radius=5.0)

        # Invalid radius
        with pytest.raises(ValidationError):
            MapLocation(position=(0, 0), location_type=LocationType.FORGE, radius=0.0)

        with pytest.raises(ValidationError):
            MapLocation(position=(0, 0), location_type=LocationType.FORGE, radius=-1.0)

    def test_is_agent_in_range(self):
        """Test checking if agent is in range of location."""
        location = MapLocation(
            position=(50.0, 50.0), location_type=LocationType.FORGE, radius=10.0
        )

        # Agent within range
        assert location.is_agent_in_range((55.0, 55.0))
        assert location.is_agent_in_range((50.0, 60.0))  # Exactly at radius

        # Agent outside range
        assert not location.is_agent_in_range((70.0, 70.0))
        assert not location.is_agent_in_range((50.0, 61.0))  # Just outside radius


class TestTeamCommunication:
    """Test cases for the TeamCommunication model."""

    def test_team_communication_creation(self):
        """Test creating a valid team communication."""
        comm = TeamCommunication(
            sender_id="team_a_agent_1",
            team="team_a",
            message="Enemy spotted at coordinates (100, 50)",
            urgency="high",
            timestamp=123.45,
        )

        assert comm.sender_id == "team_a_agent_1"
        assert comm.team == "team_a"
        assert comm.message == "Enemy spotted at coordinates (100, 50)"
        assert comm.urgency == "high"
        assert comm.timestamp == 123.45

    def test_message_validation(self):
        """Test message content validation."""
        # Valid messages
        TeamCommunication(
            sender_id="team_a_agent_1",
            team="team_a",
            message="Valid message",
            timestamp=0.0,
        )

        # Invalid messages
        with pytest.raises(ValidationError):
            TeamCommunication(
                sender_id="team_a_agent_1", team="team_a", message="", timestamp=0.0
            )

        with pytest.raises(ValidationError):
            TeamCommunication(
                sender_id="team_a_agent_1",
                team="team_a",
                message="   ",  # Only whitespace
                timestamp=0.0,
            )

        # Message too long
        with pytest.raises(ValidationError):
            TeamCommunication(
                sender_id="team_a_agent_1",
                team="team_a",
                message="x" * 201,  # Exceeds 200 character limit
                timestamp=0.0,
            )


class TestBattleWorldState:
    """Test cases for the BattleWorldState model."""

    def test_world_state_creation(self):
        """Test creating a valid world state."""
        world_state = BattleWorldState()

        assert world_state.agents == {}
        assert world_state.map_locations == {}
        assert world_state.team_scores == {"team_a": 0, "team_b": 0}
        assert world_state.game_time == 0.0
        assert world_state.game_status == "active"
        assert world_state.team_communications == {"team_a": [], "team_b": []}

    def test_agent_consistency_validation(self):
        """Test that agent keys match agent IDs."""
        agent = Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0))

        # Valid - key matches agent_id
        world_state = BattleWorldState(agents={"team_a_agent_1": agent})
        assert len(world_state.agents) == 1

        # Invalid - key doesn't match agent_id
        with pytest.raises(ValidationError):
            BattleWorldState(agents={"wrong_key": agent})

    def test_get_team_agents(self):
        """Test getting agents by team."""
        agents = {
            "team_a_agent_1": Agent(
                agent_id="team_a_agent_1", team="team_a", position=(0, 0)
            ),
            "team_a_agent_2": Agent(
                agent_id="team_a_agent_2", team="team_a", position=(10, 0)
            ),
            "team_b_agent_1": Agent(
                agent_id="team_b_agent_1", team="team_b", position=(20, 0)
            ),
        }

        world_state = BattleWorldState(agents=agents)

        team_a_agents = world_state.get_team_agents("team_a")
        assert len(team_a_agents) == 2
        assert "team_a_agent_1" in team_a_agents
        assert "team_a_agent_2" in team_a_agents

        team_b_agents = world_state.get_team_agents("team_b")
        assert len(team_b_agents) == 1
        assert "team_b_agent_1" in team_b_agents

    def test_get_alive_agents(self):
        """Test getting alive agents."""
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
                health=0,
                status=AgentStatus.DEAD,
            ),
            "team_b_agent_1": Agent(
                agent_id="team_b_agent_1",
                team="team_b",
                position=(20, 0),
                health=50,
                status=AgentStatus.ALIVE,
            ),
        }

        world_state = BattleWorldState(agents=agents)

        # All alive agents
        alive_agents = world_state.get_alive_agents()
        assert len(alive_agents) == 2
        assert "team_a_agent_1" in alive_agents
        assert "team_b_agent_1" in alive_agents
        assert "team_a_agent_2" not in alive_agents

        # Alive agents from specific team
        team_a_alive = world_state.get_alive_agents("team_a")
        assert len(team_a_alive) == 1
        assert "team_a_agent_1" in team_a_alive

    def test_add_team_message(self):
        """Test adding team messages."""
        agent = Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0))
        world_state = BattleWorldState(
            agents={"team_a_agent_1": agent}, game_time=100.0
        )

        world_state.add_team_message("team_a_agent_1", "Test message", "high")

        messages = world_state.team_communications["team_a"]
        assert len(messages) == 1
        assert messages[0].sender_id == "team_a_agent_1"
        assert messages[0].message == "Test message"
        assert messages[0].urgency == "high"
        assert messages[0].timestamp == 100.0

        # Test message limit (should keep only last 50)
        for i in range(55):
            world_state.add_team_message("team_a_agent_1", f"Message {i}")

        messages = world_state.team_communications["team_a"]
        assert len(messages) == 50  # Should be capped at 50

    def test_get_recent_team_messages(self):
        """Test getting recent team messages."""
        agent = Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0))
        world_state = BattleWorldState(agents={"team_a_agent_1": agent})

        # Add multiple messages
        for i in range(15):
            world_state.add_team_message("team_a_agent_1", f"Message {i}")

        # Get recent messages
        recent = world_state.get_recent_team_messages("team_a", 5)
        assert len(recent) == 5
        assert recent[-1].message == "Message 14"  # Most recent
        assert recent[0].message == "Message 10"  # 5th most recent

    def test_check_win_condition(self):
        """Test win condition checking."""
        # Active game - both teams have alive agents
        agents = {
            "team_a_agent_1": Agent(
                agent_id="team_a_agent_1",
                team="team_a",
                position=(0, 0),
                health=100,
                status=AgentStatus.ALIVE,
            ),
            "team_b_agent_1": Agent(
                agent_id="team_b_agent_1",
                team="team_b",
                position=(20, 0),
                health=50,
                status=AgentStatus.ALIVE,
            ),
        }
        world_state = BattleWorldState(agents=agents)
        assert world_state.check_win_condition() == "active"

        # Team A wins - Team B has no alive agents
        agents["team_b_agent_1"].status = AgentStatus.DEAD
        world_state = BattleWorldState(agents=agents)
        assert world_state.check_win_condition() == "team_a_wins"

        # Team B wins - Team A has no alive agents
        agents["team_a_agent_1"].status = AgentStatus.DEAD
        agents["team_b_agent_1"].status = AgentStatus.ALIVE
        world_state = BattleWorldState(agents=agents)
        assert world_state.check_win_condition() == "team_b_wins"

        # Draw - No agents alive
        agents["team_b_agent_1"].status = AgentStatus.DEAD
        world_state = BattleWorldState(agents=agents)
        assert world_state.check_win_condition() == "draw"

    def test_update_game_status(self):
        """Test automatic game status updates."""
        agents = {
            "team_a_agent_1": Agent(
                agent_id="team_a_agent_1",
                team="team_a",
                position=(0, 0),
                health=100,
                status=AgentStatus.ALIVE,
            ),
            "team_b_agent_1": Agent(
                agent_id="team_b_agent_1",
                team="team_b",
                position=(20, 0),
                health=0,
                status=AgentStatus.DEAD,
            ),
        }

        world_state = BattleWorldState(agents=agents)
        world_state.update_game_status()

        assert world_state.game_status == "team_a_wins"

    def test_get_prioritized_team_messages(self):
        """Test getting prioritized team messages."""
        agent = Agent(agent_id="team_a_agent_1", team="team_a", position=(0, 0))
        world_state = BattleWorldState(agents={"team_a_agent_1": agent})

        # Add messages with different urgencies and timestamps
        messages_data = [
            ("Low priority old", "low", 10.0),
            ("High priority old", "high", 15.0),
            ("Medium priority", "medium", 20.0),
            ("High priority new", "high", 25.0),
            ("Low priority new", "low", 30.0),
        ]

        for message, urgency, timestamp in messages_data:
            world_state.game_time = timestamp
            world_state.add_team_message("team_a_agent_1", message, urgency)

        # Get prioritized messages
        prioritized = world_state.get_prioritized_team_messages("team_a", 5)

        # Should be sorted by urgency (high first) then by timestamp (newest first)
        assert len(prioritized) == 5

        # High priority messages should come first
        assert prioritized[0].message == "High priority new"
        assert prioritized[0].urgency == "high"
        assert prioritized[1].message == "High priority old"
        assert prioritized[1].urgency == "high"

        # Then medium priority
        assert prioritized[2].message == "Medium priority"
        assert prioritized[2].urgency == "medium"

        # Then low priority (newest first)
        assert prioritized[3].message == "Low priority new"
        assert prioritized[3].urgency == "low"
        assert prioritized[4].message == "Low priority old"
        assert prioritized[4].urgency == "low"

    def test_get_prioritized_team_messages_empty(self):
        """Test getting prioritized messages for team with no messages."""
        world_state = BattleWorldState()
        prioritized = world_state.get_prioritized_team_messages("team_a", 10)
        assert len(prioritized) == 0
