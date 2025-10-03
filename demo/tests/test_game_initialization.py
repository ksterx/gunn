"""
Tests for game initialization and auto-start system.

This module tests the comprehensive game initialization capabilities including
automatic team creation, strategic positioning, forge placement, deterministic
setup, and restart functionality.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from demo.backend.game_initializer import (
    AutoStartManager,
    GameInitializer,
    InitializationConfig,
)
from demo.shared.enums import AgentStatus, LocationType, WeaponCondition
from demo.shared.models import Agent, BattleWorldState


class TestInitializationConfig:
    """Test initialization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InitializationConfig()

        assert config.agents_per_team == 3
        assert config.team_names == ["team_a", "team_b"]
        assert config.map_width == 200.0
        assert config.map_height == 100.0
        assert config.positioning_strategy == "corners"
        assert config.forge_placement == "corners"
        assert config.initial_health == 100
        assert config.initial_weapon_condition == WeaponCondition.EXCELLENT
        assert config.use_random_seed is True
        assert config.position_jitter == 2.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = InitializationConfig(
            agents_per_team=5,
            team_names=["red", "blue"],
            positioning_strategy="random",
            forge_placement="sides",
            use_random_seed=False,
            random_seed=123,
        )

        assert config.agents_per_team == 5
        assert config.team_names == ["red", "blue"]
        assert config.positioning_strategy == "random"
        assert config.forge_placement == "sides"
        assert config.use_random_seed is False
        assert config.random_seed == 123


class TestGameInitializer:
    """Test game initializer functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return InitializationConfig(
            agents_per_team=2,  # Smaller for testing
            use_random_seed=True,
            random_seed=42,  # Deterministic for testing
            position_jitter=0.0,  # No jitter for predictable positions
        )

    @pytest.fixture
    def initializer(self, config):
        """Create game initializer."""
        return GameInitializer(config)

    @pytest.fixture
    def world_state(self):
        """Create empty world state."""
        return BattleWorldState()

    @pytest.mark.asyncio
    async def test_initialization_basic(self, initializer, world_state):
        """Test basic game initialization."""
        result = await initializer.initialize_game(world_state)

        # Check that world state was returned
        assert result is world_state

        # Check agents were created
        assert len(world_state.agents) == 4  # 2 per team

        # Check teams are balanced
        team_a_count = sum(
            1 for agent in world_state.agents.values() if agent.team == "team_a"
        )
        team_b_count = sum(
            1 for agent in world_state.agents.values() if agent.team == "team_b"
        )
        assert team_a_count == 2
        assert team_b_count == 2

        # Check forges were created
        forges = [
            loc
            for loc in world_state.map_locations.values()
            if loc.location_type == LocationType.FORGE
        ]
        assert len(forges) == 2

        # Check game metadata
        assert world_state.game_time == 0.0
        assert world_state.game_status == "active"
        assert "team_a" in world_state.team_scores
        assert "team_b" in world_state.team_scores

    @pytest.mark.asyncio
    async def test_agent_properties(self, initializer, world_state):
        """Test that agents are created with correct properties."""
        await initializer.initialize_game(world_state)

        for agent in world_state.agents.values():
            # Check basic properties
            assert agent.health == 100
            assert agent.status == AgentStatus.ALIVE
            assert agent.weapon_condition == WeaponCondition.EXCELLENT
            assert agent.last_action_time == 0.0

            # Check team consistency
            if agent.team == "team_a":
                assert agent.agent_id.startswith("team_a_")
            else:
                assert agent.agent_id.startswith("team_b_")

            # Check position bounds
            x, y = agent.position
            assert 0 <= x <= 200.0
            assert 0 <= y <= 100.0

    @pytest.mark.asyncio
    async def test_forge_placement_corners(self, world_state):
        """Test forge placement with corners strategy."""
        config = InitializationConfig(forge_placement="corners", position_jitter=0.0)
        initializer = GameInitializer(config)

        await initializer.initialize_game(world_state)

        forges = {
            loc_id: loc
            for loc_id, loc in world_state.map_locations.items()
            if loc.location_type == LocationType.FORGE
        }

        assert len(forges) == 2

        # Check forge positions (should be in corners)
        forge_positions = [forge.position for forge in forges.values()]

        # One forge should be near top-left, one near bottom-right
        top_left_forge = min(forge_positions, key=lambda pos: pos[0] + pos[1])
        bottom_right_forge = max(forge_positions, key=lambda pos: pos[0] + pos[1])

        assert top_left_forge[0] < 50  # Left side
        assert top_left_forge[1] > 50  # Top side
        assert bottom_right_forge[0] > 150  # Right side
        assert bottom_right_forge[1] < 50  # Bottom side

    @pytest.mark.asyncio
    async def test_forge_placement_sides(self, world_state):
        """Test forge placement with sides strategy."""
        config = InitializationConfig(forge_placement="sides", position_jitter=0.0)
        initializer = GameInitializer(config)

        await initializer.initialize_game(world_state)

        forges = [
            loc
            for loc in world_state.map_locations.values()
            if loc.location_type == LocationType.FORGE
        ]

        assert len(forges) == 2

        # Check that forges are on opposite sides
        forge_positions = [forge.position for forge in forges]
        left_forge = min(forge_positions, key=lambda pos: pos[0])
        right_forge = max(forge_positions, key=lambda pos: pos[0])

        assert left_forge[0] < 50  # Left side
        assert right_forge[0] > 150  # Right side

    @pytest.mark.asyncio
    async def test_positioning_strategy_corners(self, world_state):
        """Test agent positioning with corners strategy."""
        config = InitializationConfig(
            positioning_strategy="corners", position_jitter=0.0, agents_per_team=3
        )
        initializer = GameInitializer(config)

        await initializer.initialize_game(world_state)

        team_a_agents = [
            agent for agent in world_state.agents.values() if agent.team == "team_a"
        ]
        team_b_agents = [
            agent for agent in world_state.agents.values() if agent.team == "team_b"
        ]

        # Team A should be in top area
        for agent in team_a_agents:
            assert agent.position[1] > 50  # Top half

        # Team B should be in bottom area
        for agent in team_b_agents:
            assert agent.position[1] < 50  # Bottom half

    @pytest.mark.asyncio
    async def test_positioning_strategy_lines(self, world_state):
        """Test agent positioning with lines strategy."""
        config = InitializationConfig(
            positioning_strategy="lines", position_jitter=0.0, agents_per_team=3
        )
        initializer = GameInitializer(config)

        await initializer.initialize_game(world_state)

        team_a_agents = [
            agent for agent in world_state.agents.values() if agent.team == "team_a"
        ]
        team_b_agents = [
            agent for agent in world_state.agents.values() if agent.team == "team_b"
        ]

        # Agents should be arranged in lines
        team_a_y_positions = [agent.position[1] for agent in team_a_agents]
        team_b_y_positions = [agent.position[1] for agent in team_b_agents]

        # All team A agents should have similar Y coordinates (in a line)
        assert max(team_a_y_positions) - min(team_a_y_positions) < 5.0

        # All team B agents should have similar Y coordinates (in a line)
        assert max(team_b_y_positions) - min(team_b_y_positions) < 5.0

        # Teams should be on different sides
        avg_team_a_y = sum(team_a_y_positions) / len(team_a_y_positions)
        avg_team_b_y = sum(team_b_y_positions) / len(team_b_y_positions)
        assert abs(avg_team_a_y - avg_team_b_y) > 30.0

    @pytest.mark.asyncio
    async def test_deterministic_initialization(self, world_state):
        """Test that initialization is deterministic with same seed."""
        config1 = InitializationConfig(
            use_random_seed=True,
            random_seed=123,
            position_jitter=1.0,  # Enable jitter to test randomness
        )
        config2 = InitializationConfig(
            use_random_seed=True,
            random_seed=123,
            position_jitter=1.0,  # Same jitter
        )

        initializer1 = GameInitializer(config1)
        initializer2 = GameInitializer(config2)

        world_state1 = BattleWorldState()
        world_state2 = BattleWorldState()

        await initializer1.initialize_game(world_state1)
        await initializer2.initialize_game(world_state2)

        # Agent positions should be identical (within floating point precision)
        for agent_id in world_state1.agents:
            assert agent_id in world_state2.agents
            pos1 = world_state1.agents[agent_id].position
            pos2 = world_state2.agents[agent_id].position
            assert abs(pos1[0] - pos2[0]) < 0.001
            assert abs(pos1[1] - pos2[1]) < 0.001

    @pytest.mark.asyncio
    async def test_reset_existing_state(self, initializer, world_state):
        """Test that existing state is properly reset."""
        # Add some existing data
        world_state.agents["team_a_old_agent"] = Agent(
            agent_id="team_a_old_agent", team="team_a", position=(0.0, 0.0)
        )
        world_state.game_time = 100.0
        world_state.game_status = "team_a_wins"

        await initializer.initialize_game(world_state, reset_existing=True)

        # Old data should be cleared
        assert "team_a_old_agent" not in world_state.agents
        assert world_state.game_time == 0.0
        assert world_state.game_status == "active"

        # New data should be present
        assert len(world_state.agents) == 4  # 2 per team

    @pytest.mark.asyncio
    async def test_validation_failure(self, world_state):
        """Test that validation catches initialization errors."""
        # Create config that will cause validation failure
        config = InitializationConfig(agents_per_team=0)  # Invalid
        initializer = GameInitializer(config)

        with pytest.raises(ValueError, match="agents_per_team must be positive"):
            await initializer.initialize_game(world_state)

    def test_initialization_summary(self, initializer, world_state):
        """Test initialization summary generation."""
        # Initialize first
        asyncio.run(initializer.initialize_game(world_state))

        summary = initializer.get_initialization_summary(world_state)

        assert "initialization_count" in summary
        assert summary["total_agents"] == 4
        assert summary["team_counts"]["team_a"] == 2
        assert summary["team_counts"]["team_b"] == 2
        assert summary["total_forges"] == 2
        assert summary["game_status"] == "active"
        assert summary["config"]["positioning_strategy"] == "corners"


class TestAutoStartManager:
    """Test auto-start manager functionality."""

    @pytest.fixture
    def game_initializer(self):
        """Create mock game initializer."""
        config = InitializationConfig(agents_per_team=2)
        return GameInitializer(config)

    @pytest.fixture
    def auto_start_manager(self, game_initializer):
        """Create auto-start manager."""
        return AutoStartManager(game_initializer)

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = Mock()
        orchestrator.world_state = BattleWorldState()
        orchestrator.initialize = AsyncMock()
        return orchestrator

    @pytest.fixture
    def mock_ai_decision_maker(self):
        """Create mock AI decision maker."""
        return Mock()

    @pytest.mark.asyncio
    async def test_auto_start_success(
        self, auto_start_manager, mock_orchestrator, mock_ai_decision_maker
    ):
        """Test successful auto-start."""
        success = await auto_start_manager.auto_start_game(
            mock_orchestrator, mock_ai_decision_maker, max_attempts=1
        )

        assert success is True
        mock_orchestrator.initialize.assert_called_once_with(mock_ai_decision_maker)

    @pytest.mark.asyncio
    async def test_auto_start_retry_logic(
        self, auto_start_manager, mock_orchestrator, mock_ai_decision_maker
    ):
        """Test auto-start retry logic on failure."""
        # Make initialize fail twice, then succeed
        mock_orchestrator.initialize.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            None,  # Success on third attempt
        ]

        success = await auto_start_manager.auto_start_game(
            mock_orchestrator, mock_ai_decision_maker, max_attempts=3
        )

        assert success is True
        assert mock_orchestrator.initialize.call_count == 3

    @pytest.mark.asyncio
    async def test_auto_start_max_attempts_exceeded(
        self, auto_start_manager, mock_orchestrator, mock_ai_decision_maker
    ):
        """Test auto-start failure when max attempts exceeded."""
        # Make initialize always fail
        mock_orchestrator.initialize.side_effect = Exception("Always fails")

        success = await auto_start_manager.auto_start_game(
            mock_orchestrator, mock_ai_decision_maker, max_attempts=2
        )

        assert success is False
        assert mock_orchestrator.initialize.call_count == 2

    @pytest.mark.asyncio
    async def test_restart_game(
        self, auto_start_manager, mock_orchestrator, mock_ai_decision_maker
    ):
        """Test game restart functionality."""
        success = await auto_start_manager.restart_game(
            mock_orchestrator, mock_ai_decision_maker, reason="test_restart"
        )

        assert success is True
        mock_orchestrator.initialize.assert_called()

    def test_enable_disable_auto_start(self, auto_start_manager):
        """Test enabling and disabling auto-start."""
        # Initially enabled
        assert auto_start_manager._auto_start_enabled is True

        # Disable
        auto_start_manager.disable_auto_start()
        assert auto_start_manager._auto_start_enabled is False

        # Enable
        auto_start_manager.enable_auto_start()
        assert auto_start_manager._auto_start_enabled is True

    @pytest.mark.asyncio
    async def test_auto_start_disabled(
        self, auto_start_manager, mock_orchestrator, mock_ai_decision_maker
    ):
        """Test that auto-start respects disabled state."""
        auto_start_manager.disable_auto_start()

        success = await auto_start_manager.auto_start_game(
            mock_orchestrator, mock_ai_decision_maker
        )

        assert success is False
        mock_orchestrator.initialize.assert_not_called()

    def test_restart_statistics(self, auto_start_manager):
        """Test restart statistics tracking."""
        stats = auto_start_manager.get_restart_statistics()

        assert "restart_count" in stats
        assert "auto_start_enabled" in stats
        assert "max_restart_attempts" in stats
        assert stats["restart_count"] == 0
        assert stats["auto_start_enabled"] is True


class TestInitializationIntegration:
    """Integration tests for initialization system."""

    @pytest.mark.asyncio
    async def test_full_initialization_flow(self):
        """Test complete initialization flow."""
        # Create components
        config = InitializationConfig(
            agents_per_team=2,
            positioning_strategy="corners",
            forge_placement="corners",
            use_random_seed=True,
            random_seed=42,
        )

        initializer = GameInitializer(config)
        auto_start_manager = AutoStartManager(initializer)
        world_state = BattleWorldState()

        # Mock orchestrator and AI decision maker
        mock_orchestrator = Mock()
        mock_orchestrator.world_state = world_state
        mock_orchestrator.initialize = AsyncMock()

        mock_ai_decision_maker = Mock()

        # Test full flow
        success = await auto_start_manager.auto_start_game(
            mock_orchestrator, mock_ai_decision_maker
        )

        assert success is True

        # Verify world state was initialized
        assert len(world_state.agents) == 4
        assert len(world_state.map_locations) >= 2  # At least 2 forges
        assert world_state.game_status == "active"

        # Verify orchestrator was initialized
        mock_orchestrator.initialize.assert_called_once_with(mock_ai_decision_maker)

    @pytest.mark.asyncio
    async def test_multiple_restarts(self):
        """Test multiple game restarts."""
        config = InitializationConfig(agents_per_team=1)
        initializer = GameInitializer(config)
        auto_start_manager = AutoStartManager(initializer)

        mock_orchestrator = Mock()
        mock_orchestrator.world_state = BattleWorldState()
        mock_orchestrator.initialize = AsyncMock()
        mock_ai_decision_maker = Mock()

        # Perform multiple restarts
        for i in range(3):
            success = await auto_start_manager.restart_game(
                mock_orchestrator, mock_ai_decision_maker, reason=f"restart_{i}"
            )
            assert success is True

        # Check restart count
        stats = auto_start_manager.get_restart_statistics()
        assert stats["restart_count"] == 3

    @pytest.mark.asyncio
    async def test_custom_team_configuration(self):
        """Test initialization with default team configuration (Agent model only supports team_a/team_b)."""
        config = InitializationConfig(
            agents_per_team=3,
            team_names=["team_a", "team_b"],  # Use supported team names
            positioning_strategy="lines",
        )

        initializer = GameInitializer(config)
        world_state = BattleWorldState()

        await initializer.initialize_game(world_state)

        # Check team names
        teams = set(agent.team for agent in world_state.agents.values())
        assert teams == {"team_a", "team_b"}

        # Check agent counts
        team_a_count = sum(
            1 for agent in world_state.agents.values() if agent.team == "team_a"
        )
        team_b_count = sum(
            1 for agent in world_state.agents.values() if agent.team == "team_b"
        )
        assert team_a_count == 3
        assert team_b_count == 3
