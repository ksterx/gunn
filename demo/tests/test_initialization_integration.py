"""
Integration test for the complete initialization system.

This test verifies that the game initialization and auto-start system
works end-to-end with the Gunn integration layer.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from demo.backend.ai_decision import AIDecisionMaker
from demo.backend.game_initializer import (
    AutoStartManager,
    GameInitializer,
    InitializationConfig,
)
from demo.backend.gunn_integration import BattleOrchestrator
from demo.shared.models import BattleWorldState


class TestCompleteInitializationFlow:
    """Test complete initialization flow integration."""

    @pytest.fixture
    def initialization_config(self):
        """Create test initialization configuration."""
        return InitializationConfig(
            agents_per_team=2,  # Smaller for testing
            positioning_strategy="corners",
            forge_placement="corners",
            use_random_seed=True,
            random_seed=42,
            position_jitter=0.0,  # No jitter for predictable results
        )

    @pytest.fixture
    def game_initializer(self, initialization_config):
        """Create game initializer."""
        return GameInitializer(initialization_config)

    @pytest.fixture
    def auto_start_manager(self, game_initializer):
        """Create auto-start manager."""
        return AutoStartManager(game_initializer)

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock battle orchestrator."""
        orchestrator = Mock(spec=BattleOrchestrator)
        orchestrator.world_state = BattleWorldState()
        orchestrator.initialize = AsyncMock()
        orchestrator._register_agents_with_gunn = AsyncMock()
        orchestrator._sync_world_state = AsyncMock()
        orchestrator._initialized = False
        return orchestrator

    @pytest.fixture
    def mock_ai_decision_maker(self):
        """Create mock AI decision maker."""
        return Mock(spec=AIDecisionMaker)

    @pytest.mark.asyncio
    async def test_complete_initialization_flow(
        self,
        game_initializer,
        auto_start_manager,
        mock_orchestrator,
        mock_ai_decision_maker,
    ):
        """Test the complete initialization flow from start to finish."""

        # Step 1: Initialize the game world
        await game_initializer.initialize_game(mock_orchestrator.world_state)

        # Verify world state was properly initialized
        world_state = mock_orchestrator.world_state
        assert len(world_state.agents) == 4  # 2 per team
        assert len(world_state.map_locations) >= 2  # At least 2 forges
        assert world_state.game_status == "active"
        assert world_state.game_time == 0.0

        # Verify team balance
        team_a_count = sum(
            1 for agent in world_state.agents.values() if agent.team == "team_a"
        )
        team_b_count = sum(
            1 for agent in world_state.agents.values() if agent.team == "team_b"
        )
        assert team_a_count == 2
        assert team_b_count == 2

        # Step 2: Use auto-start manager to complete initialization
        success = await auto_start_manager.auto_start_game(
            mock_orchestrator, mock_ai_decision_maker
        )

        assert success is True

        # Verify orchestrator was initialized
        mock_orchestrator.initialize.assert_called_once_with(mock_ai_decision_maker)

    @pytest.mark.asyncio
    async def test_initialization_with_different_strategies(
        self, mock_orchestrator, mock_ai_decision_maker
    ):
        """Test initialization with different positioning and forge strategies."""

        strategies = [
            ("corners", "corners"),
            ("lines", "sides"),
            ("corners", "center"),
        ]

        for positioning, forge_placement in strategies:
            # Create fresh world state for each test
            mock_orchestrator.world_state = BattleWorldState()

            config = InitializationConfig(
                agents_per_team=2,
                positioning_strategy=positioning,
                forge_placement=forge_placement,
                use_random_seed=True,
                random_seed=42,
                position_jitter=0.0,
            )

            initializer = GameInitializer(config)
            auto_start_manager = AutoStartManager(initializer)

            # Initialize and verify
            success = await auto_start_manager.auto_start_game(
                mock_orchestrator, mock_ai_decision_maker
            )

            assert success is True

            world_state = mock_orchestrator.world_state
            assert len(world_state.agents) == 4
            assert len(world_state.map_locations) >= 2

            # Verify positioning strategy was applied
            team_a_agents = [
                agent for agent in world_state.agents.values() if agent.team == "team_a"
            ]
            team_b_agents = [
                agent for agent in world_state.agents.values() if agent.team == "team_b"
            ]

            # Teams should be positioned differently
            team_a_positions = [agent.position for agent in team_a_agents]
            team_b_positions = [agent.position for agent in team_b_agents]

            # Calculate average Y positions to verify team separation
            avg_team_a_y = sum(pos[1] for pos in team_a_positions) / len(
                team_a_positions
            )
            avg_team_b_y = sum(pos[1] for pos in team_b_positions) / len(
                team_b_positions
            )

            # Teams should be separated (different average Y positions)
            assert abs(avg_team_a_y - avg_team_b_y) > 10.0

    @pytest.mark.asyncio
    async def test_restart_functionality(
        self,
        game_initializer,
        auto_start_manager,
        mock_orchestrator,
        mock_ai_decision_maker,
    ):
        """Test game restart functionality."""

        # Initial setup
        success = await auto_start_manager.auto_start_game(
            mock_orchestrator, mock_ai_decision_maker
        )
        assert success is True

        # Modify world state to simulate game progress
        mock_orchestrator.world_state.game_time = 100.0
        mock_orchestrator.world_state.game_status = "team_a_wins"

        # Restart the game
        success = await auto_start_manager.restart_game(
            mock_orchestrator, mock_ai_decision_maker, reason="test_restart"
        )

        assert success is True

        # Verify state was reset
        world_state = mock_orchestrator.world_state
        assert world_state.game_time == 0.0
        assert world_state.game_status == "active"
        assert len(world_state.agents) == 4

    @pytest.mark.asyncio
    async def test_initialization_error_recovery(
        self,
        game_initializer,
        auto_start_manager,
        mock_orchestrator,
        mock_ai_decision_maker,
    ):
        """Test error recovery during initialization."""

        # Make orchestrator initialization fail twice, then succeed
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

        # Verify world state was still properly initialized
        world_state = mock_orchestrator.world_state
        assert len(world_state.agents) == 4
        assert world_state.game_status == "active"

    def test_initialization_summary_accuracy(self, game_initializer):
        """Test that initialization summary provides accurate information."""

        world_state = BattleWorldState()

        # Initialize synchronously for this test
        asyncio.run(game_initializer.initialize_game(world_state))

        summary = game_initializer.get_initialization_summary(world_state)

        # Verify summary accuracy
        assert summary["total_agents"] == len(world_state.agents)
        assert summary["total_forges"] == sum(
            1
            for loc in world_state.map_locations.values()
            if loc.location_type.value == "forge"
        )
        assert summary["game_status"] == world_state.game_status
        assert summary["game_time"] == world_state.game_time

        # Verify team counts
        actual_team_counts = {}
        for agent in world_state.agents.values():
            actual_team_counts[agent.team] = actual_team_counts.get(agent.team, 0) + 1

        assert summary["team_counts"] == actual_team_counts

        # Verify config information
        config_info = summary["config"]
        assert (
            config_info["positioning_strategy"]
            == game_initializer.config.positioning_strategy
        )
        assert config_info["forge_placement"] == game_initializer.config.forge_placement
        assert config_info["agents_per_team"] == game_initializer.config.agents_per_team

    @pytest.mark.asyncio
    async def test_concurrent_initialization_safety(
        self, initialization_config, mock_ai_decision_maker
    ):
        """Test that concurrent initializations don't interfere with each other."""

        # Create multiple initializers and orchestrators
        initializers = [GameInitializer(initialization_config) for _ in range(3)]
        orchestrators = []

        for _ in range(3):
            orchestrator = Mock(spec=BattleOrchestrator)
            orchestrator.world_state = BattleWorldState()
            orchestrator.initialize = AsyncMock()
            orchestrators.append(orchestrator)

        # Run concurrent initializations
        tasks = []
        for i, (initializer, orchestrator) in enumerate(
            zip(initializers, orchestrators, strict=False)
        ):
            auto_start_manager = AutoStartManager(initializer)
            task = auto_start_manager.auto_start_game(
                orchestrator, mock_ai_decision_maker
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

        # Each should have properly initialized world state
        for orchestrator in orchestrators:
            world_state = orchestrator.world_state
            assert len(world_state.agents) == 4
            assert world_state.game_status == "active"
            orchestrator.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_deterministic_multi_initialization(self, mock_ai_decision_maker):
        """Test that multiple initializations with same seed produce identical results."""

        # Create two identical configurations
        config1 = InitializationConfig(
            agents_per_team=3,
            use_random_seed=True,
            random_seed=999,
            position_jitter=1.0,  # Enable jitter to test determinism
        )
        config2 = InitializationConfig(
            agents_per_team=3,
            use_random_seed=True,
            random_seed=999,
            position_jitter=1.0,
        )

        initializer1 = GameInitializer(config1)
        initializer2 = GameInitializer(config2)

        auto_start_manager1 = AutoStartManager(initializer1)
        auto_start_manager2 = AutoStartManager(initializer2)

        # Create separate orchestrators
        orchestrator1 = Mock(spec=BattleOrchestrator)
        orchestrator1.world_state = BattleWorldState()
        orchestrator1.initialize = AsyncMock()

        orchestrator2 = Mock(spec=BattleOrchestrator)
        orchestrator2.world_state = BattleWorldState()
        orchestrator2.initialize = AsyncMock()

        # Initialize both
        success1 = await auto_start_manager1.auto_start_game(
            orchestrator1, mock_ai_decision_maker
        )
        success2 = await auto_start_manager2.auto_start_game(
            orchestrator2, mock_ai_decision_maker
        )

        assert success1 is True
        assert success2 is True

        # Compare world states - they should be identical
        world_state1 = orchestrator1.world_state
        world_state2 = orchestrator2.world_state

        assert len(world_state1.agents) == len(world_state2.agents)

        # Agent positions should be identical
        for agent_id in world_state1.agents:
            assert agent_id in world_state2.agents
            pos1 = world_state1.agents[agent_id].position
            pos2 = world_state2.agents[agent_id].position
            assert abs(pos1[0] - pos2[0]) < 0.001
            assert abs(pos1[1] - pos2[1]) < 0.001
