"""
Integration tests for effect processing with battle orchestrator.

This module tests the integration between the effect processor and
the battle orchestrator, ensuring proper world state synchronization.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ..backend.effect_processor import EffectProcessor, GameStatusManager
from ..backend.gunn_integration import BattleOrchestrator
from ..shared.enums import AgentStatus, WeaponCondition
from ..shared.models import Agent, BattleWorldState


class TestEffectIntegration:
    """Test cases for effect processing integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = BattleOrchestrator()

        # Create test agents
        self.agent_a = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(10.0, 10.0),
            health=100,
            weapon_condition=WeaponCondition.EXCELLENT,
        )

        self.agent_b = Agent(
            agent_id="team_b_agent_1",
            team="team_b",
            position=(20.0, 20.0),
            health=80,
            weapon_condition=WeaponCondition.GOOD,
        )

        # Set up world state
        self.orchestrator.world_state = BattleWorldState(
            agents={"team_a_agent_1": self.agent_a, "team_b_agent_1": self.agent_b},
            game_time=10.0,
        )

        # Mock the Gunn orchestrator to avoid initialization complexity
        self.orchestrator.orchestrator = MagicMock()
        self.orchestrator.orchestrator.world_state = MagicMock()
        self.orchestrator._sync_world_state = AsyncMock()

    @pytest.mark.asyncio
    async def test_process_effects_integration(self):
        """Test processing effects through the orchestrator."""
        effects = [
            {
                "kind": "AgentDamaged",
                "payload": {
                    "target_id": "team_b_agent_1",
                    "damage": 30,
                    "new_health": 50,
                },
            },
            {
                "kind": "WeaponDegraded",
                "payload": {
                    "agent_id": "team_a_agent_1",
                    "old_condition": "excellent",
                    "new_condition": "good",
                },
            },
        ]

        result = await self.orchestrator.process_effects(effects)

        # Check processing results
        assert result["processed_count"] == 2
        assert result["failed_count"] == 0
        assert result["effect_types"]["AgentDamaged"] == 1
        assert result["effect_types"]["WeaponDegraded"] == 1

        # Check world state was updated
        assert self.agent_b.health == 50
        assert self.agent_a.weapon_condition == WeaponCondition.GOOD

        # Check that world state sync was called
        self.orchestrator._sync_world_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_effects_game_end(self):
        """Test processing effects that end the game."""
        effects = [
            {
                "kind": "AgentDied",
                "payload": {
                    "agent_id": "team_b_agent_1",
                    "killer_id": "team_a_agent_1",
                },
            }
        ]

        result = await self.orchestrator.process_effects(effects)

        # Game should end with team A victory
        assert result["game_status"] == "team_a_wins"
        assert result["status_changed"] is True

        # Agent should be marked as dead
        assert self.agent_b.status == AgentStatus.DEAD
        assert self.agent_b.health == 0

        # Team A should get a point
        assert self.orchestrator.world_state.team_scores["team_a"] == 1

    @pytest.mark.asyncio
    async def test_process_effects_empty_list(self):
        """Test processing empty effects list."""
        result = await self.orchestrator.process_effects([])

        assert result["processed_count"] == 0
        assert result["failed_count"] == 0
        assert result["game_status"] == "active"
        assert not result["status_changed"]

        # World state sync should not be called for empty effects
        self.orchestrator._sync_world_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_effects_team_communication(self):
        """Test processing team communication effects."""
        effects = [
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_a_agent_1",
                    "sender_team": "team_a",
                    "message": "Enemy spotted at coordinates (25, 30)!",
                    "urgency": "high",
                    "timestamp": 15.0,
                },
            }
        ]

        result = await self.orchestrator.process_effects(effects)

        assert result["processed_count"] == 1
        assert result["failed_count"] == 0

        # Check team communication was added
        team_messages = self.orchestrator.world_state.team_communications["team_a"]
        assert len(team_messages) == 1
        assert team_messages[0].message == "Enemy spotted at coordinates (25, 30)!"
        assert team_messages[0].urgency == "high"

        # Enemy team should not see the message
        enemy_messages = self.orchestrator.world_state.team_communications["team_b"]
        assert len(enemy_messages) == 0

    def test_get_game_statistics(self):
        """Test getting game statistics from orchestrator."""
        stats = self.orchestrator.get_game_statistics()

        assert stats["game_status"] == "active"
        assert stats["agents_alive"]["team_a"] == 1
        assert stats["agents_alive"]["team_b"] == 1
        assert stats["team_health"]["team_a"] == 100
        assert stats["team_health"]["team_b"] == 80
        assert stats["weapon_conditions"]["team_a"]["excellent"] == 1
        assert stats["weapon_conditions"]["team_b"]["good"] == 1

    @pytest.mark.asyncio
    async def test_effect_processor_initialization(self):
        """Test that effect processor is properly initialized."""
        assert isinstance(self.orchestrator.effect_processor, EffectProcessor)
        assert isinstance(self.orchestrator.game_status_manager, GameStatusManager)

        # Test that effect handlers are properly set up
        assert "AgentDamaged" in self.orchestrator.effect_processor._effect_handlers
        assert "AgentDied" in self.orchestrator.effect_processor._effect_handlers
        assert "WeaponDegraded" in self.orchestrator.effect_processor._effect_handlers
        assert "AgentHealed" in self.orchestrator.effect_processor._effect_handlers
        assert "WeaponRepaired" in self.orchestrator.effect_processor._effect_handlers
        assert "TeamMessage" in self.orchestrator.effect_processor._effect_handlers


class TestEffectProcessingConsistency:
    """Test effect processing consistency and state management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = EffectProcessor()
        self.manager = GameStatusManager()

        # Create a more complex world state
        self.agents = {}
        for team in ["team_a", "team_b"]:
            for i in range(2):
                agent_id = f"{team}_agent_{i + 1}"
                self.agents[agent_id] = Agent(
                    agent_id=agent_id,
                    team=team,
                    position=(10.0 + i * 10, 10.0),
                    health=100,
                    weapon_condition=WeaponCondition.EXCELLENT,
                )

        self.world_state = BattleWorldState(agents=self.agents, game_time=0.0)

    @pytest.mark.asyncio
    async def test_concurrent_effect_processing(self):
        """Test processing multiple effects that affect the same agents."""
        # Effects that could potentially conflict
        effects = [
            {
                "kind": "AgentDamaged",
                "payload": {
                    "target_id": "team_a_agent_1",
                    "damage": 20,
                    "new_health": 80,
                },
            },
            {
                "kind": "AgentHealed",
                "payload": {
                    "target_id": "team_a_agent_1",
                    "heal_amount": 10,
                    "new_health": 90,
                },
            },
            {
                "kind": "WeaponDegraded",
                "payload": {
                    "agent_id": "team_a_agent_1",
                    "old_condition": "excellent",
                    "new_condition": "good",
                },
            },
        ]

        result = await self.processor.process_effects(effects, self.world_state)

        # All effects should be processed
        assert result["processed_count"] == 3
        assert result["failed_count"] == 0

        # Final state should reflect the last health update
        agent = self.agents["team_a_agent_1"]
        assert agent.health == 90  # Healed after damage
        assert agent.weapon_condition == WeaponCondition.GOOD

    @pytest.mark.asyncio
    async def test_effect_processing_with_state_validation(self):
        """Test that effect processing maintains valid world state."""
        # Process effects that test boundary conditions
        effects = [
            # Damage agent to exactly 0 health
            {
                "kind": "AgentDamaged",
                "payload": {
                    "target_id": "team_a_agent_1",
                    "damage": 100,
                    "new_health": 0,
                },
            },
            # Try to heal dead agent (should work but agent stays dead)
            {
                "kind": "AgentHealed",
                "payload": {
                    "target_id": "team_a_agent_1",
                    "heal_amount": 50,
                    "new_health": 50,
                },
            },
            # Mark agent as officially dead
            {
                "kind": "AgentDied",
                "payload": {
                    "agent_id": "team_a_agent_1",
                    "killer_id": "team_b_agent_1",
                },
            },
        ]

        result = await self.processor.process_effects(effects, self.world_state)

        assert result["processed_count"] == 3

        # Agent should be dead with 0 health (death overrides heal)
        agent = self.agents["team_a_agent_1"]
        assert agent.status == AgentStatus.DEAD
        assert agent.health == 0  # Death effect sets health to 0

        # Team B should get a kill point
        assert self.world_state.team_scores["team_b"] == 1

        # World state should be consistent
        stats = self.manager.get_game_statistics(self.world_state)
        assert stats["agents_alive"]["team_a"] == 1  # One agent left
        assert stats["agents_alive"]["team_b"] == 2  # Both alive

    @pytest.mark.asyncio
    async def test_team_message_isolation(self):
        """Test that team messages are properly isolated between teams."""
        effects = [
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_a_agent_1",
                    "sender_team": "team_a",
                    "message": "Secret team A strategy",
                    "urgency": "high",
                },
            },
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_b_agent_1",
                    "sender_team": "team_b",
                    "message": "Secret team B strategy",
                    "urgency": "medium",
                },
            },
        ]

        result = await self.processor.process_effects(effects, self.world_state)

        assert result["processed_count"] == 2

        # Each team should only see their own messages
        team_a_messages = self.world_state.team_communications["team_a"]
        team_b_messages = self.world_state.team_communications["team_b"]

        assert len(team_a_messages) == 1
        assert len(team_b_messages) == 1

        assert "team A strategy" in team_a_messages[0].message
        assert "team B strategy" in team_b_messages[0].message

        # Verify team isolation
        assert team_a_messages[0].team == "team_a"
        assert team_b_messages[0].team == "team_b"
