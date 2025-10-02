"""
Comprehensive tests for effect processing and world state updates.

This module tests the EffectProcessor and GameStatusManager classes,
ensuring accurate effect processing, state consistency, and win condition detection.
"""

from unittest.mock import patch

import pytest

from ..backend.effect_processor import EffectProcessor, GameStatusManager
from ..shared.enums import AgentStatus, WeaponCondition
from ..shared.models import Agent, BattleWorldState


class TestEffectProcessor:
    """Test cases for the EffectProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = EffectProcessor()

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

        # Create test world state
        self.world_state = BattleWorldState(
            agents={"team_a_agent_1": self.agent_a, "team_b_agent_1": self.agent_b},
            game_time=10.0,
        )

    @pytest.mark.asyncio
    async def test_process_effects_empty_list(self):
        """Test processing empty effects list."""
        result = await self.processor.process_effects([], self.world_state)

        assert result["processed_count"] == 0
        assert result["failed_count"] == 0
        assert result["effect_types"] == {}
        assert result["game_status"] == "active"
        assert not result["status_changed"]

    @pytest.mark.asyncio
    async def test_process_effects_multiple_effects(self):
        """Test processing multiple effects."""
        effects = [
            {
                "kind": "AgentDamaged",
                "payload": {
                    "target_id": "team_b_agent_1",
                    "damage": 20,
                    "new_health": 60,
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

        assert result["processed_count"] == 2
        assert result["failed_count"] == 0
        assert result["effect_types"]["AgentDamaged"] == 1
        assert result["effect_types"]["WeaponDegraded"] == 1

        # Check that effects were applied
        assert self.agent_b.health == 60
        assert self.agent_a.weapon_condition == WeaponCondition.GOOD

    @pytest.mark.asyncio
    async def test_handle_agent_damaged_valid(self):
        """Test handling valid AgentDamaged effect."""
        effect = {
            "kind": "AgentDamaged",
            "payload": {"target_id": "team_b_agent_1", "damage": 30, "new_health": 50},
        }

        await self.processor._handle_agent_damaged(effect, self.world_state)

        assert self.agent_b.health == 50
        assert self.agent_b.status == AgentStatus.ALIVE  # Still alive

    @pytest.mark.asyncio
    async def test_handle_agent_damaged_to_zero_health(self):
        """Test handling AgentDamaged that reduces health to zero."""
        effect = {
            "kind": "AgentDamaged",
            "payload": {"target_id": "team_b_agent_1", "damage": 80, "new_health": 0},
        }

        await self.processor._handle_agent_damaged(effect, self.world_state)

        assert self.agent_b.health == 0
        assert self.agent_b.status == AgentStatus.DEAD  # Should be marked dead

    @pytest.mark.asyncio
    async def test_handle_agent_damaged_invalid_payload(self):
        """Test handling AgentDamaged with invalid payload."""
        effect = {
            "kind": "AgentDamaged",
            "payload": {"target_id": "nonexistent", "damage": 20, "new_health": 60},
        }

        # Should raise ValueError for invalid agent
        with pytest.raises(ValueError):
            await self.processor._handle_agent_damaged(effect, self.world_state)

        # Original agent should be unchanged
        assert self.agent_b.health == 80

    @pytest.mark.asyncio
    async def test_handle_agent_died_valid(self):
        """Test handling valid AgentDied effect."""
        effect = {
            "kind": "AgentDied",
            "payload": {"agent_id": "team_b_agent_1", "killer_id": "team_a_agent_1"},
        }

        initial_score = self.world_state.team_scores["team_a"]

        await self.processor._handle_agent_died(effect, self.world_state)

        assert self.agent_b.status == AgentStatus.DEAD
        assert self.agent_b.health == 0
        assert self.world_state.team_scores["team_a"] == initial_score + 1

    @pytest.mark.asyncio
    async def test_handle_agent_died_no_killer(self):
        """Test handling AgentDied effect without killer."""
        effect = {"kind": "AgentDied", "payload": {"agent_id": "team_b_agent_1"}}

        initial_score = self.world_state.team_scores["team_a"]

        await self.processor._handle_agent_died(effect, self.world_state)

        assert self.agent_b.status == AgentStatus.DEAD
        assert self.agent_b.health == 0
        assert (
            self.world_state.team_scores["team_a"] == initial_score
        )  # No score change

    @pytest.mark.asyncio
    async def test_handle_weapon_degraded_valid(self):
        """Test handling valid WeaponDegraded effect."""
        effect = {
            "kind": "WeaponDegraded",
            "payload": {
                "agent_id": "team_a_agent_1",
                "old_condition": "excellent",
                "new_condition": "good",
            },
        }

        await self.processor._handle_weapon_degraded(effect, self.world_state)

        assert self.agent_a.weapon_condition == WeaponCondition.GOOD

    @pytest.mark.asyncio
    async def test_handle_weapon_degraded_invalid_condition(self):
        """Test handling WeaponDegraded with invalid condition."""
        effect = {
            "kind": "WeaponDegraded",
            "payload": {
                "agent_id": "team_a_agent_1",
                "old_condition": "excellent",
                "new_condition": "invalid_condition",
            },
        }

        original_condition = self.agent_a.weapon_condition

        # Should not raise exception, just log error
        await self.processor._handle_weapon_degraded(effect, self.world_state)

        # Weapon condition should remain unchanged
        assert self.agent_a.weapon_condition == original_condition

    @pytest.mark.asyncio
    async def test_handle_agent_healed_valid(self):
        """Test handling valid AgentHealed effect."""
        self.agent_b.health = 50  # Set up injured agent

        effect = {
            "kind": "AgentHealed",
            "payload": {
                "target_id": "team_b_agent_1",
                "heal_amount": 30,
                "new_health": 80,
            },
        }

        await self.processor._handle_agent_healed(effect, self.world_state)

        assert self.agent_b.health == 80

    @pytest.mark.asyncio
    async def test_handle_agent_healed_overheal(self):
        """Test handling AgentHealed that would exceed max health."""
        self.agent_b.health = 90

        effect = {
            "kind": "AgentHealed",
            "payload": {
                "target_id": "team_b_agent_1",
                "heal_amount": 20,
                "new_health": 110,  # Would exceed 100
            },
        }

        await self.processor._handle_agent_healed(effect, self.world_state)

        assert self.agent_b.health == 100  # Should be capped at 100

    @pytest.mark.asyncio
    async def test_handle_weapon_repaired_valid(self):
        """Test handling valid WeaponRepaired effect."""
        self.agent_a.weapon_condition = WeaponCondition.DAMAGED

        effect = {
            "kind": "WeaponRepaired",
            "payload": {
                "agent_id": "team_a_agent_1",
                "old_condition": "damaged",
                "new_condition": "excellent",
            },
        }

        await self.processor._handle_weapon_repaired(effect, self.world_state)

        assert self.agent_a.weapon_condition == WeaponCondition.EXCELLENT

    @pytest.mark.asyncio
    async def test_handle_team_message_valid(self):
        """Test handling valid TeamMessage effect."""
        effect = {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_a_agent_1",
                "sender_team": "team_a",
                "message": "Enemy spotted at coordinates (25, 30)!",
                "urgency": "high",
                "timestamp": 15.0,
            },
        }

        await self.processor._handle_team_message(effect, self.world_state)

        # Check that message was added to team communications
        team_messages = self.world_state.team_communications["team_a"]
        assert len(team_messages) == 1
        assert team_messages[0].sender_id == "team_a_agent_1"
        assert team_messages[0].message == "Enemy spotted at coordinates (25, 30)!"
        assert team_messages[0].urgency == "high"
        assert team_messages[0].timestamp == 15.0

        # Check that enemy team doesn't see the message
        enemy_messages = self.world_state.team_communications["team_b"]
        assert len(enemy_messages) == 0

    @pytest.mark.asyncio
    async def test_handle_team_message_team_mismatch(self):
        """Test handling TeamMessage with team mismatch."""
        effect = {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_a_agent_1",
                "sender_team": "team_b",  # Wrong team
                "message": "Test message",
                "urgency": "medium",
            },
        }

        # Should not add message due to team mismatch
        await self.processor._handle_team_message(effect, self.world_state)

        # No messages should be added
        assert len(self.world_state.team_communications["team_a"]) == 0
        assert len(self.world_state.team_communications["team_b"]) == 0

    @pytest.mark.asyncio
    async def test_handle_team_message_message_limit(self):
        """Test team message storage limit."""
        # Add 55 messages to exceed the 50 message limit
        for i in range(55):
            effect = {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_a_agent_1",
                    "sender_team": "team_a",
                    "message": f"Message {i}",
                    "urgency": "medium",
                    "timestamp": float(i),
                },
            }
            await self.processor._handle_team_message(effect, self.world_state)

        # Should only keep last 50 messages
        team_messages = self.world_state.team_communications["team_a"]
        assert len(team_messages) == 50
        assert team_messages[0].message == "Message 5"  # First kept message
        assert team_messages[-1].message == "Message 54"  # Last message

    @pytest.mark.asyncio
    async def test_handle_agent_move_valid(self):
        """Test handling valid Move effect."""
        effect = {
            "kind": "Move",
            "source_id": "team_a_agent_1",
            "payload": {
                "target_position": (25.0, 35.0),
                "reason": "Moving to better position",
            },
        }

        await self.processor._handle_agent_move(effect, self.world_state)

        assert self.agent_a.position == (25.0, 35.0)

    @pytest.mark.asyncio
    async def test_handle_failed_effects(self):
        """Test handling various failed effect types."""
        failed_effects = [
            {
                "kind": "AttackFailed",
                "payload": {"attacker_id": "team_a_agent_1", "reason": "out_of_range"},
            },
            {
                "kind": "HealFailed",
                "payload": {
                    "healer_id": "team_a_agent_1",
                    "reason": "target_at_full_health",
                },
            },
            {
                "kind": "RepairFailed",
                "payload": {"agent_id": "team_a_agent_1", "reason": "not_at_forge"},
            },
            {
                "kind": "CommunicationFailed",
                "payload": {"sender_id": "team_a_agent_1", "reason": "empty_message"},
            },
        ]

        # These should all process without error (just logging)
        for effect in failed_effects:
            await self.processor._process_single_effect(effect, self.world_state)

        # World state should remain unchanged
        assert self.agent_a.health == 100
        assert self.agent_a.weapon_condition == WeaponCondition.EXCELLENT
        assert self.agent_a.position == (10.0, 10.0)

    @pytest.mark.asyncio
    async def test_unknown_effect_kind(self):
        """Test handling unknown effect kind."""
        effect = {"kind": "UnknownEffect", "payload": {"data": "test"}}

        # Should not raise exception, just log warning
        await self.processor._process_single_effect(effect, self.world_state)

        # World state should remain unchanged
        assert self.agent_a.health == 100

    @pytest.mark.asyncio
    async def test_process_effects_with_game_status_change(self):
        """Test processing effects that cause game status change."""
        # Kill all team B agents
        effects = [
            {
                "kind": "AgentDied",
                "payload": {
                    "agent_id": "team_b_agent_1",
                    "killer_id": "team_a_agent_1",
                },
            }
        ]

        result = await self.processor.process_effects(effects, self.world_state)

        assert result["game_status"] == "team_a_wins"
        assert result["status_changed"] is True

    @pytest.mark.asyncio
    async def test_process_effects_error_handling(self):
        """Test error handling during effect processing."""
        # Create effect that will cause an error by raising an exception
        effect = {
            "kind": "AgentDamaged",
            "payload": {
                "target_id": None,  # Invalid payload
                "new_health": "invalid",
            },
        }

        # Mock the handler to raise an exception
        with patch.object(
            self.processor, "_handle_agent_damaged", side_effect=Exception("Test error")
        ):
            result = await self.processor.process_effects([effect], self.world_state)

        assert result["processed_count"] == 0
        assert result["failed_count"] == 1


class TestGameStatusManager:
    """Test cases for the GameStatusManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = GameStatusManager()

        # Create test agents
        self.agents = {}
        for team in ["team_a", "team_b"]:
            for i in range(2):
                agent_id = f"{team}_agent_{i + 1}"
                self.agents[agent_id] = Agent(
                    agent_id=agent_id,
                    team=team,
                    position=(10.0 + i * 10, 10.0),
                    health=100,
                    status=AgentStatus.ALIVE,
                )

        self.world_state = BattleWorldState(agents=self.agents)

    def test_check_win_condition_active_game(self):
        """Test win condition check for active game."""
        status = self.manager.check_win_condition(self.world_state)
        assert status == "active"

    def test_check_win_condition_team_a_wins(self):
        """Test win condition when team A wins."""
        # Kill all team B agents
        for agent_id in ["team_b_agent_1", "team_b_agent_2"]:
            self.agents[agent_id].status = AgentStatus.DEAD

        status = self.manager.check_win_condition(self.world_state)
        assert status == "team_a_wins"

    def test_check_win_condition_team_b_wins(self):
        """Test win condition when team B wins."""
        # Kill all team A agents
        for agent_id in ["team_a_agent_1", "team_a_agent_2"]:
            self.agents[agent_id].status = AgentStatus.DEAD

        status = self.manager.check_win_condition(self.world_state)
        assert status == "team_b_wins"

    def test_check_win_condition_draw(self):
        """Test win condition for draw."""
        # Kill all agents
        for agent in self.agents.values():
            agent.status = AgentStatus.DEAD

        status = self.manager.check_win_condition(self.world_state)
        assert status == "draw"

    def test_update_game_status_no_change(self):
        """Test updating game status when no change occurs."""
        changed = self.manager.update_game_status(self.world_state)

        assert not changed
        assert self.world_state.game_status == "active"

    def test_update_game_status_with_change(self):
        """Test updating game status when change occurs."""
        # Kill all team B agents
        for agent_id in ["team_b_agent_1", "team_b_agent_2"]:
            self.agents[agent_id].status = AgentStatus.DEAD

        changed = self.manager.update_game_status(self.world_state)

        assert changed
        assert self.world_state.game_status == "team_a_wins"

    def test_get_game_statistics_active_game(self):
        """Test getting game statistics for active game."""
        stats = self.manager.get_game_statistics(self.world_state)

        assert stats["game_status"] == "active"
        assert stats["agents_alive"]["team_a"] == 2
        assert stats["agents_alive"]["team_b"] == 2
        assert stats["agents_alive"]["total"] == 4
        assert stats["team_health"]["team_a"] == 200  # 2 agents * 100 health
        assert stats["team_health"]["team_b"] == 200
        assert stats["weapon_conditions"]["team_a"]["excellent"] == 2
        assert stats["weapon_conditions"]["team_b"]["excellent"] == 2
        assert stats["total_agents"]["team_a"] == 2
        assert stats["total_agents"]["team_b"] == 2

    def test_get_game_statistics_with_casualties(self):
        """Test getting game statistics with casualties and damage."""
        # Damage some agents
        self.agents["team_a_agent_1"].health = 50
        self.agents["team_b_agent_1"].status = AgentStatus.DEAD
        self.agents["team_b_agent_1"].health = 0

        # Degrade some weapons
        self.agents["team_a_agent_2"].weapon_condition = WeaponCondition.DAMAGED

        stats = self.manager.get_game_statistics(self.world_state)

        assert stats["agents_alive"]["team_a"] == 2
        assert stats["agents_alive"]["team_b"] == 1  # One dead
        assert stats["team_health"]["team_a"] == 150  # 50 + 100
        assert stats["team_health"]["team_b"] == 100  # Only living agent
        assert stats["weapon_conditions"]["team_a"]["excellent"] == 1
        assert stats["weapon_conditions"]["team_a"]["damaged"] == 1
        assert (
            stats["weapon_conditions"]["team_b"]["excellent"] == 2
        )  # Includes dead agent

    def test_get_game_statistics_with_communications(self):
        """Test game statistics including communication counts."""
        # Add some team communications
        self.world_state.add_team_message("team_a_agent_1", "Test message 1")
        self.world_state.add_team_message("team_a_agent_2", "Test message 2")
        self.world_state.add_team_message("team_b_agent_1", "Enemy message")

        stats = self.manager.get_game_statistics(self.world_state)

        assert stats["communication_counts"]["team_a"] == 2
        assert stats["communication_counts"]["team_b"] == 1


class TestIntegrationScenarios:
    """Integration tests for complex effect processing scenarios."""

    def setup_method(self):
        """Set up complex test scenario."""
        self.processor = EffectProcessor()
        self.manager = GameStatusManager()

        # Create full team setup
        self.agents = {}
        for team in ["team_a", "team_b"]:
            for i in range(3):
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
    async def test_complete_battle_sequence(self):
        """Test a complete battle sequence with multiple effect types."""
        # Sequence of effects representing a battle
        battle_effects = [
            # Round 1: Attacks and damage
            {
                "kind": "AgentDamaged",
                "payload": {
                    "target_id": "team_b_agent_1",
                    "damage": 30,
                    "new_health": 70,
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
            # Round 2: Healing and communication
            {
                "kind": "AgentHealed",
                "payload": {
                    "target_id": "team_b_agent_1",
                    "heal_amount": 20,
                    "new_health": 90,
                },
            },
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_b_agent_2",
                    "sender_team": "team_b",
                    "message": "Agent 1 is injured, providing cover!",
                    "urgency": "high",
                },
            },
            # Round 3: Fatal damage
            {
                "kind": "AgentDamaged",
                "payload": {
                    "target_id": "team_b_agent_3",
                    "damage": 100,
                    "new_health": 0,
                },
            },
            {
                "kind": "AgentDied",
                "payload": {
                    "agent_id": "team_b_agent_3",
                    "killer_id": "team_a_agent_2",
                },
            },
        ]

        result = await self.processor.process_effects(battle_effects, self.world_state)

        # Verify all effects were processed
        assert result["processed_count"] == 6
        assert result["failed_count"] == 0

        # Verify state changes
        assert self.agents["team_b_agent_1"].health == 90
        assert self.agents["team_a_agent_1"].weapon_condition == WeaponCondition.GOOD
        assert self.agents["team_b_agent_3"].status == AgentStatus.DEAD
        assert self.world_state.team_scores["team_a"] == 1

        # Verify team communication
        team_b_messages = self.world_state.team_communications["team_b"]
        assert len(team_b_messages) == 1
        assert "injured" in team_b_messages[0].message

        # Game should still be active (team B has 2 agents left)
        assert self.world_state.game_status == "active"

    @pytest.mark.asyncio
    async def test_team_elimination_scenario(self):
        """Test scenario where entire team is eliminated."""
        # Kill all team B agents
        elimination_effects = []
        for i, agent_id in enumerate(
            ["team_b_agent_1", "team_b_agent_2", "team_b_agent_3"]
        ):
            elimination_effects.extend(
                [
                    {
                        "kind": "AgentDamaged",
                        "payload": {
                            "target_id": agent_id,
                            "damage": 100,
                            "new_health": 0,
                        },
                    },
                    {
                        "kind": "AgentDied",
                        "payload": {
                            "agent_id": agent_id,
                            "killer_id": f"team_a_agent_{i + 1}",
                        },
                    },
                ]
            )

        result = await self.processor.process_effects(
            elimination_effects, self.world_state
        )

        # All team B agents should be dead
        for agent_id in ["team_b_agent_1", "team_b_agent_2", "team_b_agent_3"]:
            assert self.agents[agent_id].status == AgentStatus.DEAD
            assert self.agents[agent_id].health == 0

        # Team A should have 3 points
        assert self.world_state.team_scores["team_a"] == 3

        # Game should end with team A victory
        assert result["game_status"] == "team_a_wins"
        assert result["status_changed"] is True

    @pytest.mark.asyncio
    async def test_concurrent_effects_processing(self):
        """Test processing effects that might occur simultaneously."""
        # Simulate concurrent combat effects
        concurrent_effects = [
            # Both teams attack simultaneously
            {
                "kind": "AgentDamaged",
                "payload": {
                    "target_id": "team_a_agent_1",
                    "damage": 25,
                    "new_health": 75,
                },
            },
            {
                "kind": "AgentDamaged",
                "payload": {
                    "target_id": "team_b_agent_1",
                    "damage": 30,
                    "new_health": 70,
                },
            },
            # Both teams communicate simultaneously
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_a_agent_2",
                    "sender_team": "team_a",
                    "message": "Taking damage, need support!",
                    "urgency": "high",
                },
            },
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_b_agent_2",
                    "sender_team": "team_b",
                    "message": "Enemy is weakened, press the attack!",
                    "urgency": "medium",
                },
            },
        ]

        result = await self.processor.process_effects(
            concurrent_effects, self.world_state
        )

        # All effects should be processed successfully
        assert result["processed_count"] == 4
        assert result["failed_count"] == 0

        # Both agents should be damaged
        assert self.agents["team_a_agent_1"].health == 75
        assert self.agents["team_b_agent_1"].health == 70

        # Both teams should have messages (team-only visibility)
        assert len(self.world_state.team_communications["team_a"]) == 1
        assert len(self.world_state.team_communications["team_b"]) == 1

        # Messages should not be visible to enemy teams
        team_a_msg = self.world_state.team_communications["team_a"][0]
        team_b_msg = self.world_state.team_communications["team_b"][0]

        assert "support" in team_a_msg.message
        assert "attack" in team_b_msg.message
        assert team_a_msg.team == "team_a"
        assert team_b_msg.team == "team_b"

    @pytest.mark.asyncio
    async def test_effect_processing_consistency(self):
        """Test that effect processing maintains world state consistency."""
        # Create effects that test various edge cases
        consistency_effects = [
            # Heal agent to full health
            {
                "kind": "AgentHealed",
                "payload": {
                    "target_id": "team_a_agent_1",
                    "heal_amount": 0,  # Already at full health
                    "new_health": 100,
                },
            },
            # Repair already excellent weapon
            {
                "kind": "WeaponRepaired",
                "payload": {
                    "agent_id": "team_a_agent_1",
                    "old_condition": "excellent",
                    "new_condition": "excellent",
                },
            },
            # Move agent to new position
            {
                "kind": "Move",
                "source_id": "team_a_agent_1",
                "payload": {"target_position": (50.0, 60.0)},
            },
        ]

        # Get initial state
        initial_stats = self.manager.get_game_statistics(self.world_state)

        result = await self.processor.process_effects(
            consistency_effects, self.world_state
        )

        # Get final state
        final_stats = self.manager.get_game_statistics(self.world_state)

        # All effects should process without error
        assert result["processed_count"] == 3
        assert result["failed_count"] == 0

        # Agent should have moved
        assert self.agents["team_a_agent_1"].position == (50.0, 60.0)

        # Health and weapon stats should remain consistent
        assert (
            final_stats["team_health"]["team_a"]
            == initial_stats["team_health"]["team_a"]
        )
        assert final_stats["weapon_conditions"] == initial_stats["weapon_conditions"]

        # Game should remain active
        assert final_stats["game_status"] == "active"
