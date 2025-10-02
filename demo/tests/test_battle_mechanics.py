"""
Comprehensive tests for battle mechanics and combat system.

This module tests all aspects of the BattleMechanics and CombatManager classes,
including damage calculations, weapon degradation, healing, repair mechanics,
and team communication.
"""

from unittest.mock import patch

import pytest

from ..backend.battle_mechanics import BattleMechanics, CombatManager
from ..shared.constants import GAME_CONFIG
from ..shared.enums import AgentStatus, LocationType, WeaponCondition
from ..shared.models import Agent, BattleWorldState, MapLocation


class TestBattleMechanics:
    """Test cases for BattleMechanics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mechanics = BattleMechanics()

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

        # Create test forge
        self.forge_a = MapLocation(
            position=(15.0, 15.0), location_type=LocationType.FORGE, radius=5.0
        )

    def test_initialization(self):
        """Test BattleMechanics initialization."""
        assert self.mechanics.attack_damage == GAME_CONFIG["attack_damage"]
        assert self.mechanics.heal_amount == GAME_CONFIG["heal_amount"]
        assert (
            self.mechanics.weapon_degradation_rate
            == GAME_CONFIG["weapon_degradation_rate"]
        )
        assert self.mechanics.movement_speed == GAME_CONFIG["movement_speed"]
        assert self.mechanics.attack_cooldown == GAME_CONFIG["attack_cooldown"]
        assert self.mechanics.heal_cooldown == GAME_CONFIG["heal_cooldown"]

    def test_calculate_attack_damage_excellent_weapon(self):
        """Test damage calculation with excellent weapon."""
        distance = 10.0
        with patch("random.uniform", return_value=1.0):  # No randomness
            damage = self.mechanics.calculate_attack_damage(
                self.agent_a, self.agent_b, distance
            )
            expected = GAME_CONFIG["attack_damage"] * 1.0  # Excellent weapon modifier
            assert damage == expected

    def test_calculate_attack_damage_good_weapon(self):
        """Test damage calculation with good weapon."""
        distance = 10.0
        with patch("random.uniform", return_value=1.0):  # No randomness
            damage = self.mechanics.calculate_attack_damage(
                self.agent_b, self.agent_a, distance
            )
            expected = int(GAME_CONFIG["attack_damage"] * 0.8)  # Good weapon modifier
            assert damage == expected

    def test_calculate_attack_damage_damaged_weapon(self):
        """Test damage calculation with damaged weapon."""
        self.agent_a.weapon_condition = WeaponCondition.DAMAGED
        distance = 10.0
        with patch("random.uniform", return_value=1.0):  # No randomness
            damage = self.mechanics.calculate_attack_damage(
                self.agent_a, self.agent_b, distance
            )
            expected = int(
                GAME_CONFIG["attack_damage"] * 0.6
            )  # Damaged weapon modifier
            assert damage == expected

    def test_calculate_attack_damage_broken_weapon(self):
        """Test damage calculation with broken weapon."""
        self.agent_a.weapon_condition = WeaponCondition.BROKEN
        distance = 10.0
        damage = self.mechanics.calculate_attack_damage(
            self.agent_a, self.agent_b, distance
        )
        assert damage == 0

    def test_calculate_attack_damage_close_range_bonus(self):
        """Test close range damage bonus."""
        distance = 3.0  # Within close range (≤5.0)
        with patch("random.uniform", return_value=1.0):  # No randomness
            damage = self.mechanics.calculate_attack_damage(
                self.agent_a, self.agent_b, distance
            )
            expected = int(GAME_CONFIG["attack_damage"] * 1.2)  # Close range bonus
            assert damage == expected

    def test_calculate_attack_damage_out_of_range(self):
        """Test damage calculation when out of range."""
        distance = 20.0  # Beyond attack range (15.0)
        damage = self.mechanics.calculate_attack_damage(
            self.agent_a, self.agent_b, distance
        )
        assert damage == 0

    def test_degrade_weapon_no_degradation(self):
        """Test weapon degradation when no degradation occurs."""
        original_condition = self.agent_a.weapon_condition
        with patch("random.random", return_value=0.5):  # Above degradation rate
            new_condition = self.mechanics.degrade_weapon(self.agent_a)
            assert new_condition == original_condition
            assert self.agent_a.weapon_condition == original_condition

    def test_degrade_weapon_with_degradation(self):
        """Test weapon degradation when degradation occurs."""
        self.agent_a.weapon_condition = WeaponCondition.EXCELLENT
        with patch("random.random", return_value=0.05):  # Below degradation rate
            new_condition = self.mechanics.degrade_weapon(self.agent_a)
            assert new_condition == WeaponCondition.GOOD
            assert self.agent_a.weapon_condition == WeaponCondition.GOOD

    def test_degrade_weapon_already_broken(self):
        """Test weapon degradation when weapon is already broken."""
        self.agent_a.weapon_condition = WeaponCondition.BROKEN
        with patch("random.random", return_value=0.05):  # Below degradation rate
            new_condition = self.mechanics.degrade_weapon(self.agent_a)
            assert new_condition == WeaponCondition.BROKEN
            assert self.agent_a.weapon_condition == WeaponCondition.BROKEN

    def test_can_perform_action_dead_agent(self):
        """Test action validation for dead agent."""
        self.agent_a.status = AgentStatus.DEAD
        assert not self.mechanics.can_perform_action(self.agent_a, "attack", 10.0)
        assert not self.mechanics.can_perform_action(self.agent_a, "heal", 10.0)
        assert not self.mechanics.can_perform_action(self.agent_a, "move", 10.0)

    def test_can_perform_action_attack_cooldown(self):
        """Test attack action with cooldown."""
        current_time = 10.0
        self.agent_a.last_action_time = 8.5  # 1.5 seconds ago

        # Should fail (cooldown is 2.0 seconds)
        assert not self.mechanics.can_perform_action(
            self.agent_a, "attack", current_time
        )

        # Should succeed after cooldown
        self.agent_a.last_action_time = 7.5  # 2.5 seconds ago
        assert self.mechanics.can_perform_action(self.agent_a, "attack", current_time)

    def test_can_perform_action_attack_broken_weapon(self):
        """Test attack action with broken weapon."""
        self.agent_a.weapon_condition = WeaponCondition.BROKEN
        self.agent_a.last_action_time = 0.0
        current_time = 10.0

        assert not self.mechanics.can_perform_action(
            self.agent_a, "attack", current_time
        )

    def test_can_perform_action_heal_cooldown(self):
        """Test heal action with cooldown."""
        current_time = 10.0
        self.agent_a.last_action_time = 8.0  # 2.0 seconds ago

        # Should fail (heal cooldown is 3.0 seconds)
        assert not self.mechanics.can_perform_action(self.agent_a, "heal", current_time)

        # Should succeed after cooldown
        self.agent_a.last_action_time = 6.5  # 3.5 seconds ago
        assert self.mechanics.can_perform_action(self.agent_a, "heal", current_time)

    def test_can_perform_action_no_cooldown_actions(self):
        """Test actions that have no cooldown."""
        current_time = 10.0
        self.agent_a.last_action_time = 9.9  # Very recent action

        assert self.mechanics.can_perform_action(self.agent_a, "move", current_time)
        assert self.mechanics.can_perform_action(self.agent_a, "repair", current_time)
        assert self.mechanics.can_perform_action(
            self.agent_a, "communicate", current_time
        )

    def test_calculate_movement_time(self):
        """Test movement time calculation."""
        start_pos = (0.0, 0.0)
        end_pos = (3.0, 4.0)  # Distance = 5.0
        expected_time = 5.0 / GAME_CONFIG["movement_speed"]

        time = self.mechanics.calculate_movement_time(start_pos, end_pos)
        assert abs(time - expected_time) < 0.001

    def test_is_at_forge_true(self):
        """Test forge detection when agent is at forge."""
        self.agent_a.position = (15.0, 15.0)  # Same as forge position
        forge_locations = {"forge_a": self.forge_a}

        assert self.mechanics.is_at_forge(self.agent_a, forge_locations)

    def test_is_at_forge_within_radius(self):
        """Test forge detection when agent is within forge radius."""
        self.agent_a.position = (18.0, 18.0)  # Within 5.0 radius
        forge_locations = {"forge_a": self.forge_a}

        assert self.mechanics.is_at_forge(self.agent_a, forge_locations)

    def test_is_at_forge_false(self):
        """Test forge detection when agent is not at forge."""
        self.agent_a.position = (25.0, 25.0)  # Outside forge radius
        forge_locations = {"forge_a": self.forge_a}

        assert not self.mechanics.is_at_forge(self.agent_a, forge_locations)

    def test_is_at_forge_wrong_team(self):
        """Test forge detection for wrong team forge."""
        # Agent is team_a, but we only have forge_b
        forge_b = MapLocation(
            position=(15.0, 15.0), location_type=LocationType.FORGE, radius=5.0
        )
        self.agent_a.position = (15.0, 15.0)
        forge_locations = {"forge_b": forge_b}

        assert not self.mechanics.is_at_forge(self.agent_a, forge_locations)

    def test_calculate_heal_amount_normal(self):
        """Test normal healing amount calculation."""
        self.agent_b.health = 50
        heal_amount = self.mechanics.calculate_heal_amount(self.agent_a, self.agent_b)
        assert heal_amount == GAME_CONFIG["heal_amount"]

    def test_calculate_heal_amount_self_heal(self):
        """Test self-healing amount calculation."""
        self.agent_a.health = 50
        heal_amount = self.mechanics.calculate_heal_amount(self.agent_a, self.agent_a)
        expected = int(GAME_CONFIG["heal_amount"] * 0.8)
        assert heal_amount == expected

    def test_calculate_heal_amount_near_full_health(self):
        """Test healing when target is near full health."""
        self.agent_b.health = 95
        heal_amount = self.mechanics.calculate_heal_amount(self.agent_a, self.agent_b)
        assert heal_amount == 5  # Can only heal to 100

    def test_is_valid_attack_target_valid(self):
        """Test valid attack target validation."""
        world_state = BattleWorldState(
            agents={"team_a_agent_1": self.agent_a, "team_b_agent_1": self.agent_b}
        )

        is_valid, reason = self.mechanics.is_valid_attack_target(
            self.agent_a, self.agent_b, world_state
        )
        assert is_valid
        assert reason == "valid"

    def test_is_valid_attack_target_teammate(self):
        """Test attack target validation for teammate."""
        teammate = Agent(
            agent_id="team_a_agent_2", team="team_a", position=(15.0, 15.0)
        )
        world_state = BattleWorldState()

        is_valid, reason = self.mechanics.is_valid_attack_target(
            self.agent_a, teammate, world_state
        )
        assert not is_valid
        assert reason == "cannot_attack_teammate"

    def test_is_valid_attack_target_dead(self):
        """Test attack target validation for dead agent."""
        self.agent_b.status = AgentStatus.DEAD
        world_state = BattleWorldState()

        is_valid, reason = self.mechanics.is_valid_attack_target(
            self.agent_a, self.agent_b, world_state
        )
        assert not is_valid
        assert reason == "target_is_dead"

    def test_is_valid_attack_target_out_of_range(self):
        """Test attack target validation for out of range target."""
        self.agent_b.position = (100.0, 100.0)  # Far away
        world_state = BattleWorldState()

        is_valid, reason = self.mechanics.is_valid_attack_target(
            self.agent_a, self.agent_b, world_state
        )
        assert not is_valid
        assert reason == "out_of_range"

    def test_is_valid_attack_target_broken_weapon(self):
        """Test attack target validation with broken weapon."""
        self.agent_a.weapon_condition = WeaponCondition.BROKEN
        world_state = BattleWorldState()

        is_valid, reason = self.mechanics.is_valid_attack_target(
            self.agent_a, self.agent_b, world_state
        )
        assert not is_valid
        assert reason == "weapon_broken"

    def test_is_valid_heal_target_valid(self):
        """Test valid heal target validation."""
        teammate = Agent(
            agent_id="team_a_agent_2", team="team_a", position=(15.0, 15.0), health=50
        )

        is_valid, reason = self.mechanics.is_valid_heal_target(self.agent_a, teammate)
        assert is_valid
        assert reason == "valid"

    def test_is_valid_heal_target_enemy(self):
        """Test heal target validation for enemy."""
        is_valid, reason = self.mechanics.is_valid_heal_target(
            self.agent_a, self.agent_b
        )
        assert not is_valid
        assert reason == "cannot_heal_enemy"

    def test_is_valid_heal_target_dead(self):
        """Test heal target validation for dead teammate."""
        teammate = Agent(
            agent_id="team_a_agent_2",
            team="team_a",
            position=(15.0, 15.0),
            status=AgentStatus.DEAD,
        )

        is_valid, reason = self.mechanics.is_valid_heal_target(self.agent_a, teammate)
        assert not is_valid
        assert reason == "target_is_dead"

    def test_is_valid_heal_target_full_health(self):
        """Test heal target validation for full health teammate."""
        teammate = Agent(
            agent_id="team_a_agent_2", team="team_a", position=(15.0, 15.0), health=100
        )

        is_valid, reason = self.mechanics.is_valid_heal_target(self.agent_a, teammate)
        assert not is_valid
        assert reason == "target_at_full_health"


class TestCombatManager:
    """Test cases for CombatManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mechanics = BattleMechanics()
        self.combat_manager = CombatManager(self.mechanics)

        # Create test world state
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

        self.forge_a = MapLocation(
            position=(15.0, 15.0), location_type=LocationType.FORGE, radius=5.0
        )

        self.world_state = BattleWorldState(
            agents={"team_a_agent_1": self.agent_a, "team_b_agent_1": self.agent_b},
            map_locations={"forge_a": self.forge_a},
            game_time=10.0,
        )

    @pytest.mark.asyncio
    async def test_process_attack_successful(self):
        """Test successful attack processing."""
        with patch("random.uniform", return_value=1.0):  # No randomness
            effects = await self.combat_manager.process_attack(
                "team_a_agent_1", "team_b_agent_1", self.world_state
            )

        # Should have damage effect and possibly weapon degradation
        assert len(effects) >= 1

        damage_effect = next(e for e in effects if e["kind"] == "AgentDamaged")
        assert damage_effect["payload"]["attacker_id"] == "team_a_agent_1"
        assert damage_effect["payload"]["target_id"] == "team_b_agent_1"
        assert damage_effect["payload"]["damage"] > 0
        assert damage_effect["payload"]["new_health"] < 80

        # Check that target health was actually updated
        assert self.agent_b.health < 80

    @pytest.mark.asyncio
    async def test_process_attack_agent_not_found(self):
        """Test attack processing with non-existent agent."""
        effects = await self.combat_manager.process_attack(
            "nonexistent", "team_b_agent_1", self.world_state
        )

        assert len(effects) == 1
        assert effects[0]["kind"] == "AttackFailed"
        assert effects[0]["payload"]["reason"] == "agent_not_found"

    @pytest.mark.asyncio
    async def test_process_attack_invalid_target(self):
        """Test attack processing with invalid target."""
        # Create a teammate agent
        teammate = Agent(
            agent_id="team_a_agent_2", team="team_a", position=(15.0, 15.0), health=100
        )
        self.world_state.agents["team_a_agent_2"] = teammate

        effects = await self.combat_manager.process_attack(
            "team_a_agent_1", "team_a_agent_2", self.world_state
        )

        assert len(effects) == 1
        assert effects[0]["kind"] == "AttackFailed"
        assert effects[0]["payload"]["reason"] == "cannot_attack_teammate"

    @pytest.mark.asyncio
    async def test_process_attack_kills_target(self):
        """Test attack that kills the target."""
        # Set target to low health
        self.agent_b.health = 1

        with patch("random.uniform", return_value=1.0):  # No randomness
            effects = await self.combat_manager.process_attack(
                "team_a_agent_1", "team_b_agent_1", self.world_state
            )

        # Should have damage effect and death effect
        damage_effect = next(e for e in effects if e["kind"] == "AgentDamaged")
        death_effect = next(e for e in effects if e["kind"] == "AgentDied")

        assert damage_effect["payload"]["new_health"] == 0
        assert death_effect["payload"]["agent_id"] == "team_b_agent_1"
        assert death_effect["payload"]["killer_id"] == "team_a_agent_1"

        # Check that target is actually dead
        assert self.agent_b.health == 0
        assert self.agent_b.status == AgentStatus.DEAD

    @pytest.mark.asyncio
    async def test_process_attack_weapon_degradation(self):
        """Test weapon degradation during attack."""
        with (
            patch("random.uniform", return_value=1.0),
            patch("random.random", return_value=0.05),
        ):  # Force degradation
            effects = await self.combat_manager.process_attack(
                "team_a_agent_1", "team_b_agent_1", self.world_state
            )

        # Should have damage effect and weapon degradation effect
        degradation_effect = next(e for e in effects if e["kind"] == "WeaponDegraded")

        assert degradation_effect["payload"]["agent_id"] == "team_a_agent_1"
        assert degradation_effect["payload"]["old_condition"] == "excellent"
        assert degradation_effect["payload"]["new_condition"] == "good"

        # Check that weapon was actually degraded
        assert self.agent_a.weapon_condition == WeaponCondition.GOOD

    @pytest.mark.asyncio
    async def test_process_heal_successful(self):
        """Test successful healing processing."""
        # Set up injured teammate
        teammate = Agent(
            agent_id="team_a_agent_2", team="team_a", position=(15.0, 15.0), health=50
        )
        self.world_state.agents["team_a_agent_2"] = teammate

        effects = await self.combat_manager.process_heal(
            "team_a_agent_1", "team_a_agent_2", self.world_state
        )

        assert len(effects) == 1
        heal_effect = effects[0]

        assert heal_effect["kind"] == "AgentHealed"
        assert heal_effect["payload"]["healer_id"] == "team_a_agent_1"
        assert heal_effect["payload"]["target_id"] == "team_a_agent_2"
        assert heal_effect["payload"]["heal_amount"] > 0
        assert heal_effect["payload"]["new_health"] > 50

        # Check that target was actually healed
        assert teammate.health > 50

    @pytest.mark.asyncio
    async def test_process_heal_self(self):
        """Test self-healing processing."""
        self.agent_a.health = 60

        effects = await self.combat_manager.process_heal(
            "team_a_agent_1", "team_a_agent_1", self.world_state
        )

        assert len(effects) == 1
        heal_effect = effects[0]

        assert heal_effect["kind"] == "AgentHealed"
        assert heal_effect["payload"]["is_self_heal"] is True
        assert heal_effect["payload"]["heal_amount"] > 0

        # Self-heal should be less effective
        expected_heal = int(GAME_CONFIG["heal_amount"] * 0.8)
        assert heal_effect["payload"]["heal_amount"] == expected_heal

    @pytest.mark.asyncio
    async def test_process_heal_invalid_target(self):
        """Test healing with invalid target."""
        effects = await self.combat_manager.process_heal(
            "team_a_agent_1", "team_b_agent_1", self.world_state
        )

        assert len(effects) == 1
        assert effects[0]["kind"] == "HealFailed"
        assert effects[0]["payload"]["reason"] == "cannot_heal_enemy"

    @pytest.mark.asyncio
    async def test_process_repair_successful(self):
        """Test successful weapon repair."""
        # Set up agent at forge with damaged weapon
        self.agent_a.position = (15.0, 15.0)  # At forge
        self.agent_a.weapon_condition = WeaponCondition.DAMAGED

        effects = await self.combat_manager.process_repair(
            "team_a_agent_1", self.world_state
        )

        assert len(effects) == 1
        repair_effect = effects[0]

        assert repair_effect["kind"] == "WeaponRepaired"
        assert repair_effect["payload"]["agent_id"] == "team_a_agent_1"
        assert repair_effect["payload"]["old_condition"] == "damaged"
        assert repair_effect["payload"]["new_condition"] == "excellent"

        # Check that weapon was actually repaired
        assert self.agent_a.weapon_condition == WeaponCondition.EXCELLENT

    @pytest.mark.asyncio
    async def test_process_repair_not_at_forge(self):
        """Test repair when not at forge."""
        # Agent is not at forge position
        self.agent_a.position = (50.0, 50.0)
        self.agent_a.weapon_condition = WeaponCondition.DAMAGED

        effects = await self.combat_manager.process_repair(
            "team_a_agent_1", self.world_state
        )

        assert len(effects) == 1
        assert effects[0]["kind"] == "RepairFailed"
        assert effects[0]["payload"]["reason"] == "not_at_forge"

    @pytest.mark.asyncio
    async def test_process_repair_weapon_already_excellent(self):
        """Test repair when weapon is already excellent."""
        # Set up agent at forge with excellent weapon
        self.agent_a.position = (15.0, 15.0)  # At forge
        self.agent_a.weapon_condition = WeaponCondition.EXCELLENT

        effects = await self.combat_manager.process_repair(
            "team_a_agent_1", self.world_state
        )

        assert len(effects) == 1
        assert effects[0]["kind"] == "RepairFailed"
        assert effects[0]["payload"]["reason"] == "weapon_already_excellent"

    @pytest.mark.asyncio
    async def test_process_communication_successful(self):
        """Test successful team communication."""
        message = "Enemy spotted at coordinates (25, 30)!"
        urgency = "high"

        effects = await self.combat_manager.process_communication(
            "team_a_agent_1", message, urgency, self.world_state
        )

        assert len(effects) == 1
        comm_effect = effects[0]

        assert comm_effect["kind"] == "TeamMessage"
        assert comm_effect["payload"]["sender_id"] == "team_a_agent_1"
        assert comm_effect["payload"]["sender_team"] == "team_a"
        assert comm_effect["payload"]["message"] == message
        assert comm_effect["payload"]["urgency"] == urgency
        assert comm_effect["payload"]["team_only"] is True

        # Check that message was added to world state
        team_messages = self.world_state.get_recent_team_messages("team_a", 1)
        assert len(team_messages) == 1
        assert team_messages[0].message == message

    @pytest.mark.asyncio
    async def test_process_communication_empty_message(self):
        """Test communication with empty message."""
        effects = await self.combat_manager.process_communication(
            "team_a_agent_1", "", "medium", self.world_state
        )

        assert len(effects) == 1
        assert effects[0]["kind"] == "CommunicationFailed"
        assert effects[0]["payload"]["reason"] == "empty_message"

    @pytest.mark.asyncio
    async def test_process_communication_invalid_urgency(self):
        """Test communication with invalid urgency level."""
        message = "Test message"

        effects = await self.combat_manager.process_communication(
            "team_a_agent_1", message, "invalid", self.world_state
        )

        assert len(effects) == 1
        comm_effect = effects[0]

        assert comm_effect["kind"] == "TeamMessage"
        assert comm_effect["payload"]["urgency"] == "medium"  # Should default to medium

    @pytest.mark.asyncio
    async def test_process_communication_sender_not_found(self):
        """Test communication with non-existent sender."""
        effects = await self.combat_manager.process_communication(
            "nonexistent", "Test message", "medium", self.world_state
        )

        assert len(effects) == 1
        assert effects[0]["kind"] == "CommunicationFailed"
        assert effects[0]["payload"]["reason"] == "sender_not_found"

    def test_combat_manager_initialization_default_mechanics(self):
        """Test CombatManager initialization with default mechanics."""
        manager = CombatManager()
        assert isinstance(manager.mechanics, BattleMechanics)

    def test_combat_manager_initialization_custom_mechanics(self):
        """Test CombatManager initialization with custom mechanics."""
        custom_mechanics = BattleMechanics()
        manager = CombatManager(custom_mechanics)
        assert manager.mechanics is custom_mechanics


class TestIntegrationScenarios:
    """Integration tests for complex battle scenarios."""

    def setup_method(self):
        """Set up complex test scenario."""
        self.mechanics = BattleMechanics()
        self.combat_manager = CombatManager(self.mechanics)

        # Create full team setup - position teams closer for combat
        self.agents = {}
        for team in ["team_a", "team_b"]:
            for i in range(3):
                agent_id = f"{team}_agent_{i + 1}"
                # Position teams within attack range (15.0)
                y_pos = 10.0 if team == "team_a" else 20.0  # 10 units apart
                self.agents[agent_id] = Agent(
                    agent_id=agent_id,
                    team=team,
                    position=(10.0 + i * 5, y_pos),
                    health=100,
                    weapon_condition=WeaponCondition.EXCELLENT,
                )

        # Create forges
        self.forges = {
            "forge_a": MapLocation(
                position=(5.0, 5.0), location_type=LocationType.FORGE, radius=5.0
            ),
            "forge_b": MapLocation(
                position=(25.0, 55.0), location_type=LocationType.FORGE, radius=5.0
            ),
        }

        self.world_state = BattleWorldState(
            agents=self.agents, map_locations=self.forges, game_time=0.0
        )

    @pytest.mark.asyncio
    async def test_full_combat_sequence(self):
        """Test a complete combat sequence with multiple actions."""
        # 1. Team A agent attacks Team B agent
        attack_effects = await self.combat_manager.process_attack(
            "team_a_agent_1", "team_b_agent_1", self.world_state
        )

        assert any(e["kind"] == "AgentDamaged" for e in attack_effects)

        # 2. Injured agent heals themselves
        injured_agent = self.agents["team_b_agent_1"]
        if injured_agent.health < 100:
            heal_effects = await self.combat_manager.process_heal(
                "team_b_agent_1", "team_b_agent_1", self.world_state
            )
            assert any(e["kind"] == "AgentHealed" for e in heal_effects)

        # 3. Agent moves to forge and repairs weapon (if degraded)
        attacker = self.agents["team_a_agent_1"]
        if attacker.weapon_condition != WeaponCondition.EXCELLENT:
            # Move to forge
            attacker.position = (5.0, 5.0)

            repair_effects = await self.combat_manager.process_repair(
                "team_a_agent_1", self.world_state
            )
            assert any(e["kind"] == "WeaponRepaired" for e in repair_effects)

        # 4. Team communication
        comm_effects = await self.combat_manager.process_communication(
            "team_a_agent_2", "Engaging enemy team!", "high", self.world_state
        )
        assert any(e["kind"] == "TeamMessage" for e in comm_effects)

    @pytest.mark.asyncio
    async def test_team_elimination_scenario(self):
        """Test scenario where one team is eliminated."""
        # Weaken all team_b agents
        for agent_id in ["team_b_agent_1", "team_b_agent_2", "team_b_agent_3"]:
            self.agents[agent_id].health = 1

        # Attack all team_b agents
        for i, target_id in enumerate(
            ["team_b_agent_1", "team_b_agent_2", "team_b_agent_3"]
        ):
            attacker_id = f"team_a_agent_{i + 1}"

            with patch("random.uniform", return_value=1.0):  # Ensure damage
                effects = await self.combat_manager.process_attack(
                    attacker_id, target_id, self.world_state
                )

            # Should have death effect
            assert any(e["kind"] == "AgentDied" for e in effects)

        # Check win condition
        assert self.world_state.check_win_condition() == "team_a_wins"

    @pytest.mark.asyncio
    async def test_weapon_degradation_and_repair_cycle(self):
        """Test complete weapon degradation and repair cycle."""
        agent = self.agents["team_a_agent_1"]
        target = self.agents["team_b_agent_1"]

        # Force weapon degradation through multiple attacks
        with patch("random.random", return_value=0.05):  # Force degradation
            for _ in range(10):  # Multiple attacks to degrade weapon
                with patch("random.uniform", return_value=1.0):
                    await self.combat_manager.process_attack(
                        agent.agent_id, target.agent_id, self.world_state
                    )

                if agent.weapon_condition == WeaponCondition.BROKEN:
                    break

        # Weapon should be degraded
        assert agent.weapon_condition != WeaponCondition.EXCELLENT

        # Move to forge and repair
        agent.position = (5.0, 5.0)  # At team_a forge
        repair_effects = await self.combat_manager.process_repair(
            agent.agent_id, self.world_state
        )

        assert any(e["kind"] == "WeaponRepaired" for e in repair_effects)
        assert agent.weapon_condition == WeaponCondition.EXCELLENT
