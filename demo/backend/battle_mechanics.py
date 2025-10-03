"""
Battle mechanics and combat system.

This module handles combat calculations, weapon degradation, healing mechanics,
and other game rules for the multi-agent battle simulation.
"""

import random
from typing import Any

from ..shared.constants import GAME_CONFIG
from ..shared.enums import AgentStatus, WeaponCondition
from ..shared.models import Agent, BattleWorldState, MapLocation
from ..shared.utils import calculate_distance


class BattleMechanics:
    """Handles combat calculations and game rules."""

    def __init__(self):
        """Initialize battle mechanics with configuration from constants."""
        self.attack_damage = GAME_CONFIG["attack_damage"]
        self.heal_amount = GAME_CONFIG["heal_amount"]
        self.weapon_degradation_rate = GAME_CONFIG["weapon_degradation_rate"]
        self.movement_speed = GAME_CONFIG["movement_speed"]
        self.attack_cooldown = GAME_CONFIG["attack_cooldown"]
        self.heal_cooldown = GAME_CONFIG["heal_cooldown"]

    def calculate_attack_damage(
        self, attacker: Agent, target: Agent, distance: float
    ) -> int:
        """
        Calculate damage based on weapon condition and distance.

        Args:
            attacker: The attacking agent
            target: The target agent
            distance: Distance between attacker and target

        Returns:
            Damage amount (0 if attack is invalid)
        """
        if attacker.weapon_condition == WeaponCondition.BROKEN:
            return 0

        base_damage = self.attack_damage

        # Weapon condition modifier
        condition_modifiers = {
            WeaponCondition.EXCELLENT: 1.0,
            WeaponCondition.GOOD: 0.8,
            WeaponCondition.DAMAGED: 0.6,
            WeaponCondition.BROKEN: 0.0,
        }

        damage = base_damage * condition_modifiers[attacker.weapon_condition]

        # Distance modifier (closer = more damage)
        if distance <= 5.0:
            damage *= 1.2  # Close range bonus
        elif distance > attacker.attack_range:
            damage = 0  # Out of range

        # Add some randomness (±20%)
        damage *= random.uniform(0.8, 1.2)

        return max(0, int(damage))

    def degrade_weapon(self, agent: Agent) -> WeaponCondition:
        """
        Degrade weapon condition after use.

        Args:
            agent: The agent whose weapon to degrade

        Returns:
            New weapon condition
        """
        conditions = [
            WeaponCondition.EXCELLENT,
            WeaponCondition.GOOD,
            WeaponCondition.DAMAGED,
            WeaponCondition.BROKEN,
        ]
        current_index = conditions.index(agent.weapon_condition)

        if random.random() < self.weapon_degradation_rate:
            new_index = min(current_index + 1, len(conditions) - 1)
            agent.weapon_condition = conditions[new_index]

        return agent.weapon_condition

    def can_perform_action(
        self, agent: Agent, action_type: str, current_time: float
    ) -> bool:
        """
        Check if agent can perform the requested action.

        Args:
            agent: The agent attempting the action
            action_type: Type of action ("attack", "heal", "move", "repair", "communicate")
            current_time: Current game time

        Returns:
            True if action is allowed, False otherwise
        """
        if agent.status != AgentStatus.ALIVE:
            return False

        time_since_last = current_time - agent.last_action_time

        if action_type == "attack":
            return (
                time_since_last >= self.attack_cooldown
                and agent.weapon_condition != WeaponCondition.BROKEN
            )
        elif action_type == "heal":
            return time_since_last >= self.heal_cooldown
        elif action_type in ["move", "repair", "communicate"]:
            return True  # These actions have no cooldown restrictions

        return False

    def calculate_movement_time(
        self, start_pos: tuple[float, float], end_pos: tuple[float, float]
    ) -> float:
        """
        Calculate time needed for movement.

        Args:
            start_pos: Starting position
            end_pos: Ending position

        Returns:
            Time in seconds needed for movement
        """
        distance = calculate_distance(start_pos, end_pos)
        return distance / self.movement_speed

    def is_at_forge(
        self, agent: Agent, forge_locations: dict[str, MapLocation]
    ) -> bool:
        """
        Check if agent is at their team's forge.

        Args:
            agent: The agent to check
            forge_locations: Dictionary of forge locations

        Returns:
            True if agent is at their team's forge
        """
        team_forge = f"forge_{agent.team.split('_')[1]}"  # team_a -> forge_a

        if team_forge not in forge_locations:
            return False

        forge = forge_locations[team_forge]
        distance = calculate_distance(agent.position, forge.position)

        return distance <= forge.radius

    def calculate_heal_amount(self, healer: Agent, target: Agent) -> int:
        """
        Calculate healing amount based on healer and target conditions.

        Args:
            healer: The agent performing the heal
            target: The agent being healed

        Returns:
            Amount of health to restore
        """
        base_heal = self.heal_amount

        # Self-heal is less effective
        if healer.agent_id == target.agent_id:
            base_heal = int(base_heal * 0.8)

        # Can't heal beyond max health
        max_possible_heal = 100 - target.health

        return min(base_heal, max_possible_heal)

    def is_valid_attack_target(
        self, attacker: Agent, target: Agent, world_state: BattleWorldState
    ) -> tuple[bool, str]:
        """
        Validate if an attack target is valid.

        Args:
            attacker: The attacking agent
            target: The target agent
            world_state: Current world state

        Returns:
            Tuple of (is_valid, reason)
        """
        # Can't attack teammates
        if attacker.team == target.team:
            return False, "cannot_attack_teammate"

        # Can't attack dead agents
        if not target.is_alive():
            return False, "target_is_dead"

        # Check range
        distance = calculate_distance(attacker.position, target.position)
        if distance > attacker.attack_range:
            return False, "out_of_range"

        # Check weapon condition
        if attacker.weapon_condition == WeaponCondition.BROKEN:
            return False, "weapon_broken"

        return True, "valid"

    def is_valid_heal_target(self, healer: Agent, target: Agent) -> tuple[bool, str]:
        """
        Validate if a heal target is valid.

        Args:
            healer: The healing agent
            target: The target agent

        Returns:
            Tuple of (is_valid, reason)
        """
        # Can only heal teammates or self
        if healer.team != target.team:
            return False, "cannot_heal_enemy"

        # Can't heal dead agents
        if not target.is_alive():
            return False, "target_is_dead"

        # Can't heal if already at full health
        if target.health >= 100:
            return False, "target_at_full_health"

        return True, "valid"


class CombatManager:
    """Manages combat resolution and effects."""

    def __init__(self, battle_mechanics: BattleMechanics | None = None):
        """
        Initialize combat manager.

        Args:
            battle_mechanics: Battle mechanics instance (creates new if None)
        """
        self.mechanics = battle_mechanics or BattleMechanics()

    async def process_attack(
        self, attacker_id: str, target_id: str, world_state: BattleWorldState
    ) -> list[dict[str, Any]]:
        """
        Process an attack and return resulting effects.

        Args:
            attacker_id: ID of the attacking agent
            target_id: ID of the target agent
            world_state: Current world state

        Returns:
            List of effect dictionaries
        """
        effects = []

        attacker = world_state.agents.get(attacker_id)
        target = world_state.agents.get(target_id)

        if not attacker or not target:
            effects.append(
                {
                    "kind": "AttackFailed",
                    "payload": {
                        "attacker_id": attacker_id,
                        "target_id": target_id,
                        "reason": "agent_not_found",
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )
            return effects

        # Validate attack
        is_valid, reason = self.mechanics.is_valid_attack_target(
            attacker, target, world_state
        )
        if not is_valid:
            effects.append(
                {
                    "kind": "AttackFailed",
                    "payload": {
                        "attacker_id": attacker_id,
                        "target_id": target_id,
                        "reason": reason,
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )
            return effects

        # Calculate distance and damage
        distance = calculate_distance(attacker.position, target.position)
        damage = self.mechanics.calculate_attack_damage(attacker, target, distance)

        if damage > 0:
            # Apply damage
            old_health = target.health
            new_health = max(0, target.health - damage)
            target.health = new_health

            effects.append(
                {
                    "kind": "AgentDamaged",
                    "payload": {
                        "attacker_id": attacker_id,
                        "target_id": target_id,
                        "damage": damage,
                        "old_health": old_health,
                        "new_health": new_health,
                        "position": target.position,
                        "distance": distance,
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )

            # Check if target died
            if new_health == 0:
                target.status = AgentStatus.DEAD
                effects.append(
                    {
                        "kind": "AgentDied",
                        "payload": {
                            "agent_id": target_id,
                            "killer_id": attacker_id,
                            "position": target.position,
                            "timestamp": world_state.game_time,
                        },
                        "source_id": "combat_manager",
                        "schema_version": "1.0.0",
                    }
                )

        # Degrade attacker's weapon
        old_condition = attacker.weapon_condition
        new_condition = self.mechanics.degrade_weapon(attacker)

        if old_condition != new_condition:
            effects.append(
                {
                    "kind": "WeaponDegraded",
                    "payload": {
                        "agent_id": attacker_id,
                        "old_condition": old_condition.value,
                        "new_condition": new_condition.value,
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )

        # Update attacker's last action time
        attacker.last_action_time = world_state.game_time

        return effects

    async def process_heal(
        self, healer_id: str, target_id: str, world_state: BattleWorldState
    ) -> list[dict[str, Any]]:
        """
        Process a healing action and return resulting effects.

        Args:
            healer_id: ID of the healing agent
            target_id: ID of the target agent (can be same as healer)
            world_state: Current world state

        Returns:
            List of effect dictionaries
        """
        effects = []

        healer = world_state.agents.get(healer_id)
        target = world_state.agents.get(target_id)

        if not healer or not target:
            effects.append(
                {
                    "kind": "HealFailed",
                    "payload": {
                        "healer_id": healer_id,
                        "target_id": target_id,
                        "reason": "agent_not_found",
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )
            return effects

        # Validate heal
        is_valid, reason = self.mechanics.is_valid_heal_target(healer, target)
        if not is_valid:
            effects.append(
                {
                    "kind": "HealFailed",
                    "payload": {
                        "healer_id": healer_id,
                        "target_id": target_id,
                        "reason": reason,
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )
            return effects

        # Calculate and apply healing
        heal_amount = self.mechanics.calculate_heal_amount(healer, target)
        old_health = target.health
        new_health = min(100, target.health + heal_amount)
        actual_heal = new_health - target.health
        target.health = new_health

        effects.append(
            {
                "kind": "AgentHealed",
                "payload": {
                    "healer_id": healer_id,
                    "target_id": target_id,
                    "heal_amount": actual_heal,
                    "old_health": old_health,
                    "new_health": new_health,
                    "position": target.position,
                    "is_self_heal": healer_id == target_id,
                    "timestamp": world_state.game_time,
                },
                "source_id": "combat_manager",
                "schema_version": "1.0.0",
            }
        )

        # Update healer's last action time
        healer.last_action_time = world_state.game_time

        return effects

    async def process_repair(
        self, agent_id: str, world_state: BattleWorldState
    ) -> list[dict[str, Any]]:
        """
        Process weapon repair at forge and return resulting effects.

        Args:
            agent_id: ID of the agent repairing their weapon
            world_state: Current world state

        Returns:
            List of effect dictionaries
        """
        effects = []

        agent = world_state.agents.get(agent_id)
        if not agent:
            effects.append(
                {
                    "kind": "RepairFailed",
                    "payload": {
                        "agent_id": agent_id,
                        "reason": "agent_not_found",
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )
            return effects

        # Check if agent is at their team's forge
        if not self.mechanics.is_at_forge(agent, world_state.map_locations):
            effects.append(
                {
                    "kind": "RepairFailed",
                    "payload": {
                        "agent_id": agent_id,
                        "reason": "not_at_forge",
                        "required_location": f"forge_{agent.team.split('_')[1]}",
                        "agent_position": agent.position,
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )
            return effects

        # Check if weapon needs repair
        if agent.weapon_condition == WeaponCondition.EXCELLENT:
            effects.append(
                {
                    "kind": "RepairFailed",
                    "payload": {
                        "agent_id": agent_id,
                        "reason": "weapon_already_excellent",
                        "current_condition": agent.weapon_condition.value,
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )
            return effects

        # Repair weapon to excellent condition
        old_condition = agent.weapon_condition
        agent.weapon_condition = WeaponCondition.EXCELLENT

        effects.append(
            {
                "kind": "WeaponRepaired",
                "payload": {
                    "agent_id": agent_id,
                    "old_condition": old_condition.value,
                    "new_condition": WeaponCondition.EXCELLENT.value,
                    "forge_location": f"forge_{agent.team.split('_')[1]}",
                    "agent_position": agent.position,
                    "timestamp": world_state.game_time,
                },
                "source_id": "combat_manager",
                "schema_version": "1.0.0",
            }
        )

        # Update agent's last action time
        agent.last_action_time = world_state.game_time

        return effects

    async def process_communication(
        self, sender_id: str, message: str, urgency: str, world_state: BattleWorldState
    ) -> list[dict[str, Any]]:
        """
        Process team communication and return resulting effects.

        Args:
            sender_id: ID of the sending agent
            message: Message content
            urgency: Message urgency level ("low", "medium", "high")
            world_state: Current world state

        Returns:
            List of effect dictionaries
        """
        effects = []

        sender = world_state.agents.get(sender_id)
        if not sender:
            effects.append(
                {
                    "kind": "CommunicationFailed",
                    "payload": {
                        "sender_id": sender_id,
                        "reason": "sender_not_found",
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )
            return effects

        # Validate message
        if not message or not message.strip():
            effects.append(
                {
                    "kind": "CommunicationFailed",
                    "payload": {
                        "sender_id": sender_id,
                        "reason": "empty_message",
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "combat_manager",
                    "schema_version": "1.0.0",
                }
            )
            return effects

        # Validate urgency
        if urgency not in ["low", "medium", "high"]:
            urgency = "medium"  # Default to medium if invalid

        # Add message to world state
        world_state.add_team_message(sender_id, message.strip(), urgency)

        # Create team message effect
        effects.append(
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": sender_id,
                    "sender_team": sender.team,
                    "message": message.strip(),
                    "urgency": urgency,
                    "timestamp": world_state.game_time,
                    "team_only": True,  # Ensures only team members can see this
                },
                "source_id": "combat_manager",
                "schema_version": "1.0.0",
            }
        )

        return effects
