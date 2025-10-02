#!/usr/bin/env python3
"""
Effect Processing Demo

This script demonstrates the effect processing system for the battle demo,
showing how effects are processed and how they update the world state.
"""

import asyncio
import sys
from pathlib import Path

# Add the demo directory to the path
demo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(demo_dir))

from backend.effect_processor import EffectProcessor, GameStatusManager
from shared.enums import LocationType, WeaponCondition
from shared.models import Agent, BattleWorldState, MapLocation


async def main():
    """Run the effect processing demonstration."""
    print("🎮 Battle Demo - Effect Processing System")
    print("=" * 50)

    # Initialize the effect processor and game status manager
    processor = EffectProcessor()
    status_manager = GameStatusManager()

    # Create a sample battle world
    print("\n📋 Setting up battle world...")

    # Create agents
    agents = {
        "team_a_agent_1": Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(10.0, 10.0),
            health=100,
            weapon_condition=WeaponCondition.EXCELLENT,
        ),
        "team_a_agent_2": Agent(
            agent_id="team_a_agent_2",
            team="team_a",
            position=(15.0, 10.0),
            health=100,
            weapon_condition=WeaponCondition.GOOD,
        ),
        "team_b_agent_1": Agent(
            agent_id="team_b_agent_1",
            team="team_b",
            position=(90.0, 90.0),
            health=100,
            weapon_condition=WeaponCondition.EXCELLENT,
        ),
        "team_b_agent_2": Agent(
            agent_id="team_b_agent_2",
            team="team_b",
            position=(85.0, 90.0),
            health=80,
            weapon_condition=WeaponCondition.DAMAGED,
        ),
    }

    # Create map locations
    map_locations = {
        "forge_a": MapLocation(
            position=(5.0, 5.0), location_type=LocationType.FORGE, radius=5.0
        ),
        "forge_b": MapLocation(
            position=(95.0, 95.0), location_type=LocationType.FORGE, radius=5.0
        ),
    }

    # Create world state
    world_state = BattleWorldState(
        agents=agents, map_locations=map_locations, game_time=0.0
    )

    print(
        f"✅ Created world with {len(agents)} agents and {len(map_locations)} locations"
    )

    # Display initial statistics
    initial_stats = status_manager.get_game_statistics(world_state)
    print(f"📊 Initial game status: {initial_stats['game_status']}")
    print(
        f"   Team A: {initial_stats['agents_alive']['team_a']} alive, {initial_stats['team_health']['team_a']} total HP"
    )
    print(
        f"   Team B: {initial_stats['agents_alive']['team_b']} alive, {initial_stats['team_health']['team_b']} total HP"
    )

    # Simulate a battle sequence
    print("\n⚔️  Simulating battle sequence...")

    # Round 1: Combat and communication
    print("\n🔥 Round 1: Initial combat")
    round1_effects = [
        {
            "kind": "AgentDamaged",
            "payload": {
                "attacker_id": "team_a_agent_1",
                "target_id": "team_b_agent_2",
                "damage": 30,
                "old_health": 80,
                "new_health": 50,
                "timestamp": 1.0,
            },
        },
        {
            "kind": "WeaponDegraded",
            "payload": {
                "agent_id": "team_a_agent_1",
                "old_condition": "excellent",
                "new_condition": "good",
                "timestamp": 1.0,
            },
        },
        {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_b_agent_1",
                "sender_team": "team_b",
                "message": "Agent 2 is injured! Providing cover fire!",
                "urgency": "high",
                "timestamp": 1.0,
            },
        },
    ]

    result1 = await processor.process_effects(round1_effects, world_state)
    print(f"   Processed {result1['processed_count']} effects")
    print(f"   Team B Agent 2 health: {agents['team_b_agent_2'].health} HP")
    print(
        f"   Team A Agent 1 weapon: {agents['team_a_agent_1'].weapon_condition.value}"
    )
    print(f"   Team B messages: {len(world_state.team_communications['team_b'])}")

    # Round 2: Healing and repositioning
    print("\n💚 Round 2: Healing and tactical movement")
    round2_effects = [
        {
            "kind": "AgentHealed",
            "payload": {
                "healer_id": "team_b_agent_1",
                "target_id": "team_b_agent_2",
                "heal_amount": 25,
                "old_health": 50,
                "new_health": 75,
                "timestamp": 2.0,
            },
        },
        {
            "kind": "Move",
            "source_id": "team_b_agent_2",
            "payload": {
                "target_position": (95.0, 95.0),  # Move to forge
                "reason": "Moving to forge for weapon repair",
            },
        },
        {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_a_agent_2",
                "sender_team": "team_a",
                "message": "Enemy is retreating to their forge. Advance!",
                "urgency": "medium",
                "timestamp": 2.0,
            },
        },
    ]

    result2 = await processor.process_effects(round2_effects, world_state)
    print(f"   Processed {result2['processed_count']} effects")
    print(f"   Team B Agent 2 health: {agents['team_b_agent_2'].health} HP")
    print(f"   Team B Agent 2 position: {agents['team_b_agent_2'].position}")
    print(f"   Team A messages: {len(world_state.team_communications['team_a'])}")

    # Round 3: Weapon repair and counter-attack
    print("\n🔧 Round 3: Weapon repair and counter-attack")
    round3_effects = [
        {
            "kind": "WeaponRepaired",
            "payload": {
                "agent_id": "team_b_agent_2",
                "old_condition": "damaged",
                "new_condition": "excellent",
                "forge_location": "forge_b",
                "timestamp": 3.0,
            },
        },
        {
            "kind": "AgentDamaged",
            "payload": {
                "attacker_id": "team_b_agent_2",
                "target_id": "team_a_agent_1",
                "damage": 40,
                "old_health": 100,
                "new_health": 60,
                "timestamp": 3.0,
            },
        },
    ]

    result3 = await processor.process_effects(round3_effects, world_state)
    print(f"   Processed {result3['processed_count']} effects")
    print(
        f"   Team B Agent 2 weapon: {agents['team_b_agent_2'].weapon_condition.value}"
    )
    print(f"   Team A Agent 1 health: {agents['team_a_agent_1'].health} HP")

    # Round 4: Fatal combat
    print("\n💀 Round 4: Fatal combat")
    round4_effects = [
        {
            "kind": "AgentDamaged",
            "payload": {
                "attacker_id": "team_b_agent_1",
                "target_id": "team_a_agent_2",
                "damage": 100,
                "old_health": 100,
                "new_health": 0,
                "timestamp": 4.0,
            },
        },
        {
            "kind": "AgentDied",
            "payload": {
                "agent_id": "team_a_agent_2",
                "killer_id": "team_b_agent_1",
                "position": (15.0, 10.0),
                "timestamp": 4.0,
            },
        },
        {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_b_agent_1",
                "sender_team": "team_b",
                "message": "Enemy agent eliminated! Press the advantage!",
                "urgency": "high",
                "timestamp": 4.0,
            },
        },
    ]

    result4 = await processor.process_effects(round4_effects, world_state)
    print(f"   Processed {result4['processed_count']} effects")
    print(f"   Team A Agent 2 status: {agents['team_a_agent_2'].status.value}")
    print(f"   Team B score: {world_state.team_scores['team_b']}")
    print(f"   Game status: {result4['game_status']}")

    # Display final statistics
    print("\n📊 Final Battle Statistics")
    print("-" * 30)
    final_stats = status_manager.get_game_statistics(world_state)

    print(f"Game Status: {final_stats['game_status']}")
    print(f"Game Time: {world_state.game_time:.1f}s")
    print()

    print("Team Statistics:")
    for team in ["team_a", "team_b"]:
        alive = final_stats["agents_alive"][team]
        total_hp = final_stats["team_health"][team]
        score = final_stats["team_scores"][team]
        messages = final_stats["communication_counts"][team]

        print(f"  {team.upper()}:")
        print(f"    Agents alive: {alive}/{final_stats['total_agents'][team]}")
        print(f"    Total health: {total_hp} HP")
        print(f"    Score: {score}")
        print(f"    Messages sent: {messages}")

    print("\nWeapon Conditions:")
    for team in ["team_a", "team_b"]:
        conditions = final_stats["weapon_conditions"][team]
        print(f"  {team.upper()}: {dict(conditions)}")

    # Show team communications (demonstrating team-only visibility)
    print("\n💬 Team Communications (Team-Only Visibility)")
    print("-" * 50)

    for team in ["team_a", "team_b"]:
        messages = world_state.get_recent_team_messages(team, 5)
        print(f"\n{team.upper()} Messages ({len(messages)} total):")

        if messages:
            for i, msg in enumerate(messages, 1):
                print(f"  {i}. [{msg.urgency.upper()}] {msg.sender_id}: {msg.message}")
        else:
            print("  No messages")

    print("\n✅ Effect processing demonstration complete!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Agent damage and health updates")
    print("  ✓ Weapon degradation and repair")
    print("  ✓ Agent movement and positioning")
    print("  ✓ Team-only communication visibility")
    print("  ✓ Win condition detection")
    print("  ✓ Comprehensive game statistics")
    print("  ✓ State consistency and validation")


if __name__ == "__main__":
    asyncio.run(main())
