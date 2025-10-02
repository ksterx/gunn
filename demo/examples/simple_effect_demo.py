#!/usr/bin/env python3
"""
Simple Effect Processing Demo

This script demonstrates the effect processing system without complex imports.
"""

import asyncio
import os
import sys

# Add the project root to the path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from demo.backend.effect_processor import EffectProcessor, GameStatusManager
from demo.shared.enums import WeaponCondition
from demo.shared.models import Agent, BattleWorldState


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
        "team_b_agent_1": Agent(
            agent_id="team_b_agent_1",
            team="team_b",
            position=(90.0, 90.0),
            health=80,
            weapon_condition=WeaponCondition.GOOD,
        ),
    }

    # Create world state
    world_state = BattleWorldState(agents=agents, game_time=0.0)

    print(f"✅ Created world with {len(agents)} agents")

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
    print("\n⚔️  Simulating battle effects...")

    # Combat effects
    print("\n🔥 Processing combat effects")
    combat_effects = [
        {
            "kind": "AgentDamaged",
            "payload": {
                "attacker_id": "team_a_agent_1",
                "target_id": "team_b_agent_1",
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
                "message": "Taking heavy damage! Need support!",
                "urgency": "high",
                "timestamp": 1.0,
            },
        },
    ]

    result = await processor.process_effects(combat_effects, world_state)
    print(f"   ✅ Processed {result['processed_count']} effects")
    print(f"   📉 Team B Agent health: {agents['team_b_agent_1'].health} HP")
    print(
        f"   🔧 Team A Agent weapon: {agents['team_a_agent_1'].weapon_condition.value}"
    )
    print(f"   💬 Team B messages: {len(world_state.team_communications['team_b'])}")

    # Healing effects
    print("\n💚 Processing healing effects")
    heal_effects = [
        {
            "kind": "AgentHealed",
            "payload": {
                "healer_id": "team_b_agent_1",
                "target_id": "team_b_agent_1",
                "heal_amount": 20,
                "old_health": 50,
                "new_health": 70,
                "is_self_heal": True,
                "timestamp": 2.0,
            },
        }
    ]

    result = await processor.process_effects(heal_effects, world_state)
    print(f"   ✅ Processed {result['processed_count']} effects")
    print(f"   📈 Team B Agent health: {agents['team_b_agent_1'].health} HP")

    # Fatal combat
    print("\n💀 Processing fatal combat")
    fatal_effects = [
        {
            "kind": "AgentDamaged",
            "payload": {
                "attacker_id": "team_a_agent_1",
                "target_id": "team_b_agent_1",
                "damage": 70,
                "old_health": 70,
                "new_health": 0,
                "timestamp": 3.0,
            },
        },
        {
            "kind": "AgentDied",
            "payload": {
                "agent_id": "team_b_agent_1",
                "killer_id": "team_a_agent_1",
                "position": (90.0, 90.0),
                "timestamp": 3.0,
            },
        },
    ]

    result = await processor.process_effects(fatal_effects, world_state)
    print(f"   ✅ Processed {result['processed_count']} effects")
    print(f"   💀 Team B Agent status: {agents['team_b_agent_1'].status.value}")
    print(f"   🏆 Game status: {result['game_status']}")
    print(f"   📊 Team A score: {world_state.team_scores['team_a']}")

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

    # Show team communications
    print("\n💬 Team Communications")
    print("-" * 25)

    for team in ["team_a", "team_b"]:
        messages = world_state.get_recent_team_messages(team, 5)
        print(f"\n{team.upper()} Messages:")

        if messages:
            for i, msg in enumerate(messages, 1):
                print(f"  {i}. [{msg.urgency.upper()}] {msg.sender_id}: {msg.message}")
        else:
            print("  No messages")

    print("\n✅ Effect processing demonstration complete!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Agent damage and health updates")
    print("  ✓ Weapon degradation")
    print("  ✓ Team-only communication visibility")
    print("  ✓ Win condition detection")
    print("  ✓ Game statistics tracking")


if __name__ == "__main__":
    asyncio.run(main())
