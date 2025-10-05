"""Demo of intelligent staleness detection in SpatialObservationPolicy.

This demo showcases how the SpatialObservationPolicy prevents false positives
by only triggering staleness when relevant preconditions change:

1. Move intents: Only stale if agent position or target area changes
2. Speak intents: Only stale if nearby agents change (join/leave conversation)
3. Unrelated changes: Don't trigger staleness

This improves efficiency by avoiding unnecessary LLM generation cancellations.
"""

import asyncio

from gunn.policies.observation import (
    PolicyConfig,
    SpatialObservationPolicy,
    StalenessConfig,
)
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Intent


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_result(description: str, is_stale: bool) -> None:
    """Print staleness check result."""
    status = "🔴 STALE" if is_stale else "✅ NOT STALE"
    print(f"{description}: {status}")


async def demo_move_intent_staleness():
    """Demonstrate Move intent staleness detection."""
    print_section("Move Intent Staleness Detection")

    policy = SpatialObservationPolicy(
        PolicyConfig(distance_limit=10.0),
        StalenessConfig(move_position_threshold=1.0),
    )

    # Scenario 1: Agent position unchanged
    print("Scenario 1: Agent position unchanged")
    old_state = WorldState(
        entities={"agent1": {"type": "agent"}},
        spatial_index={"agent1": (0.0, 0.0, 0.0)},
    )
    new_state = WorldState(
        entities={"agent1": {"type": "agent"}},
        spatial_index={"agent1": (0.0, 0.0, 0.0)},
    )

    move_intent: Intent = {
        "kind": "Move",
        "payload": {"to": [10.0, 5.0, 0.0]},
        "context_seq": 1,
        "req_id": "req1",
        "agent_id": "agent1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    is_stale = policy.is_intent_stale(move_intent, old_state, new_state)
    print_result("Agent hasn't moved", is_stale)
    print("  → Intent remains valid, no cancellation needed\n")

    # Scenario 2: Agent moved significantly
    print("Scenario 2: Agent moved 2 units (beyond 1.0 threshold)")
    new_state_moved = WorldState(
        entities={"agent1": {"type": "agent"}},
        spatial_index={"agent1": (2.0, 0.0, 0.0)},
    )

    is_stale = policy.is_intent_stale(move_intent, old_state, new_state_moved)
    print_result("Agent moved significantly", is_stale)
    print("  → Intent is stale, should cancel and regenerate\n")

    # Scenario 3: Obstacle appeared near target
    print("Scenario 3: New obstacle appeared near target position")
    new_state_obstacle = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "obstacle1": {"type": "obstacle"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "obstacle1": (10.5, 5.0, 0.0),  # Near target [10, 5, 0]
        },
    )

    is_stale = policy.is_intent_stale(move_intent, old_state, new_state_obstacle)
    print_result("Obstacle appeared near target", is_stale)
    print("  → Intent is stale, path may be blocked\n")

    # Scenario 4: Unrelated agent moved far away
    print("Scenario 4: Unrelated agent moved far away")
    old_state_multi = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "agent2": (50.0, 0.0, 0.0),
        },
    )
    new_state_multi = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "agent2": (60.0, 0.0, 0.0),  # Moved, but far away
        },
    )

    is_stale = policy.is_intent_stale(move_intent, old_state_multi, new_state_multi)
    print_result("Unrelated agent moved far away", is_stale)
    print("  → Intent NOT stale, prevents false positive!\n")


async def demo_speak_intent_staleness():
    """Demonstrate Speak intent staleness detection."""
    print_section("Speak Intent Staleness Detection")

    policy = SpatialObservationPolicy(
        PolicyConfig(distance_limit=10.0),
        StalenessConfig(speak_proximity_threshold=5.0),
    )

    speak_intent: Intent = {
        "kind": "Speak",
        "payload": {"text": "Hello everyone!"},
        "context_seq": 1,
        "req_id": "req1",
        "agent_id": "agent1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    # Scenario 1: Nearby agents unchanged
    print("Scenario 1: Nearby agents unchanged")
    old_state = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "agent2": (3.0, 0.0, 0.0),
        },
    )
    new_state = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "agent2": (3.0, 0.0, 0.0),
        },
    )

    is_stale = policy.is_intent_stale(speak_intent, old_state, new_state)
    print_result("Same conversation participants", is_stale)
    print("  → Intent remains valid, audience unchanged\n")

    # Scenario 2: New agent joined conversation
    print("Scenario 2: New agent joined conversation range")
    new_state_joined = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
            "agent3": {"type": "agent"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "agent2": (3.0, 0.0, 0.0),
            "agent3": (4.0, 0.0, 0.0),  # Within 5.0 threshold
        },
    )

    is_stale = policy.is_intent_stale(speak_intent, old_state, new_state_joined)
    print_result("Agent joined conversation", is_stale)
    print("  → Intent is stale, should regenerate for new audience\n")

    # Scenario 3: Agent left conversation
    print("Scenario 3: Agent left conversation range")
    old_state_multi = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
            "agent3": {"type": "agent"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "agent2": (3.0, 0.0, 0.0),
            "agent3": (4.0, 0.0, 0.0),
        },
    )
    new_state_left = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
            "agent3": {"type": "agent"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "agent2": (3.0, 0.0, 0.0),
            "agent3": (10.0, 0.0, 0.0),  # Beyond 5.0 threshold
        },
    )

    is_stale = policy.is_intent_stale(speak_intent, old_state_multi, new_state_left)
    print_result("Agent left conversation", is_stale)
    print("  → Intent is stale, audience changed\n")

    # Scenario 4: Non-agent entity appeared
    print("Scenario 4: Non-agent entity appeared nearby")
    new_state_tree = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
            "tree1": {"type": "tree"},  # Not an agent
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "agent2": (3.0, 0.0, 0.0),
            "tree1": (2.0, 0.0, 0.0),
        },
    )

    is_stale = policy.is_intent_stale(speak_intent, old_state, new_state_tree)
    print_result("Non-agent entity appeared", is_stale)
    print("  → Intent NOT stale, prevents false positive!\n")


async def demo_agent_specific_thresholds():
    """Demonstrate agent-specific staleness thresholds."""
    print_section("Agent-Specific Staleness Thresholds")

    # Create policy with agent-specific thresholds
    policy = SpatialObservationPolicy(
        PolicyConfig(distance_limit=10.0),
        StalenessConfig(
            move_position_threshold=1.0,  # Default threshold
            agent_specific_thresholds={
                "vip_agent": {"Move": 5.0},  # VIP gets higher threshold
            },
        ),
    )

    old_state = WorldState(
        entities={
            "regular_agent": {"type": "agent"},
            "vip_agent": {"type": "agent"},
        },
        spatial_index={
            "regular_agent": (0.0, 0.0, 0.0),
            "vip_agent": (0.0, 0.0, 0.0),
        },
    )

    # Both agents moved 2 units
    new_state = WorldState(
        entities={
            "regular_agent": {"type": "agent"},
            "vip_agent": {"type": "agent"},
        },
        spatial_index={
            "regular_agent": (2.0, 0.0, 0.0),
            "vip_agent": (2.0, 0.0, 0.0),
        },
    )

    # Regular agent intent (2 units > 1.0 threshold)
    regular_intent: Intent = {
        "kind": "Move",
        "payload": {"to": [10.0, 0.0, 0.0]},
        "context_seq": 1,
        "req_id": "req1",
        "agent_id": "regular_agent",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    # VIP agent intent (2 units < 5.0 threshold)
    vip_intent: Intent = {
        "kind": "Move",
        "payload": {"to": [10.0, 0.0, 0.0]},
        "context_seq": 1,
        "req_id": "req2",
        "agent_id": "vip_agent",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    print("Both agents moved 2 units:")
    is_stale_regular = policy.is_intent_stale(regular_intent, old_state, new_state)
    print_result("Regular agent (threshold: 1.0)", is_stale_regular)

    is_stale_vip = policy.is_intent_stale(vip_intent, old_state, new_state)
    print_result("VIP agent (threshold: 5.0)", is_stale_vip)

    print("\n  → Agent-specific thresholds allow fine-grained control!")


async def demo_efficiency_comparison():
    """Demonstrate efficiency improvement over naive staleness."""
    print_section("Efficiency Comparison: Intelligent vs Naive Staleness")

    policy = SpatialObservationPolicy(
        PolicyConfig(distance_limit=10.0),
        StalenessConfig(
            move_position_threshold=1.0,
            speak_proximity_threshold=5.0,
        ),
    )

    # Simulate a busy world with many agents
    old_state = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
            "agent3": {"type": "agent"},
            "agent4": {"type": "agent"},
            "agent5": {"type": "agent"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),
            "agent2": (50.0, 0.0, 0.0),
            "agent3": (100.0, 0.0, 0.0),
            "agent4": (150.0, 0.0, 0.0),
            "agent5": (200.0, 0.0, 0.0),
        },
    )

    # Many agents moved, but far from agent1
    new_state = WorldState(
        entities={
            "agent1": {"type": "agent"},
            "agent2": {"type": "agent"},
            "agent3": {"type": "agent"},
            "agent4": {"type": "agent"},
            "agent5": {"type": "agent"},
        },
        spatial_index={
            "agent1": (0.0, 0.0, 0.0),  # agent1 unchanged
            "agent2": (55.0, 0.0, 0.0),  # Moved
            "agent3": (105.0, 0.0, 0.0),  # Moved
            "agent4": (155.0, 0.0, 0.0),  # Moved
            "agent5": (205.0, 0.0, 0.0),  # Moved
        },
    )

    agent1_move: Intent = {
        "kind": "Move",
        "payload": {"to": [5.0, 0.0, 0.0]},
        "context_seq": 1,
        "req_id": "req1",
        "agent_id": "agent1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    agent1_speak: Intent = {
        "kind": "Speak",
        "payload": {"text": "Hello!"},
        "context_seq": 1,
        "req_id": "req2",
        "agent_id": "agent1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    print("World state: 4 agents moved far from agent1")
    print("\nNaive staleness detection:")
    print("  ❌ Would cancel agent1's intents (any world change = stale)")
    print("  ❌ Wastes LLM generation for unrelated changes")

    print("\nIntelligent staleness detection:")
    move_stale = policy.is_intent_stale(agent1_move, old_state, new_state)
    speak_stale = policy.is_intent_stale(agent1_speak, old_state, new_state)

    print_result("  agent1 Move intent", move_stale)
    print_result("  agent1 Speak intent", speak_stale)
    print("  ✅ Prevents false positives, saves LLM costs!")


async def main():
    """Run all staleness detection demos."""
    print("\n" + "=" * 70)
    print("  Intelligent Staleness Detection Demo")
    print("  Preventing False Positives in Multi-Agent Simulations")
    print("=" * 70)

    await demo_move_intent_staleness()
    await demo_speak_intent_staleness()
    await demo_agent_specific_thresholds()
    await demo_efficiency_comparison()

    print_section("Summary")
    print("Intelligent staleness detection provides:")
    print("  ✅ Intent-specific logic (Move, Speak, Custom)")
    print("  ✅ Spatial awareness (position, proximity)")
    print("  ✅ False positive prevention")
    print("  ✅ Agent-specific thresholds")
    print("  ✅ Improved efficiency and reduced LLM costs")
    print("\nThis enables more efficient multi-agent simulations!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
