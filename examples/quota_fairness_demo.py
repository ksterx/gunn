"""Demonstration of quota management and priority fairness system.

This example shows how to use:
1. QuotaController for rate limiting
2. PriorityAging to prevent starvation
3. WeightedRoundRobinScheduler with aging
"""

import asyncio
import time

from gunn.schemas.types import Intent
from gunn.utils import (
    AgingPolicy,
    PriorityAging,
    QuotaController,
    QuotaExceededError,
    QuotaPolicy,
    WeightedRoundRobinScheduler,
)


def create_intent(agent_id: str, req_id: str, text: str) -> Intent:
    """Create a sample intent."""
    return {
        "kind": "Speak",
        "payload": {"text": text},
        "context_seq": 0,
        "req_id": req_id,
        "agent_id": agent_id,
        "priority": 0,
        "schema_version": "1.0.0",
    }


async def demo_quota_controller():
    """Demonstrate quota controller functionality."""
    print("\n=== Quota Controller Demo ===\n")

    # Configure quota policy
    policy = QuotaPolicy(
        intents_per_minute=10,  # Low limit for demo
        tokens_per_minute=1000,
        burst_size=3,  # Allow small bursts
    )

    controller = QuotaController(policy)

    # Agent 1: Burst of intents
    print("Agent 1: Attempting burst of intents...")
    for i in range(5):
        try:
            await controller.check_intent_quota("agent1")
            print(f"  Intent {i + 1}: ✓ Allowed")
        except QuotaExceededError as e:
            print(f"  Intent {i + 1}: ✗ Quota exceeded - {e}")

    # Check available quota
    available = await controller.get_available_intents("agent1")
    print(f"\nAgent 1 available intents: {available:.2f}")

    # Agent 2: Independent quota
    print("\nAgent 2: Attempting intents (independent quota)...")
    for i in range(3):
        try:
            await controller.check_intent_quota("agent2")
            print(f"  Intent {i + 1}: ✓ Allowed")
        except QuotaExceededError:
            print(f"  Intent {i + 1}: ✗ Quota exceeded")

    # Show statistics
    stats = controller.get_stats()
    print("\nQuota Statistics:")
    print(f"  Total intents: {stats['total_intents']}")
    print(f"  Quota exceeded: {stats['quota_exceeded_count']}")


def demo_priority_aging():
    """Demonstrate priority aging system."""
    print("\n=== Priority Aging Demo ===\n")

    # Configure aggressive aging for demo
    policy = AgingPolicy(
        aging_rate=2.0,  # Fast aging
        max_priority_boost=5,
        min_wait_time_seconds=0.5,
    )

    aging = PriorityAging(policy)

    # Track some intents
    intent1 = create_intent("agent1", "req1", "First message")
    intent2 = create_intent("agent2", "req2", "Second message")

    aging.track_intent(intent1, priority=1)
    print("Tracked intent1 with priority 1")

    time.sleep(0.6)  # Wait for aging to start

    aging.track_intent(intent2, priority=1)
    print("Tracked intent2 with priority 1")

    time.sleep(0.3)  # Let first intent age more

    # Check aged priorities
    aged1 = aging.get_aged_priority("req1")
    aged2 = aging.get_aged_priority("req2")

    print("\nAged priorities:")
    print(f"  Intent1: {aged1} (waited longer)")
    print(f"  Intent2: {aged2} (waited less)")

    # Show statistics
    stats = aging.get_stats()
    print("\nAging Statistics:")
    print(f"  Tracked intents: {stats['tracked_intents']}")
    print(f"  Total aged: {stats['total_aged']}")
    print(f"  Max boost applied: {stats['max_boost_applied']}")
    print(f"  Avg wait time: {stats['avg_wait_time']:.2f}s")


def demo_scheduler_with_aging():
    """Demonstrate scheduler with priority aging."""
    print("\n=== Scheduler with Aging Demo ===\n")

    # Configure aging
    aging_policy = AgingPolicy(
        aging_rate=5.0,  # Very fast for demo
        max_priority_boost=10,
        min_wait_time_seconds=0.1,
    )

    # Create scheduler
    scheduler = WeightedRoundRobinScheduler(
        default_weight=1,
        priority_levels=3,
        aging_policy=aging_policy,
    )

    # Set different weights for agents
    scheduler.set_agent_weight("agent1", 2)  # Higher weight
    scheduler.set_agent_weight("agent2", 1)  # Normal weight

    print("Agent weights: agent1=2, agent2=1")

    # Enqueue low-priority intent first
    low_priority = create_intent("agent1", "req_low", "Low priority message")
    scheduler.enqueue(low_priority, priority=2)  # Low priority
    print("\nEnqueued low-priority intent from agent1")

    # Wait for aging
    time.sleep(0.2)

    # Enqueue high-priority intents
    for i in range(3):
        intent = create_intent("agent2", f"req_high_{i}", f"High priority {i}")
        scheduler.enqueue(intent, priority=0)  # High priority
    print("Enqueued 3 high-priority intents from agent2")

    # Dequeue and show order
    print("\nProcessing order:")
    processed = []
    for i in range(4):
        intent = scheduler.dequeue()
        if intent:
            processed.append(intent["req_id"])
            print(f"  {i + 1}. {intent['req_id']} (agent: {intent['agent_id']})")

    # Show that low-priority was processed (not starved)
    if "req_low" in processed:
        print("\n✓ Low-priority intent was processed (aging prevented starvation)")

    # Show statistics
    stats = scheduler.get_stats()
    print("\nScheduler Statistics:")
    print(f"  Total enqueued: {stats['total_enqueued']}")
    print(f"  Total dequeued: {stats['total_dequeued']}")
    print(f"  Agent queue depths: {stats['agent_queue_depths']}")
    print(f"  Aging stats: {stats['priority_aging']}")


async def demo_integrated_system():
    """Demonstrate integrated quota + scheduling system."""
    print("\n=== Integrated System Demo ===\n")

    # Setup components
    quota_policy = QuotaPolicy(
        intents_per_minute=20,
        burst_size=5,
    )
    quota_controller = QuotaController(quota_policy)

    aging_policy = AgingPolicy(
        aging_rate=1.0,
        max_priority_boost=3,
    )
    scheduler = WeightedRoundRobinScheduler(
        aging_policy=aging_policy,
        max_queue_depth=10,
    )

    print("Processing intents with quota enforcement and fair scheduling...")

    # Simulate multiple agents submitting intents
    agents = ["agent1", "agent2", "agent3"]
    intent_count = 0

    for agent_id in agents:
        for i in range(4):
            try:
                # Check quota
                await quota_controller.check_intent_quota(agent_id)

                # Create and enqueue intent
                intent = create_intent(agent_id, f"{agent_id}_req{i}", f"Message {i}")
                priority = i % 3  # Vary priorities
                scheduler.enqueue(intent, priority=priority)

                intent_count += 1
                print(f"  ✓ {agent_id}: Intent {i} enqueued (priority={priority})")

            except QuotaExceededError:
                print(f"  ✗ {agent_id}: Intent {i} blocked by quota")

    # Process intents
    print(f"\nProcessing {intent_count} intents...")
    processed_count = 0

    while processed_count < intent_count:
        intent = scheduler.dequeue()
        if intent:
            processed_count += 1
            # Simulate processing
            await asyncio.sleep(0.01)

    print(f"✓ Processed all {processed_count} intents")

    # Show final statistics
    quota_stats = quota_controller.get_stats()
    scheduler_stats = scheduler.get_stats()

    print("\nFinal Statistics:")
    print(f"  Quota - Total intents: {quota_stats['total_intents']}")
    print(f"  Quota - Exceeded count: {quota_stats['quota_exceeded_count']}")
    print(f"  Scheduler - Enqueued: {scheduler_stats['total_enqueued']}")
    print(f"  Scheduler - Dequeued: {scheduler_stats['total_dequeued']}")


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Quota Management and Priority Fairness Demonstration")
    print("=" * 60)

    # Run demos
    await demo_quota_controller()
    demo_priority_aging()
    demo_scheduler_with_aging()
    await demo_integrated_system()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
