#!/usr/bin/env python3
"""
Demonstration of TimedQueue functionality.

This script shows how TimedQueue can be used for latency simulation
in multi-agent scenarios.
"""

import asyncio
import time
from typing import Any

from .timing import TimedQueue


async def demo_basic_usage() -> None:
    """Demonstrate basic TimedQueue usage."""
    print("=== Basic TimedQueue Demo ===")

    queue = TimedQueue()
    loop = asyncio.get_running_loop()

    # Schedule items with different delays
    base_time = loop.time()
    await queue.put_at(base_time + 0.1, "First item (100ms delay)")
    await queue.put_at(base_time + 0.05, "Second item (50ms delay)")
    await queue.put_at(base_time + 0.15, "Third item (150ms delay)")

    print(f"Scheduled 3 items at {time.strftime('%H:%M:%S')}")

    # Retrieve items (should come out in delivery time order)
    for _ in range(3):
        start = time.time()
        item = await queue.get()
        end = time.time()
        wait_time = (end - start) * 1000
        timestamp = time.strftime("%H:%M:%S")
        print(f"Received: '{item}' at {timestamp} (waited {wait_time:.1f}ms)")


async def demo_multi_agent_simulation() -> None:
    """Demonstrate multi-agent observation delivery simulation."""
    print("\n=== Multi-Agent Simulation Demo ===")

    # Create per-agent queues with different latency models
    agents = {
        "Alice": {"queue": TimedQueue(), "latency": 0.02},  # 20ms network latency
        "Bob": {"queue": TimedQueue(), "latency": 0.01},  # 10ms network latency
        "Charlie": {"queue": TimedQueue(), "latency": 0.03},  # 30ms network latency
    }

    loop = asyncio.get_running_loop()
    event_time = loop.time()

    # Simulate a world event that all agents should observe
    world_event = {
        "type": "PlayerMoved",
        "player": "Dave",
        "position": {"x": 100, "y": 200},
        "timestamp": event_time,
    }

    print(f"World event occurred: {world_event['type']}")

    # Schedule observation delivery to each agent based on their latency
    for agent_name, agent_data in agents.items():
        latency: float = agent_data["latency"]  # type: ignore
        queue: TimedQueue = agent_data["queue"]  # type: ignore
        delivery_time = event_time + latency
        await queue.put_at(
            delivery_time,
            {"agent": agent_name, "observation": world_event, "view_seq": 42},
        )

    # Simulate agents receiving observations concurrently
    async def agent_observer(name: str, agent_data: dict[str, Any]) -> Any:
        start_time = time.time()
        observation = await agent_data["queue"].get()
        end_time = time.time()

        actual_latency = (end_time - start_time) * 1000  # Convert to ms
        expected_latency = agent_data["latency"] * 1000

        print(
            f"{name} received observation after {actual_latency:.1f}ms "
            f"(expected {expected_latency:.1f}ms)"
        )

        return observation

    # Start all agents observing concurrently
    observers = [agent_observer(name, data) for name, data in agents.items()]

    observations = await asyncio.gather(*observers)

    print(f"All {len(observations)} agents received the observation")


async def demo_high_frequency_events() -> None:
    """Demonstrate high-frequency event processing."""
    print("\n=== High-Frequency Events Demo ===")

    queue = TimedQueue()
    loop = asyncio.get_running_loop()

    # Generate 50 events at 10ms intervals (100 events/sec rate)
    num_events = 50
    interval = 0.01  # 10ms
    base_time = loop.time()

    print(f"Scheduling {num_events} events at 100 events/sec...")

    for i in range(num_events):
        delivery_time = base_time + (i * interval)
        await queue.put_at(delivery_time, f"Event_{i:02d}")

    # Process events and measure throughput
    start_time = time.time()
    processed = 0

    async def event_processor() -> None:
        nonlocal processed
        while processed < num_events:
            await queue.get()
            processed += 1
            if processed % 10 == 0:  # Print every 10th event
                print(f"Processed {processed}/{num_events} events")

    await event_processor()

    end_time = time.time()
    total_time = end_time - start_time
    throughput = num_events / total_time

    print(f"Processed {num_events} events in {total_time:.3f}s")
    print(f"Throughput: {throughput:.1f} events/sec")


async def main() -> None:
    """Run all demonstrations."""
    await demo_basic_usage()
    await demo_multi_agent_simulation()
    await demo_high_frequency_events()
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
