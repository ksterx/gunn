"""
Integration tests for TimedQueue to verify real-world usage patterns.
"""

import asyncio
import time
from typing import Any

import pytest

from .timing import TimedQueue


@pytest.mark.asyncio
async def test_multi_agent_observation_delivery() -> None:
    """Test TimedQueue in a multi-agent observation delivery scenario."""
    # Simulate per-agent queues as would be used in the orchestrator
    agent_queues = {
        "agent_1": TimedQueue(),
        "agent_2": TimedQueue(),
        "agent_3": TimedQueue(),
    }

    loop = asyncio.get_running_loop()
    base_time = loop.time()

    # Simulate different latency models for each agent
    latencies = {
        "agent_1": 0.01,  # 10ms latency
        "agent_2": 0.02,  # 20ms latency
        "agent_3": 0.005,  # 5ms latency
    }

    # Schedule observations for all agents
    for agent_id, queue in agent_queues.items():
        delivery_time = base_time + latencies[agent_id]
        await queue.put_at(
            delivery_time,
            {
                "view_seq": 1,
                "patches": [
                    {
                        "op": "add",
                        "path": "/entities/player1",
                        "value": {"x": 10, "y": 20},
                    }
                ],
                "context_digest": "abc123",
            },
        )

    # Simulate agents receiving observations
    async def agent_receiver(agent_id: str, queue: TimedQueue) -> dict[str, Any]:
        """Simulate an agent receiving observations."""
        observation = await queue.get()
        end_time = loop.time()

        actual_latency = end_time - base_time
        expected_latency = latencies[agent_id]

        return {
            "agent_id": agent_id,
            "observation": observation,
            "actual_latency": actual_latency,
            "expected_latency": expected_latency,
            "timing_error": abs(actual_latency - expected_latency),
        }

    # Start all agents receiving concurrently
    receivers = [
        agent_receiver(agent_id, queue) for agent_id, queue in agent_queues.items()
    ]

    results = await asyncio.gather(*receivers)

    # Verify all agents received their observations
    assert len(results) == 3

    # Verify timing accuracy (within 5ms tolerance)
    for result in results:
        assert (
            result["timing_error"] <= 0.005
        ), f"Agent {result['agent_id']} timing error: {result['timing_error']}s"
        assert result["observation"]["view_seq"] == 1
        assert len(result["observation"]["patches"]) == 1


@pytest.mark.asyncio
async def test_high_throughput_scenario() -> None:
    """Test TimedQueue under high throughput conditions."""
    queue = TimedQueue()
    loop = asyncio.get_running_loop()

    # Simulate high-frequency event generation (100 events/sec per requirement 11.3)
    num_events = 100
    event_interval = 0.01  # 10ms between events
    base_time = loop.time()

    # Schedule many events rapidly
    for i in range(num_events):
        delivery_time = base_time + (i * event_interval)
        await queue.put_at(delivery_time, f"event_{i}")

    # Measure throughput
    start_time = time.time()
    received_events = []

    for _ in range(num_events):
        event = await queue.get()
        received_events.append(event)

    end_time = time.time()
    total_time = end_time - start_time
    throughput = num_events / total_time

    # Verify all events received in order
    assert len(received_events) == num_events
    for i, event in enumerate(received_events):
        assert event == f"event_{i}"

    # Verify throughput meets requirement (≥100 events/sec)
    assert (
        throughput >= 90
    ), f"Throughput too low: {throughput} events/sec"  # Allow some tolerance


@pytest.mark.asyncio
async def test_cancellation_responsiveness() -> None:
    """Test queue responsiveness for cancellation scenarios."""
    queue = TimedQueue()
    loop = asyncio.get_running_loop()

    # Schedule an item for future delivery
    future_time = loop.time() + 0.1  # 100ms in future
    await queue.put_at(future_time, "future_item")

    # Start a get operation that will wait
    async def waiting_getter() -> Any:
        return await queue.get()

    get_task = asyncio.create_task(waiting_getter())

    # Wait a bit to ensure getter is waiting
    await asyncio.sleep(0.01)

    # Schedule an immediate item (simulating interruption)
    await queue.put_at(loop.time(), "interrupt_item")

    # The getter should receive the interrupt item quickly
    start_time = loop.time()
    result = await get_task
    end_time = loop.time()

    response_time = end_time - start_time

    assert result == "interrupt_item"
    # Should respond within 20ms (requirement 6.4)
    assert response_time <= 0.02, f"Response time too slow: {response_time}s"


@pytest.mark.asyncio
async def test_memory_efficiency() -> None:
    """Test that TimedQueue doesn't leak memory under sustained load."""
    queue = TimedQueue()
    loop = asyncio.get_running_loop()

    # Add and remove many items to test memory management
    num_cycles = 1000
    base_time = loop.time()

    for cycle in range(num_cycles):
        # Add items
        for i in range(10):
            await queue.put_at(base_time, f"cycle_{cycle}_item_{i}")

        # Remove items
        for _ in range(10):
            await queue.get()

        # Queue should be empty after each cycle
        assert queue.empty()
        assert queue.qsize() == 0

    # Final verification
    assert queue.empty()
    assert queue.qsize() == 0
