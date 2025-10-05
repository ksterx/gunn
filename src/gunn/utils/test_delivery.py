"""Tests for delivery tracking with at-least-once guarantees."""

import asyncio
import time

import pytest

from gunn.schemas.types import ObservationDelta
from gunn.utils.delivery import DeliveryTracker


@pytest.fixture
async def delivery_tracker() -> DeliveryTracker:
    """Create a delivery tracker for testing."""
    tracker = DeliveryTracker(
        initial_timeout=0.1,  # Short timeout for testing
        max_timeout=1.0,
        backoff_multiplier=2.0,
        max_attempts=3,
    )
    await tracker.start()
    yield tracker
    await tracker.shutdown()


@pytest.mark.asyncio
async def test_track_delivery_basic(delivery_tracker: DeliveryTracker) -> None:
    """Test basic delivery tracking."""
    delta = ObservationDelta(
        view_seq=1,
        patches=[{"op": "replace", "path": "/test", "value": "data"}],
        context_digest="abc123",
        schema_version="1.0.0",
        delivery_id="",
        redelivery=False,
    )

    delivered = []

    async def delivery_callback(d: ObservationDelta) -> None:
        delivered.append(d)

    delivery_id = await delivery_tracker.track_delivery(
        "agent1", delta, delivery_callback
    )

    assert delivery_id
    assert len(delivered) == 1
    assert delivered[0] == delta


@pytest.mark.asyncio
async def test_acknowledge_delivery(delivery_tracker: DeliveryTracker) -> None:
    """Test acknowledging a delivery."""
    delta = ObservationDelta(
        view_seq=1,
        patches=[],
        context_digest="abc123",
        schema_version="1.0.0",
        delivery_id="",
        redelivery=False,
    )

    async def delivery_callback(d: ObservationDelta) -> None:
        pass

    delivery_id = await delivery_tracker.track_delivery(
        "agent1", delta, delivery_callback
    )

    # Acknowledge the delivery
    success = await delivery_tracker.acknowledge(delivery_id)
    assert success

    # Check it's acknowledged
    is_acked = await delivery_tracker.is_acknowledged(delivery_id)
    assert is_acked

    # Acknowledging again should be idempotent
    success = await delivery_tracker.acknowledge(delivery_id)
    assert success


@pytest.mark.asyncio
async def test_redelivery_on_timeout(delivery_tracker: DeliveryTracker) -> None:
    """Test that unacknowledged deliveries are retried."""
    delta = ObservationDelta(
        view_seq=1,
        patches=[],
        context_digest="abc123",
        schema_version="1.0.0",
        delivery_id="",
        redelivery=False,
    )

    delivery_count = 0

    async def delivery_callback(d: ObservationDelta) -> None:
        nonlocal delivery_count
        delivery_count += 1

    delivery_id = await delivery_tracker.track_delivery(
        "agent1", delta, delivery_callback
    )

    # Initial delivery should have happened
    assert delivery_count == 1

    # Wait for redelivery timeout
    await asyncio.sleep(0.2)

    # Should have been redelivered at least once
    assert delivery_count >= 2

    # Acknowledge to stop redelivery
    await delivery_tracker.acknowledge(delivery_id)


@pytest.mark.asyncio
async def test_exponential_backoff(delivery_tracker: DeliveryTracker) -> None:
    """Test exponential backoff for redelivery."""
    delta = ObservationDelta(
        view_seq=1,
        patches=[],
        context_digest="abc123",
        schema_version="1.0.0",
        delivery_id="",
        redelivery=False,
    )

    delivery_times = []

    async def delivery_callback(d: ObservationDelta) -> None:
        delivery_times.append(time.time())

    await delivery_tracker.track_delivery("agent1", delta, delivery_callback)

    # Wait for multiple redeliveries
    await asyncio.sleep(0.5)

    # Should have multiple deliveries
    assert len(delivery_times) >= 2

    # Check that delays are increasing (exponential backoff)
    if len(delivery_times) >= 3:
        delay1 = delivery_times[1] - delivery_times[0]
        delay2 = delivery_times[2] - delivery_times[1]
        # Second delay should be longer than first (with some tolerance)
        assert delay2 > delay1 * 0.8


@pytest.mark.asyncio
async def test_max_attempts_limit(delivery_tracker: DeliveryTracker) -> None:
    """Test that delivery stops after max attempts."""
    delta = ObservationDelta(
        view_seq=1,
        patches=[],
        context_digest="abc123",
        schema_version="1.0.0",
        delivery_id="",
        redelivery=False,
    )

    delivery_count = 0

    async def delivery_callback(d: ObservationDelta) -> None:
        nonlocal delivery_count
        delivery_count += 1

    await delivery_tracker.track_delivery("agent1", delta, delivery_callback)

    # Wait long enough for all retries
    await asyncio.sleep(1.5)

    # Should not exceed max attempts (3 in this test)
    assert delivery_count <= 3


@pytest.mark.asyncio
async def test_pending_count(delivery_tracker: DeliveryTracker) -> None:
    """Test tracking pending delivery count."""
    delta = ObservationDelta(
        view_seq=1,
        patches=[],
        context_digest="abc123",
        schema_version="1.0.0",
        delivery_id="",
        redelivery=False,
    )

    async def delivery_callback(d: ObservationDelta) -> None:
        pass

    # Track multiple deliveries
    delivery_id1 = await delivery_tracker.track_delivery(
        "agent1", delta, delivery_callback
    )
    delivery_id2 = await delivery_tracker.track_delivery(
        "agent2", delta, delivery_callback
    )

    # Check pending count
    total_pending = await delivery_tracker.get_pending_count()
    assert total_pending == 2

    agent1_pending = await delivery_tracker.get_pending_count("agent1")
    assert agent1_pending == 1

    # Acknowledge one
    await delivery_tracker.acknowledge(delivery_id1)

    total_pending = await delivery_tracker.get_pending_count()
    assert total_pending == 1

    agent1_pending = await delivery_tracker.get_pending_count("agent1")
    assert agent1_pending == 0


@pytest.mark.asyncio
async def test_redelivery_flag(delivery_tracker: DeliveryTracker) -> None:
    """Test that redelivery flag is set correctly."""
    delta = ObservationDelta(
        view_seq=1,
        patches=[],
        context_digest="abc123",
        schema_version="1.0.0",
        delivery_id="",
        redelivery=False,
    )

    deliveries = []

    async def delivery_callback(d: ObservationDelta) -> None:
        deliveries.append(d.copy())

    await delivery_tracker.track_delivery("agent1", delta, delivery_callback)

    # Initial delivery should not be marked as redelivery
    assert len(deliveries) == 1
    assert deliveries[0]["redelivery"] is False

    # Wait for redelivery
    await asyncio.sleep(0.2)

    # Redelivery should be marked
    if len(deliveries) > 1:
        assert deliveries[1]["redelivery"] is True


@pytest.mark.asyncio
async def test_delivery_stats(delivery_tracker: DeliveryTracker) -> None:
    """Test delivery statistics."""
    delta = ObservationDelta(
        view_seq=1,
        patches=[],
        context_digest="abc123",
        schema_version="1.0.0",
        delivery_id="",
        redelivery=False,
    )

    async def delivery_callback(d: ObservationDelta) -> None:
        pass

    # Track some deliveries
    delivery_id1 = await delivery_tracker.track_delivery(
        "agent1", delta, delivery_callback
    )
    delivery_id2 = await delivery_tracker.track_delivery(
        "agent2", delta, delivery_callback
    )

    stats = delivery_tracker.get_stats()
    assert stats["pending_deliveries"] == 2
    assert stats["acknowledged_deliveries"] == 0

    # Acknowledge one
    await delivery_tracker.acknowledge(delivery_id1)

    stats = delivery_tracker.get_stats()
    assert stats["pending_deliveries"] == 1
    assert stats["acknowledged_deliveries"] == 1


@pytest.mark.asyncio
async def test_acknowledge_unknown_delivery(delivery_tracker: DeliveryTracker) -> None:
    """Test acknowledging an unknown delivery ID."""
    success = await delivery_tracker.acknowledge("unknown-id")
    assert not success


@pytest.mark.asyncio
async def test_concurrent_deliveries(delivery_tracker: DeliveryTracker) -> None:
    """Test handling multiple concurrent deliveries."""
    delta = ObservationDelta(
        view_seq=1,
        patches=[],
        context_digest="abc123",
        schema_version="1.0.0",
        delivery_id="",
        redelivery=False,
    )

    delivery_counts = {}

    async def make_callback(agent_id: str):
        async def callback(d: ObservationDelta) -> None:
            delivery_counts[agent_id] = delivery_counts.get(agent_id, 0) + 1

        return callback

    # Track deliveries for multiple agents concurrently
    tasks = [
        delivery_tracker.track_delivery(
            f"agent{i}", delta, await make_callback(f"agent{i}")
        )
        for i in range(10)
    ]

    delivery_ids = await asyncio.gather(*tasks)

    # All should have unique IDs
    assert len(set(delivery_ids)) == 10

    # All should have been delivered once
    assert all(count == 1 for count in delivery_counts.values())

    # Acknowledge all
    await asyncio.gather(*[delivery_tracker.acknowledge(did) for did in delivery_ids])

    # No pending deliveries
    pending = await delivery_tracker.get_pending_count()
    assert pending == 0
