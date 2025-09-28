"""
Unit tests for TimedQueue implementation.

Tests concurrent operations and timing accuracy with ±5ms tolerance as required.
"""

import asyncio
import time

import pytest

from gunn.utils.timing import TimedQueue


class TestTimedQueue:
    """Test suite for TimedQueue class."""

    @pytest.fixture
    def queue(self) -> TimedQueue:
        """Create a fresh TimedQueue for each test."""
        return TimedQueue()

    @pytest.mark.asyncio
    async def test_basic_put_and_get(self, queue: TimedQueue) -> None:
        """Test basic put_at and get operations."""
        loop = asyncio.get_running_loop()
        now = loop.time()

        # Schedule item for immediate delivery
        await queue.put_at(now, "test_item")

        # Should get the item immediately
        item = await queue.get()
        assert item == "test_item"

    @pytest.mark.asyncio
    async def test_put_in_convenience_method(self, queue: TimedQueue) -> None:
        """Test put_in convenience method."""
        # Schedule item for delivery in 10ms
        await queue.put_in(0.01, "delayed_item")

        start_time = time.time()
        item = await queue.get()
        end_time = time.time()

        assert item == "delayed_item"
        # Should take at least 10ms (with some tolerance for timing precision)
        assert (end_time - start_time) >= 0.005  # 5ms tolerance

    @pytest.mark.asyncio
    async def test_timed_delivery_accuracy(self, queue):
        """Test timing accuracy within ±5ms tolerance."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        delay = 0.05  # 50ms delay

        await queue.put_at(now + delay, "timed_item")

        start_time = loop.time()
        item = await queue.get()
        end_time = loop.time()

        assert item == "timed_item"
        actual_delay = end_time - start_time

        # Verify timing accuracy within ±5ms tolerance
        assert (
            abs(actual_delay - delay) <= 0.005
        ), f"Timing error: expected {delay}s, got {actual_delay}s"

    @pytest.mark.asyncio
    async def test_ordering_by_delivery_time(self, queue):
        """Test that items are delivered in order of delivery time."""
        loop = asyncio.get_running_loop()
        now = loop.time()

        # Add items in reverse order of delivery time
        await queue.put_at(now + 0.03, "third")
        await queue.put_at(now + 0.01, "first")
        await queue.put_at(now + 0.02, "second")

        # Should get items in delivery time order
        assert await queue.get() == "first"
        assert await queue.get() == "second"
        assert await queue.get() == "third"

    @pytest.mark.asyncio
    async def test_ordering_by_sequence_for_same_time(self, queue):
        """Test that items with same delivery time are ordered by sequence."""
        loop = asyncio.get_running_loop()
        now = loop.time()

        # Add multiple items for same delivery time
        await queue.put_at(now, "first")
        await queue.put_at(now, "second")
        await queue.put_at(now, "third")

        # Should get items in insertion order (sequence order)
        assert await queue.get() == "first"
        assert await queue.get() == "second"
        assert await queue.get() == "third"

    @pytest.mark.asyncio
    async def test_concurrent_put_operations(self, queue):
        """Test concurrent put operations with proper locking."""
        loop = asyncio.get_running_loop()
        now = loop.time()

        async def put_items(start_index, count):
            """Put multiple items concurrently."""
            for i in range(count):
                await queue.put_at(now + 0.01, f"item_{start_index + i}")

        # Start multiple concurrent put operations
        await asyncio.gather(put_items(0, 10), put_items(10, 10), put_items(20, 10))  # type: ignore

        # Should have all 30 items
        assert queue.qsize() == 30

        # All items should be retrievable
        items = []
        for _ in range(30):
            items.append(await queue.get())

        assert len(items) == 30
        assert len(set(items)) == 30  # All items should be unique

    @pytest.mark.asyncio
    async def test_concurrent_get_operations(self, queue):
        """Test concurrent get operations."""
        loop = asyncio.get_running_loop()
        now = loop.time()

        # Add items for immediate delivery
        for i in range(10):
            await queue.put_at(now, f"item_{i}")

        async def get_item():
            """Get a single item."""
            return await queue.get()

        # Start multiple concurrent get operations
        results = await asyncio.gather(*[get_item() for _ in range(10)])  # type: ignore

        assert len(results) == 10
        assert len(set(results)) == 10  # All results should be unique

    @pytest.mark.asyncio
    async def test_get_blocks_until_item_ready(self, queue):
        """Test that get() blocks until item is ready for delivery."""
        loop = asyncio.get_running_loop()

        async def delayed_put():
            """Put item after a delay."""
            await asyncio.sleep(0.02)  # 20ms delay
            await queue.put_at(loop.time(), "delayed_item")

        # Start delayed put in background
        put_task = asyncio.create_task(delayed_put())  # type: ignore

        # get() should block until item is available
        start_time = time.time()
        item = await queue.get()
        end_time = time.time()

        assert item == "delayed_item"
        assert (end_time - start_time) >= 0.015  # Should have waited at least 15ms

        await put_task  # Clean up

    @pytest.mark.asyncio
    async def test_get_nowait_success(self, queue):
        """Test get_nowait when item is ready."""
        loop = asyncio.get_running_loop()
        now = loop.time()

        await queue.put_at(now, "ready_item")

        # Should get item immediately without blocking
        item = queue.get_nowait()
        assert item == "ready_item"

    @pytest.mark.asyncio
    async def test_get_nowait_empty_queue(self, queue):
        """Test get_nowait raises QueueEmpty when no items."""
        with pytest.raises(asyncio.QueueEmpty):
            queue.get_nowait()

    @pytest.mark.asyncio
    async def test_get_nowait_item_not_ready(self, queue):
        """Test get_nowait raises QueueEmpty when item not ready."""
        loop = asyncio.get_running_loop()
        future_time = loop.time() + 1.0  # 1 second in future

        await queue.put_at(future_time, "future_item")

        with pytest.raises(asyncio.QueueEmpty):
            queue.get_nowait()

    @pytest.mark.asyncio
    async def test_queue_size_and_empty(self, queue):
        """Test qsize() and empty() methods."""
        assert queue.empty()
        assert queue.qsize() == 0

        loop = asyncio.get_running_loop()
        now = loop.time()

        await queue.put_at(now, "item1")
        assert not queue.empty()
        assert queue.qsize() == 1

        await queue.put_at(now, "item2")
        assert queue.qsize() == 2

        await queue.get()
        assert queue.qsize() == 1

        await queue.get()
        assert queue.empty()
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_peek_next_delivery_time(self, queue):
        """Test peek_next_delivery_time method."""
        assert queue.peek_next_delivery_time() is None

        loop = asyncio.get_running_loop()
        delivery_time = loop.time() + 0.1

        await queue.put_at(delivery_time, "item")

        peeked_time = queue.peek_next_delivery_time()
        assert peeked_time == delivery_time

        # Peek should not remove the item
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_close_queue(self, queue):
        """Test queue closing behavior."""
        loop = asyncio.get_running_loop()
        now = loop.time()

        # Add an item before closing
        await queue.put_at(now, "existing_item")

        await queue.close()

        # Should not be able to add new items
        with pytest.raises(RuntimeError, match="Cannot put items into closed queue"):
            await queue.put_at(now, "new_item")

        # Should still be able to get existing items
        item = await queue.get()
        assert item == "existing_item"

        # get() on empty closed queue should raise RuntimeError
        with pytest.raises(RuntimeError, match="Queue is closed and empty"):
            await queue.get()

    @pytest.mark.asyncio
    async def test_close_queue_get_nowait(self, queue):
        """Test get_nowait on closed empty queue."""
        await queue.close()

        with pytest.raises(RuntimeError, match="Queue is closed and empty"):
            queue.get_nowait()

    @pytest.mark.asyncio
    async def test_multiple_waiters_notification(self, queue):
        """Test that multiple get() operations are properly notified."""

        async def waiter(waiter_id):
            """Wait for an item and return it with waiter ID."""
            item = await queue.get()
            return f"waiter_{waiter_id}_{item}"

        # Start multiple waiters
        waiter_tasks = [asyncio.create_task(waiter(i)) for i in range(3)]  # type: ignore

        # Give waiters time to start waiting
        await asyncio.sleep(0.01)

        # Add items
        loop = asyncio.get_running_loop()
        now = loop.time()
        for i in range(3):
            await queue.put_at(now, f"item_{i}")

        # All waiters should get their items
        results = await asyncio.gather(*waiter_tasks)
        assert len(results) == 3
        assert all("waiter_" in result for result in results)

    @pytest.mark.asyncio
    async def test_timing_precision_under_load(self, queue):
        """Test timing precision under concurrent load."""
        loop = asyncio.get_running_loop()
        base_time = loop.time()

        # Schedule many items with precise timing
        expected_times = []
        for i in range(20):
            delivery_time = base_time + (i * 0.005)  # 5ms intervals
            expected_times.append(delivery_time)
            await queue.put_at(delivery_time, f"item_{i}")

        # Retrieve all items and measure timing
        actual_times = []
        for _ in range(20):
            start = loop.time()
            await queue.get()
            actual_times.append(start)

        # Check that timing is reasonably accurate
        for expected, actual in zip(expected_times, actual_times, strict=True):
            timing_error = abs(actual - expected)
            assert (
                timing_error <= 0.01
            ), f"Timing error too large: {timing_error}s"  # 10ms tolerance under load

    @pytest.mark.asyncio
    async def test_responsiveness_with_frequent_updates(self, queue):
        """Test that get() remains responsive with frequent put operations."""
        loop = asyncio.get_running_loop()

        async def frequent_putter():
            """Continuously add items."""
            for i in range(50):
                await queue.put_at(loop.time() + 0.001, f"frequent_{i}")
                await asyncio.sleep(0.001)  # 1ms between puts

        # Start frequent putting in background
        put_task = asyncio.create_task(frequent_putter())  # type: ignore

        # Get items and measure responsiveness
        start_time = loop.time()
        items_received = 0

        try:
            while items_received < 10:  # Get first 10 items
                await queue.get()
                items_received += 1
        except TimeoutError:
            pytest.fail("get() operation timed out - not responsive enough")

        end_time = loop.time()
        total_time = end_time - start_time

        # Should be able to get 10 items quickly despite frequent puts
        assert total_time < 0.1, f"Too slow: {total_time}s for 10 items"

        put_task.cancel()
        try:
            await put_task
        except asyncio.CancelledError:
            pass
