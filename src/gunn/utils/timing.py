"""
TimedQueue implementation for latency simulation.

This module provides a priority queue for timed delivery with latency simulation,
supporting concurrent operations and precise timing control.
"""

import asyncio
import heapq
from dataclasses import dataclass
from typing import Any


@dataclass
class _QueueItem:
    """Internal queue item with delivery time and sequence for ordering."""

    deliver_at: float
    sequence: int
    item: Any

    def __lt__(self, other: "_QueueItem") -> bool:
        """Compare items for heap ordering: earlier time first, then sequence."""
        if self.deliver_at != other.deliver_at:
            return self.deliver_at < other.deliver_at
        return self.sequence < other.sequence


class TimedQueue:
    """
    Priority queue for timed delivery with latency simulation.

    Supports scheduling items for delivery at specific times with proper locking
    for put operations and lock-free sleep for get operations to avoid blocking.

    Requirements addressed:
    - 6.4: ObservationDelta delivery latency (core in-proc) ≤ 20ms
    - 6.5: Timed delivery using per-agent TimedQueues with latency models
    - 4.7: Non-blocking operations per agent
    """

    def __init__(self) -> None:
        """Initialize empty timed queue."""
        self._heap: list[_QueueItem] = []
        self._sequence_counter: int = 0
        self._put_lock = asyncio.Lock()
        self._get_condition = asyncio.Condition()
        self._closed = False

    async def put_at(self, deliver_at: float, item: Any) -> None:
        """
        Schedule item for delivery at specific time.

        Args:
            deliver_at: Absolute time (from asyncio.get_running_loop().time()) when item
                should be delivered
            item: Item to deliver

        Raises:
            RuntimeError: If queue is closed
        """
        if self._closed:
            raise RuntimeError("Cannot put items into closed queue")

        async with self._put_lock:
            self._sequence_counter += 1
            queue_item = _QueueItem(
                deliver_at=deliver_at, sequence=self._sequence_counter, item=item
            )
            heapq.heappush(self._heap, queue_item)

        # Notify waiting get() operations that a new item is available
        async with self._get_condition:
            self._get_condition.notify()

    async def put_in(self, delay_seconds: float, item: Any) -> None:
        """
        Schedule item for delivery after specified delay.

        Args:
            delay_seconds: Delay in seconds from now
            item: Item to deliver
        """
        loop = asyncio.get_running_loop()
        deliver_at = loop.time() + delay_seconds
        await self.put_at(deliver_at, item)

    async def get(self) -> Any:
        """
        Get next item when its delivery time arrives.

        Uses lock-free sleep to avoid blocking other operations.

        Returns:
            The next item when its delivery time arrives

        Raises:
            RuntimeError: If queue is closed and empty
        """
        loop = asyncio.get_running_loop()

        while True:
            # Check if we have items and get the next one if ready
            async with self._get_condition:
                if not self._heap:
                    if self._closed:
                        raise RuntimeError("Queue is closed and empty")
                    # Wait for new items to be added
                    await self._get_condition.wait()
                    continue

                # Peek at the next item without removing it yet
                next_item = self._heap[0]
                now = loop.time()
                delay = next_item.deliver_at - now

                if delay <= 0:
                    # Item is ready for delivery
                    heapq.heappop(self._heap)
                    return next_item.item

            # Sleep outside the lock to avoid blocking other operations
            # Use a small maximum sleep to ensure responsiveness
            sleep_time = min(delay, 0.01)  # Max 10ms sleep to maintain responsiveness
            await asyncio.sleep(sleep_time)

    def get_nowait(self) -> Any:
        """
        Get next item if it's ready for delivery now.

        Returns:
            The next item if ready, otherwise raises asyncio.QueueEmpty

        Raises:
            asyncio.QueueEmpty: If no items are ready for delivery
            RuntimeError: If queue is closed and empty
        """
        if self._closed and not self._heap:
            raise RuntimeError("Queue is closed and empty")

        if not self._heap:
            raise asyncio.QueueEmpty("No items in queue")

        loop = asyncio.get_running_loop()
        now = loop.time()

        next_item = self._heap[0]
        if next_item.deliver_at <= now:
            heapq.heappop(self._heap)
            return next_item.item
        else:
            raise asyncio.QueueEmpty("Next item not ready for delivery")

    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        return len(self._heap)

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._heap) == 0

    async def close(self) -> None:
        """
        Close the queue, preventing new items from being added.

        Existing get() operations will continue until the queue is empty.
        """
        self._closed = True
        async with self._get_condition:
            self._get_condition.notify_all()

    def peek_next_delivery_time(self) -> float | None:
        """
        Get the delivery time of the next item without removing it.

        Returns:
            Delivery time of next item, or None if queue is empty
        """
        if not self._heap:
            return None
        return self._heap[0].deliver_at
