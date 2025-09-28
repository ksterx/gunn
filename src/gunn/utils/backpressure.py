"""Backpressure policies for queue management.

This module implements different backpressure policies to handle queue overflow
and resource exhaustion scenarios with configurable strategies.
"""

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Generic, TypeVar

from gunn.utils.errors import BackpressureError
from gunn.utils.telemetry import record_backpressure_event, record_queue_high_watermark

T = TypeVar("T")


class BackpressurePolicy(ABC, Generic[T]):
    """Abstract base class for backpressure policies.

    Defines the interface for handling queue overflow scenarios
    with different strategies for managing resource exhaustion.

    Requirements addressed:
    - 10.2: Backpressure policies (defer, shed oldest, drop newest)
    - 10.5: Queue depth monitoring and backpressure triggers
    """

    def __init__(self, threshold: int, agent_id: str = "unknown"):
        """Initialize backpressure policy.

        Args:
            threshold: Queue depth threshold that triggers backpressure
            agent_id: Agent identifier for metrics and logging
        """
        self.threshold = threshold
        self.agent_id = agent_id

    @abstractmethod
    async def handle_overflow(self, queue: deque[T], new_item: T) -> bool:
        """Handle queue overflow when threshold is exceeded.

        Args:
            queue: The queue that is overflowing
            new_item: New item trying to be added

        Returns:
            True if item was handled (added or dropped), False if should defer

        Raises:
            BackpressureError: If backpressure should be applied
        """
        pass

    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Get the policy name for metrics and logging."""
        pass

    def check_threshold(self, current_depth: int, queue_type: str = "queue") -> None:
        """Check if queue depth exceeds threshold and record metrics.

        Args:
            current_depth: Current queue depth
            queue_type: Type of queue for metrics

        Raises:
            BackpressureError: If threshold is exceeded
        """
        if current_depth >= self.threshold:
            # Record high watermark metric
            record_queue_high_watermark(self.agent_id, queue_type, current_depth)

            # Record backpressure event
            record_backpressure_event(self.agent_id, queue_type, self.policy_name)

            raise BackpressureError(
                self.agent_id,
                queue_type,
                current_depth,
                self.threshold,
                self.policy_name,
            )


class DeferPolicy(BackpressurePolicy[T]):
    """Defer policy - block new items when threshold exceeded.

    This is the default policy that raises BackpressureError to defer
    processing until queue depth decreases below threshold.

    Requirements addressed:
    - 10.2: Backpressure policies with defer as default
    """

    @property
    def policy_name(self) -> str:
        return "defer"

    async def handle_overflow(self, queue: deque[T], new_item: T) -> bool:
        """Handle overflow by deferring (raising BackpressureError).

        Args:
            queue: The overflowing queue
            new_item: New item to be deferred

        Returns:
            False (never handles the item, always defers)

        Raises:
            BackpressureError: Always raised to defer the operation
        """
        self.check_threshold(len(queue), "queue")
        return False  # Never reached due to exception


class ShedOldestPolicy(BackpressurePolicy[T]):
    """Shed oldest policy - drop oldest items to make room for new ones.

    When queue exceeds threshold, removes oldest items to make room
    for new items, maintaining queue size at threshold.

    Requirements addressed:
    - 10.2: Backpressure policies with shed oldest option
    """

    @property
    def policy_name(self) -> str:
        return "shed_oldest"

    async def handle_overflow(self, queue: deque[T], new_item: T) -> bool:
        """Handle overflow by dropping oldest items.

        Args:
            queue: The overflowing queue
            new_item: New item to add

        Returns:
            True (item was handled by adding and shedding old items)
        """
        current_depth = len(queue)

        if current_depth >= self.threshold:
            # Record metrics before shedding
            record_queue_high_watermark(self.agent_id, "queue", current_depth)
            record_backpressure_event(self.agent_id, "queue", self.policy_name)

            # Shed oldest items to make room
            items_to_shed = current_depth - self.threshold + 1
            for _ in range(items_to_shed):
                if queue:
                    queue.popleft()  # Remove oldest

            # Add new item
            queue.append(new_item)
            return True

        # No overflow, add normally
        queue.append(new_item)
        return True


class DropNewestPolicy(BackpressurePolicy[T]):
    """Drop newest policy - drop new items when threshold exceeded.

    When queue exceeds threshold, drops the new item instead of
    adding it, preserving existing queued items.

    Requirements addressed:
    - 10.2: Backpressure policies with drop newest option
    """

    @property
    def policy_name(self) -> str:
        return "drop_newest"

    async def handle_overflow(self, queue: deque[T], new_item: T) -> bool:
        """Handle overflow by dropping the new item.

        Args:
            queue: The overflowing queue
            new_item: New item to potentially drop

        Returns:
            True (item was handled by dropping it)
        """
        current_depth = len(queue)

        if current_depth >= self.threshold:
            # Record metrics before dropping
            record_queue_high_watermark(self.agent_id, "queue", current_depth)
            record_backpressure_event(self.agent_id, "queue", self.policy_name)

            # Drop the new item (don't add it)
            return True

        # No overflow, add normally
        queue.append(new_item)
        return True


class BackpressureManager:
    """Manager for applying backpressure policies to queues.

    Provides a unified interface for managing different backpressure
    policies and applying them consistently across the system.

    Requirements addressed:
    - 10.2: Configurable backpressure policies per agent class
    - 10.5: Queue depth monitoring and backpressure triggers
    """

    def __init__(self) -> None:
        """Initialize backpressure manager."""
        self._policies: dict[str, type[BackpressurePolicy[Any]]] = {
            "defer": DeferPolicy,
            "shed_oldest": ShedOldestPolicy,
            "drop_newest": DropNewestPolicy,
        }

    def create_policy(
        self, policy_name: str, threshold: int, agent_id: str = "unknown"
    ) -> BackpressurePolicy[Any]:
        """Create a backpressure policy instance.

        Args:
            policy_name: Name of the policy (defer, shed_oldest, drop_newest)
            threshold: Queue depth threshold
            agent_id: Agent identifier for metrics

        Returns:
            BackpressurePolicy instance

        Raises:
            ValueError: If policy_name is not recognized
        """
        if policy_name not in self._policies:
            available = ", ".join(self._policies.keys())
            raise ValueError(
                f"Unknown backpressure policy '{policy_name}'. Available: {available}"
            )

        policy_class = self._policies[policy_name]
        return policy_class(threshold, agent_id)

    def register_policy(
        self, name: str, policy_class: type[BackpressurePolicy[Any]]
    ) -> None:
        """Register a custom backpressure policy.

        Args:
            name: Policy name
            policy_class: Policy class implementing BackpressurePolicy
        """
        self._policies[name] = policy_class

    @property
    def available_policies(self) -> list[str]:
        """Get list of available policy names."""
        return list(self._policies.keys())


# Global backpressure manager instance
backpressure_manager = BackpressureManager()


class BackpressureQueue(Generic[T]):
    """Queue with integrated backpressure policy support.

    A queue implementation that automatically applies backpressure
    policies when depth thresholds are exceeded.

    Requirements addressed:
    - 10.2: Backpressure policies integrated with queue operations
    - 10.5: Automatic backpressure triggers based on queue depth
    """

    def __init__(
        self, policy: BackpressurePolicy[T], maxsize: int = 0, queue_type: str = "queue"
    ):
        """Initialize backpressure queue.

        Args:
            policy: Backpressure policy to apply
            maxsize: Maximum queue size (0 for unlimited)
            queue_type: Queue type for metrics and logging
        """
        self.policy = policy
        self.maxsize = maxsize
        self.queue_type = queue_type
        self._queue: deque[T] = deque()
        self._lock = asyncio.Lock()

    async def put(self, item: T) -> None:
        """Put an item in the queue with backpressure handling.

        Args:
            item: Item to add to queue

        Raises:
            BackpressureError: If policy defers the operation
        """
        async with self._lock:
            current_size = len(self._queue)

            # Determine effective threshold (use maxsize if it's lower than policy threshold)
            effective_threshold = self.policy.threshold
            if self.maxsize > 0:
                effective_threshold = min(effective_threshold, self.maxsize)

            # Check if we need to handle overflow
            if current_size >= effective_threshold:
                # If maxsize is the limiting factor and we're using a shedding policy,
                # handle the shedding ourselves to respect maxsize
                if (
                    self.maxsize > 0
                    and effective_threshold == self.maxsize
                    and self.policy.policy_name in ["shed_oldest", "drop_newest"]
                ):
                    if self.policy.policy_name == "shed_oldest":
                        # Shed oldest items to make room
                        items_to_shed = current_size - self.maxsize + 1
                        for _ in range(items_to_shed):
                            if self._queue:
                                self._queue.popleft()
                        self._queue.append(item)
                    elif self.policy.policy_name == "drop_newest":
                        # Drop the new item if at maxsize
                        if current_size >= self.maxsize:
                            pass  # Drop the new item
                        else:
                            self._queue.append(item)
                else:
                    # Let the policy handle it normally
                    await self.policy.handle_overflow(self._queue, item)
            else:
                self._queue.append(item)

    async def get(self) -> T:
        """Get an item from the queue.

        Returns:
            Next item from queue

        Raises:
            asyncio.QueueEmpty: If queue is empty
        """
        async with self._lock:
            if not self._queue:
                raise asyncio.QueueEmpty()
            return self._queue.popleft()

    def qsize(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def full(self) -> bool:
        """Check if queue is full (based on maxsize)."""
        if self.maxsize <= 0:
            return False
        return len(self._queue) >= self.maxsize
