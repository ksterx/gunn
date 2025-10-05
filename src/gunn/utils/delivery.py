"""Delivery tracking for at-least-once observation delivery guarantees."""

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gunn.schemas.types import ObservationDelta


@dataclass
class DeliveryAttempt:
    """Tracks a single delivery attempt."""

    delivery_id: str
    agent_id: str
    delta: ObservationDelta
    attempt_count: int
    last_attempt_time: float
    next_retry_time: float
    acknowledged: bool
    callback: Callable[[ObservationDelta], Any]


class DeliveryTracker:
    """Tracks observation deliveries with at-least-once guarantees.

    This class implements reliable delivery with:
    - Unique delivery IDs for idempotent handling
    - Configurable timeout and exponential backoff
    - Redelivery on timeout
    - Acknowledgment tracking
    """

    def __init__(
        self,
        initial_timeout: float = 5.0,
        max_timeout: float = 60.0,
        backoff_multiplier: float = 2.0,
        max_attempts: int = 5,
    ) -> None:
        """Initialize delivery tracker.

        Args:
            initial_timeout: Initial timeout in seconds before first retry
            max_timeout: Maximum timeout between retries
            backoff_multiplier: Multiplier for exponential backoff
            max_attempts: Maximum number of delivery attempts before giving up
        """
        self.initial_timeout = initial_timeout
        self.max_timeout = max_timeout
        self.backoff_multiplier = backoff_multiplier
        self.max_attempts = max_attempts

        # Track pending deliveries: delivery_id -> DeliveryAttempt
        self._pending: dict[str, DeliveryAttempt] = {}

        # Track acknowledged deliveries for deduplication
        self._acknowledged: set[str] = set()

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Redelivery task
        self._redelivery_task: asyncio.Task[None] | None = None
        self._shutdown = False

    async def start(self) -> None:
        """Start the redelivery background task."""
        if self._redelivery_task is None:
            self._redelivery_task = asyncio.create_task(self._redelivery_loop())

    async def shutdown(self) -> None:
        """Shutdown the delivery tracker."""
        self._shutdown = True
        if self._redelivery_task:
            self._redelivery_task.cancel()
            try:
                await self._redelivery_task
            except asyncio.CancelledError:
                pass

    async def track_delivery(
        self,
        agent_id: str,
        delta: ObservationDelta,
        delivery_callback: Callable[[ObservationDelta], Any],
    ) -> str:
        """Track a new delivery attempt.

        Args:
            agent_id: Agent receiving the observation
            delta: Observation delta to deliver
            delivery_callback: Async callback to invoke for delivery

        Returns:
            delivery_id: Unique identifier for this delivery
        """
        delivery_id = str(uuid.uuid4())
        now = time.time()

        # Create delivery attempt
        attempt = DeliveryAttempt(
            delivery_id=delivery_id,
            agent_id=agent_id,
            delta=delta,
            attempt_count=1,
            last_attempt_time=now,
            next_retry_time=now + self.initial_timeout,
            acknowledged=False,
            callback=delivery_callback,
        )

        async with self._lock:
            self._pending[delivery_id] = attempt

        # Perform initial delivery outside the lock
        try:
            await delivery_callback(delta)
        except Exception:
            # Delivery failed, will be retried by redelivery loop
            pass

        return delivery_id

    async def acknowledge(self, delivery_id: str) -> bool:
        """Acknowledge receipt of a delivery.

        Args:
            delivery_id: Unique delivery identifier

        Returns:
            True if acknowledgment was successful, False if delivery not found
        """
        async with self._lock:
            # Check if already acknowledged (idempotent)
            if delivery_id in self._acknowledged:
                return True

            # Mark as acknowledged
            if delivery_id in self._pending:
                self._pending[delivery_id].acknowledged = True
                self._acknowledged.add(delivery_id)
                del self._pending[delivery_id]
                return True

            return False

    async def is_acknowledged(self, delivery_id: str) -> bool:
        """Check if a delivery has been acknowledged.

        Args:
            delivery_id: Unique delivery identifier

        Returns:
            True if acknowledged, False otherwise
        """
        async with self._lock:
            return delivery_id in self._acknowledged

    async def get_pending_count(self, agent_id: str | None = None) -> int:
        """Get count of pending deliveries.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            Number of pending deliveries
        """
        async with self._lock:
            if agent_id is None:
                return len(self._pending)
            return sum(
                1 for attempt in self._pending.values() if attempt.agent_id == agent_id
            )

    async def _redelivery_loop(self) -> None:
        """Background loop for redelivering unacknowledged observations."""
        while not self._shutdown:
            try:
                await asyncio.sleep(0.05)  # Check frequently for responsive redelivery

                now = time.time()
                to_redeliver: list[DeliveryAttempt] = []

                async with self._lock:
                    for attempt in list(self._pending.values()):
                        # Skip if already acknowledged
                        if attempt.acknowledged:
                            continue

                        # Check if retry time has arrived
                        if now >= attempt.next_retry_time:
                            # Check if max attempts exceeded
                            if attempt.attempt_count >= self.max_attempts:
                                # Give up on this delivery
                                del self._pending[attempt.delivery_id]
                                continue

                            to_redeliver.append(attempt)

                # Perform redeliveries outside the lock
                for attempt in to_redeliver:
                    await self._redeliver(attempt)

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue
                pass

    async def _redeliver(self, attempt: DeliveryAttempt) -> None:
        """Redeliver an observation.

        Args:
            attempt: Delivery attempt to retry
        """
        # Update attempt tracking
        async with self._lock:
            # Double-check it's still pending
            if attempt.delivery_id not in self._pending:
                return

            # Update attempt tracking
            attempt.attempt_count += 1
            attempt.last_attempt_time = time.time()

            # Calculate next retry time with exponential backoff
            timeout = min(
                self.initial_timeout
                * (self.backoff_multiplier ** (attempt.attempt_count - 1)),
                self.max_timeout,
            )
            attempt.next_retry_time = attempt.last_attempt_time + timeout

            # Mark as redelivery
            attempt.delta["redelivery"] = True

        # Perform redelivery outside the lock
        try:
            await attempt.callback(attempt.delta)
        except Exception:
            # Delivery failed, will be retried again
            pass

    def get_stats(self) -> dict[str, Any]:
        """Get delivery statistics.

        Returns:
            Dictionary with delivery stats
        """
        return {
            "pending_deliveries": len(self._pending),
            "acknowledged_deliveries": len(self._acknowledged),
            "total_tracked": len(self._pending) + len(self._acknowledged),
        }
