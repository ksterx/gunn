"""Priority aging system to prevent starvation in scheduling.

This module implements priority aging where intents gain priority
over time to prevent low-priority intents from being starved by
high-priority ones.
"""

import time
from dataclasses import dataclass
from typing import Any

from gunn.schemas.types import Intent
from gunn.utils.telemetry import get_logger


@dataclass
class AgingPolicy:
    """Configuration for priority aging.

    Attributes:
        enabled: Whether aging is enabled
        aging_rate: Priority increase per second of waiting
        max_priority_boost: Maximum priority boost from aging
        min_wait_time_seconds: Minimum wait time before aging starts
    """

    enabled: bool = True
    aging_rate: float = 0.1  # Priority increase per second
    max_priority_boost: int = 5  # Maximum boost
    min_wait_time_seconds: float = 1.0  # Start aging after 1 second


@dataclass
class IntentMetadata:
    """Metadata for tracking intent aging.

    Attributes:
        intent: The original intent
        original_priority: Original priority value
        enqueue_time: When intent was enqueued
        aged_priority: Current priority after aging
    """

    intent: Intent
    original_priority: int
    enqueue_time: float
    aged_priority: int


class PriorityAging:
    """Manages priority aging to prevent starvation.

    Tracks intent wait times and increases priority over time
    to ensure that even low-priority intents eventually get processed.

    Requirements addressed:
    - 18.2: PriorityAging system to prevent starvation
    - 18.3: Weighted round-robin with aging-adjusted priorities
    """

    def __init__(self, policy: AgingPolicy | None = None):
        """Initialize priority aging system.

        Args:
            policy: Aging policy configuration
        """
        self.policy = policy or AgingPolicy()

        # Track intent metadata by req_id
        self._metadata: dict[str, IntentMetadata] = {}

        # Statistics
        self._total_aged: int = 0
        self._max_boost_applied: int = 0

        self._logger = get_logger("gunn.priority_aging")

    def track_intent(self, intent: Intent, priority: int) -> None:
        """Start tracking an intent for aging.

        Args:
            intent: Intent to track
            priority: Original priority value
        """
        req_id = intent["req_id"]

        if req_id in self._metadata:
            self._logger.warning(
                "Intent already tracked for aging",
                req_id=req_id,
                agent_id=intent["agent_id"],
            )
            return

        self._metadata[req_id] = IntentMetadata(
            intent=intent,
            original_priority=priority,
            enqueue_time=time.monotonic(),
            aged_priority=priority,
        )

        self._logger.debug(
            "Intent tracked for aging",
            req_id=req_id,
            agent_id=intent["agent_id"],
            priority=priority,
        )

    def get_aged_priority(self, req_id: str) -> int:
        """Get current aged priority for an intent.

        Args:
            req_id: Request ID of the intent

        Returns:
            Current priority after aging, or original if not tracked
        """
        if not self.policy.enabled:
            metadata = self._metadata.get(req_id)
            return metadata.original_priority if metadata else 0

        metadata = self._metadata.get(req_id)
        if not metadata:
            return 0

        # Calculate aging
        wait_time = time.monotonic() - metadata.enqueue_time

        if wait_time < self.policy.min_wait_time_seconds:
            return metadata.original_priority

        # Calculate priority boost
        effective_wait = wait_time - self.policy.min_wait_time_seconds
        boost = int(effective_wait * self.policy.aging_rate)
        boost = min(boost, self.policy.max_priority_boost)

        # Higher priority number = higher priority
        aged_priority = metadata.original_priority + boost

        # Update metadata
        if aged_priority != metadata.aged_priority:
            metadata.aged_priority = aged_priority

            if boost > 0:
                self._total_aged += 1
                self._max_boost_applied = max(self._max_boost_applied, boost)

                self._logger.debug(
                    "Priority aged",
                    req_id=req_id,
                    agent_id=metadata.intent["agent_id"],
                    original_priority=metadata.original_priority,
                    aged_priority=aged_priority,
                    boost=boost,
                    wait_time=wait_time,
                )

        return aged_priority

    def untrack_intent(self, req_id: str) -> None:
        """Stop tracking an intent (after processing).

        Args:
            req_id: Request ID of the intent
        """
        if req_id in self._metadata:
            metadata = self._metadata.pop(req_id)

            self._logger.debug(
                "Intent untracked",
                req_id=req_id,
                agent_id=metadata.intent["agent_id"],
                wait_time=time.monotonic() - metadata.enqueue_time,
                final_priority=metadata.aged_priority,
            )

    def get_wait_time(self, req_id: str) -> float:
        """Get how long an intent has been waiting.

        Args:
            req_id: Request ID of the intent

        Returns:
            Wait time in seconds, or 0 if not tracked
        """
        metadata = self._metadata.get(req_id)
        if not metadata:
            return 0.0

        return time.monotonic() - metadata.enqueue_time

    def get_tracked_intents(self) -> list[str]:
        """Get list of currently tracked intent IDs.

        Returns:
            List of req_ids being tracked
        """
        return list(self._metadata.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get aging statistics.

        Returns:
            Dictionary with aging statistics
        """
        # Calculate current wait times
        now = time.monotonic()
        wait_times = [now - meta.enqueue_time for meta in self._metadata.values()]

        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
        max_wait = max(wait_times) if wait_times else 0.0

        return {
            "policy": {
                "enabled": self.policy.enabled,
                "aging_rate": self.policy.aging_rate,
                "max_priority_boost": self.policy.max_priority_boost,
                "min_wait_time_seconds": self.policy.min_wait_time_seconds,
            },
            "tracked_intents": len(self._metadata),
            "total_aged": self._total_aged,
            "max_boost_applied": self._max_boost_applied,
            "avg_wait_time": avg_wait,
            "max_wait_time": max_wait,
        }

    def clear(self) -> None:
        """Clear all tracked intents."""
        count = len(self._metadata)
        self._metadata.clear()

        self._logger.info("Priority aging cleared", cleared_intents=count)
