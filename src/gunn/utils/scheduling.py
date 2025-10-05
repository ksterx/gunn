"""Scheduling utilities for fair intent processing.

This module provides scheduling algorithms for fair processing of intents
across multiple agents, including weighted round robin for fairness.
"""

from collections import defaultdict, deque
from typing import Any, Protocol

from gunn.schemas.types import Intent
from gunn.utils.priority_aging import AgingPolicy, PriorityAging
from gunn.utils.telemetry import get_logger


class IntentScheduler(Protocol):
    """Protocol for intent scheduling algorithms."""

    def enqueue(self, intent: Intent, priority: int = 0) -> None:
        """Enqueue an intent for processing.

        Args:
            intent: Intent to enqueue
            priority: Priority level (higher = more important)
        """
        ...

    def dequeue(self) -> Intent | None:
        """Dequeue the next intent for processing.

        Returns:
            Next intent to process, or None if queue is empty
        """
        ...

    def get_queue_depth(self, agent_id: str | None = None) -> int:
        """Get queue depth for specific agent or total.

        Args:
            agent_id: Agent ID to check, or None for total depth

        Returns:
            Queue depth
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler statistics
        """
        ...


class WeightedRoundRobinScheduler:
    """Weighted Round Robin scheduler with priority aging for fair intent processing.

    Provides fair scheduling across agents with configurable weights
    and priority levels. Ensures no agent can monopolize processing
    while respecting priority and fairness constraints. Integrates
    priority aging to prevent starvation of low-priority intents.

    Requirements addressed:
    - 3.4: Priority and fairness policies
    - 9: Fairness (Weighted Round Robin) in intent validation pipeline
    - 18.2: PriorityAging system to prevent starvation
    - 18.3: Weighted round-robin with aging-adjusted priorities
    """

    def __init__(
        self,
        default_weight: int = 1,
        max_queue_depth: int = 100,
        priority_levels: int = 3,
        aging_policy: AgingPolicy | None = None,
    ):
        """Initialize weighted round robin scheduler.

        Args:
            default_weight: Default weight for agents
            max_queue_depth: Maximum queue depth per agent
            priority_levels: Number of priority levels (0 = highest)
            aging_policy: Priority aging policy configuration
        """
        self.default_weight = default_weight
        self.max_queue_depth = max_queue_depth
        self.priority_levels = priority_levels

        # Per-agent queues organized by priority level
        # Structure: {agent_id: {priority: deque[Intent]}}
        self._queues: dict[str, dict[int, deque[Intent]]] = defaultdict(
            lambda: defaultdict(deque)
        )

        # Agent weights and round-robin state
        self._weights: dict[str, int] = {}
        self._current_weights: dict[str, int] = {}  # Current remaining weight
        self._agent_order: list[str] = []  # Round-robin order
        self._current_agent_index: int = 0

        # Priority aging
        self._priority_aging = PriorityAging(aging_policy)

        # Statistics
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._agent_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"enqueued": 0, "dequeued": 0, "dropped": 0}
        )

        self._logger = get_logger("gunn.scheduling.wrr")

    def set_agent_weight(self, agent_id: str, weight: int) -> None:
        """Set weight for specific agent.

        Args:
            agent_id: Agent identifier
            weight: Weight for this agent (higher = more processing time)

        Raises:
            ValueError: If weight is not positive
        """
        if weight <= 0:
            raise ValueError("Agent weight must be positive")

        old_weight = self._weights.get(agent_id, self.default_weight)
        self._weights[agent_id] = weight

        # Update current weight if agent exists
        if agent_id in self._current_weights:
            self._current_weights[agent_id] = weight

        # Add to agent order if new
        if agent_id not in self._agent_order:
            self._agent_order.append(agent_id)

        self._logger.info(
            "Agent weight updated",
            agent_id=agent_id,
            old_weight=old_weight,
            new_weight=weight,
        )

    def enqueue(self, intent: Intent, priority: int = 0) -> None:
        """Enqueue an intent for processing with priority aging tracking.

        Args:
            intent: Intent to enqueue
            priority: Priority level (0 = highest priority)

        Raises:
            ValueError: If priority is invalid or queue is full
        """
        if not 0 <= priority < self.priority_levels:
            raise ValueError(
                f"Priority must be between 0 and {self.priority_levels - 1}"
            )

        agent_id = intent["agent_id"]

        # Check queue depth limit
        current_depth = self.get_queue_depth(agent_id)
        if current_depth >= self.max_queue_depth:
            self._agent_stats[agent_id]["dropped"] += 1
            self._logger.warning(
                "Intent dropped due to queue depth limit",
                agent_id=agent_id,
                req_id=intent["req_id"],
                current_depth=current_depth,
                max_depth=self.max_queue_depth,
            )
            raise ValueError(f"Queue full for agent {agent_id}")

        # Initialize agent if new
        if agent_id not in self._weights:
            self.set_agent_weight(agent_id, self.default_weight)

        # Track for priority aging
        self._priority_aging.track_intent(intent, priority)

        # Enqueue intent
        self._queues[agent_id][priority].append(intent)
        self._total_enqueued += 1
        self._agent_stats[agent_id]["enqueued"] += 1

        self._logger.debug(
            "Intent enqueued",
            agent_id=agent_id,
            req_id=intent["req_id"],
            priority=priority,
            queue_depth=current_depth + 1,
        )

    def dequeue(self) -> Intent | None:
        """Dequeue the next intent using weighted round robin.

        Returns:
            Next intent to process, or None if all queues are empty
        """
        if not self._agent_order:
            return None

        # Try to find an intent using weighted round robin
        attempts = 0
        max_attempts = len(self._agent_order) * 2  # Avoid infinite loops

        while attempts < max_attempts:
            attempts += 1

            # Get current agent
            if self._current_agent_index >= len(self._agent_order):
                self._current_agent_index = 0
                self._reset_weights()

            agent_id = self._agent_order[self._current_agent_index]

            # Check if agent has remaining weight
            current_weight = self._current_weights.get(agent_id, 0)
            if current_weight <= 0:
                self._current_agent_index += 1
                continue

            # Try to dequeue from this agent (highest priority first)
            intent = self._dequeue_from_agent(agent_id)
            if intent:
                # Consume weight
                self._current_weights[agent_id] -= 1
                self._total_dequeued += 1
                self._agent_stats[agent_id]["dequeued"] += 1

                self._logger.debug(
                    "Intent dequeued",
                    agent_id=agent_id,
                    req_id=intent["req_id"],
                    remaining_weight=self._current_weights[agent_id],
                )

                return intent

            # No intent from this agent, move to next
            self._current_agent_index += 1

        return None

    def _dequeue_from_agent(self, agent_id: str) -> Intent | None:
        """Dequeue highest aged priority intent from specific agent.

        Uses priority aging to determine effective priority, preventing
        starvation of lower-priority intents. Checks all intents across
        all priority queues to find the one with highest aged priority.

        Priority comparison:
        1. Aged priority (higher is better)
        2. Queue priority (lower number = higher priority)
        3. FIFO within same aged priority

        Args:
            agent_id: Agent to dequeue from

        Returns:
            Intent or None if agent has no intents
        """
        agent_queues = self._queues[agent_id]

        # Find intent with highest aged priority across all queues
        best_intent: Intent | None = None
        best_aged_priority: int = -1
        best_queue_priority: int = self.priority_levels  # Start with lowest priority
        best_queue_index: int = -1

        for queue_priority in range(self.priority_levels):
            queue = agent_queues.get(queue_priority)
            if not queue:
                continue

            # Check all intents in this queue to find highest aged priority
            for idx, intent in enumerate(queue):
                aged_priority = self._priority_aging.get_aged_priority(intent["req_id"])

                # Compare: aged priority first, then queue priority (lower is better)
                is_better = False
                if aged_priority > best_aged_priority:
                    is_better = True
                elif aged_priority == best_aged_priority:
                    # Same aged priority - use queue priority (lower is better)
                    if queue_priority < best_queue_priority:
                        is_better = True
                    elif queue_priority == best_queue_priority and best_intent is None:
                        # Same queue, first one wins (FIFO)
                        is_better = True

                if is_better:
                    best_intent = intent
                    best_aged_priority = aged_priority
                    best_queue_priority = queue_priority
                    best_queue_index = idx

        # Dequeue the best intent
        if best_intent:
            # Remove from queue at specific index
            queue = agent_queues[best_queue_priority]
            queue.remove(best_intent)
            self._priority_aging.untrack_intent(best_intent["req_id"])
            return best_intent

        return None

    def _reset_weights(self) -> None:
        """Reset current weights for all agents."""
        for agent_id in self._agent_order:
            weight = self._weights.get(agent_id, self.default_weight)
            self._current_weights[agent_id] = weight

        # self._logger.debug("Agent weights reset for new round")

    def get_queue_depth(self, agent_id: str | None = None) -> int:
        """Get queue depth for specific agent or total.

        Args:
            agent_id: Agent ID to check, or None for total depth

        Returns:
            Queue depth
        """
        if agent_id:
            agent_queues = self._queues[agent_id]
            return sum(len(queue) for queue in agent_queues.values())
        else:
            total = 0
            for agent_queues in self._queues.values():
                total += sum(len(queue) for queue in agent_queues.values())
            return total

    def get_agent_queue_depths(self) -> dict[str, int]:
        """Get queue depths for all agents.

        Returns:
            Dictionary mapping agent_id to queue depth
        """
        return {
            agent_id: self.get_queue_depth(agent_id) for agent_id in self._agent_order
        }

    def get_priority_distribution(self, agent_id: str) -> dict[int, int]:
        """Get distribution of intents by priority for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary mapping priority to count
        """
        agent_queues = self._queues[agent_id]
        return {
            priority: len(queue) for priority, queue in agent_queues.items() if queue
        }

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics including aging information.

        Returns:
            Dictionary with comprehensive scheduler statistics
        """
        total_depth = self.get_queue_depth()
        agent_depths = self.get_agent_queue_depths()

        return {
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_queue_depth": total_depth,
            "agent_count": len(self._agent_order),
            "agent_queue_depths": agent_depths,
            "agent_weights": dict(self._weights),
            "current_weights": dict(self._current_weights),
            "current_agent_index": self._current_agent_index,
            "agent_stats": dict(self._agent_stats),
            "max_queue_depth": self.max_queue_depth,
            "priority_levels": self.priority_levels,
            "priority_aging": self._priority_aging.get_stats(),
        }

    def remove_agent(self, agent_id: str) -> int:
        """Remove agent and return number of dropped intents.

        Args:
            agent_id: Agent to remove

        Returns:
            Number of intents that were dropped
        """
        if agent_id not in self._agent_order:
            return 0

        # Count intents to be dropped
        agent_queues = self._queues[agent_id]
        dropped_count = sum(len(queue) for queue in agent_queues.values())

        # Remove agent
        del self._queues[agent_id]
        if agent_id in self._weights:
            del self._weights[agent_id]
        if agent_id in self._current_weights:
            del self._current_weights[agent_id]

        self._agent_order.remove(agent_id)

        # Adjust current agent index if needed
        if self._current_agent_index >= len(self._agent_order):
            self._current_agent_index = 0

        self._agent_stats[agent_id]["dropped"] += dropped_count

        self._logger.info(
            "Agent removed from scheduler",
            agent_id=agent_id,
            dropped_intents=dropped_count,
        )

        return dropped_count

    def clear(self) -> int:
        """Clear all queues and return number of dropped intents.

        Returns:
            Total number of intents that were dropped
        """
        total_dropped = self.get_queue_depth()

        self._queues.clear()
        self._current_weights.clear()
        self._agent_order.clear()
        self._current_agent_index = 0
        self._priority_aging.clear()

        self._logger.info(
            "Scheduler cleared",
            dropped_intents=total_dropped,
        )

        return total_dropped


class PriorityScheduler:
    """Simple priority-based scheduler without fairness.

    Processes intents strictly by priority level, which can lead to
    starvation of lower-priority intents but ensures highest priority
    intents are always processed first.
    """

    def __init__(self, priority_levels: int = 3, max_queue_depth: int = 1000):
        """Initialize priority scheduler.

        Args:
            priority_levels: Number of priority levels (0 = highest)
            max_queue_depth: Maximum total queue depth
        """
        self.priority_levels = priority_levels
        self.max_queue_depth = max_queue_depth

        # Priority queues: {priority: deque[Intent]}
        self._queues: dict[int, deque[Intent]] = {
            priority: deque() for priority in range(priority_levels)
        }

        self._total_enqueued = 0
        self._total_dequeued = 0
        self._logger = get_logger("gunn.scheduling.priority")

    def enqueue(self, intent: Intent, priority: int = 0) -> None:
        """Enqueue intent by priority.

        Args:
            intent: Intent to enqueue
            priority: Priority level (0 = highest)

        Raises:
            ValueError: If priority is invalid or queue is full
        """
        if not 0 <= priority < self.priority_levels:
            raise ValueError(
                f"Priority must be between 0 and {self.priority_levels - 1}"
            )

        if self.get_queue_depth() >= self.max_queue_depth:
            raise ValueError("Queue is full")

        self._queues[priority].append(intent)
        self._total_enqueued += 1

    def dequeue(self) -> Intent | None:
        """Dequeue highest priority intent.

        Returns:
            Next intent to process, or None if all queues are empty
        """
        # Process from highest priority (0) to lowest
        for priority in range(self.priority_levels):
            if self._queues[priority]:
                intent = self._queues[priority].popleft()
                self._total_dequeued += 1
                return intent

        return None

    def get_queue_depth(self, agent_id: str | None = None) -> int:
        """Get total queue depth.

        Args:
            agent_id: Ignored for priority scheduler

        Returns:
            Total queue depth
        """
        return sum(len(queue) for queue in self._queues.values())

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler statistics
        """
        priority_depths = {
            priority: len(queue) for priority, queue in self._queues.items()
        }

        return {
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_queue_depth": self.get_queue_depth(),
            "priority_depths": priority_depths,
            "priority_levels": self.priority_levels,
            "max_queue_depth": self.max_queue_depth,
        }
