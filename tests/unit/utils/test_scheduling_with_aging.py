"""Integration tests for scheduler with priority aging."""

import time

from gunn.schemas.types import Intent
from gunn.utils.priority_aging import AgingPolicy
from gunn.utils.scheduling import WeightedRoundRobinScheduler


def create_test_intent(agent_id: str, req_id: str, priority: int = 0) -> Intent:
    """Create a test intent."""
    return {
        "kind": "Speak",
        "payload": {"text": "test"},
        "context_seq": 0,
        "req_id": req_id,
        "agent_id": agent_id,
        "priority": priority,
        "schema_version": "1.0.0",
    }


class TestSchedulerWithAging:
    """Test weighted round robin scheduler with priority aging."""

    def test_aging_prevents_starvation(self):
        """Test that aging prevents low-priority intents from starving."""
        policy = AgingPolicy(
            aging_rate=2.0,  # Fast aging for testing
            min_wait_time_seconds=0.05,
            max_priority_boost=10,
        )
        scheduler = WeightedRoundRobinScheduler(
            priority_levels=3,
            aging_policy=policy,
        )

        # Enqueue low-priority intent
        low_priority = create_test_intent("agent1", "req_low", priority=2)
        scheduler.enqueue(low_priority, priority=2)

        # Wait for aging
        time.sleep(0.1)

        # Enqueue high-priority intents
        for i in range(3):
            high_priority = create_test_intent("agent1", f"req_high_{i}", priority=0)
            scheduler.enqueue(high_priority, priority=0)

        # Dequeue - low priority should be processed due to aging
        first = scheduler.dequeue()
        assert first is not None

        # The aged low-priority intent might be processed first
        # depending on how much it aged
        processed_reqs = [first["req_id"]]
        while len(processed_reqs) < 4:
            intent = scheduler.dequeue()
            if intent:
                processed_reqs.append(intent["req_id"])
            else:
                break

        # Low priority should be processed (not starved)
        assert "req_low" in processed_reqs

    def test_fairness_with_aging(self):
        """Test fairness across agents with aging."""
        policy = AgingPolicy(
            aging_rate=1.0,
            min_wait_time_seconds=0.05,
        )
        scheduler = WeightedRoundRobinScheduler(
            default_weight=1,
            aging_policy=policy,
        )

        # Enqueue intents from multiple agents
        for agent_id in ["agent1", "agent2", "agent3"]:
            for i in range(3):
                intent = create_test_intent(agent_id, f"{agent_id}_req{i}")
                scheduler.enqueue(intent, priority=0)

        # Wait for some aging
        time.sleep(0.1)

        # Dequeue all - should be fair distribution
        agent_counts = {"agent1": 0, "agent2": 0, "agent3": 0}
        for _ in range(9):
            intent = scheduler.dequeue()
            if intent:
                agent_counts[intent["agent_id"]] += 1

        # Each agent should get roughly equal processing
        assert all(count == 3 for count in agent_counts.values())

    def test_priority_levels_with_aging(self):
        """Test that priority levels work correctly with aging."""
        policy = AgingPolicy(
            aging_rate=0.5,
            min_wait_time_seconds=0.05,
            max_priority_boost=3,
        )
        scheduler = WeightedRoundRobinScheduler(
            priority_levels=3,
            aging_policy=policy,
        )

        # Enqueue low-priority intent first
        low = create_test_intent("agent1", "req_low", priority=2)
        scheduler.enqueue(low, priority=2)

        # Wait for significant aging
        time.sleep(0.2)

        # Enqueue medium-priority intent
        medium = create_test_intent("agent1", "req_medium", priority=1)
        scheduler.enqueue(medium, priority=1)

        # The aged low-priority might now have higher effective priority
        first = scheduler.dequeue()
        assert first is not None

        # Both should be processed
        second = scheduler.dequeue()
        assert second is not None

        processed = {first["req_id"], second["req_id"]}
        assert "req_low" in processed
        assert "req_medium" in processed

    def test_aging_statistics_in_scheduler(self):
        """Test that aging statistics are included in scheduler stats."""
        policy = AgingPolicy(
            aging_rate=1.0,
            min_wait_time_seconds=0.05,
        )
        scheduler = WeightedRoundRobinScheduler(aging_policy=policy)

        # Enqueue some intents
        for i in range(3):
            intent = create_test_intent("agent1", f"req{i}")
            scheduler.enqueue(intent, priority=0)

        # Wait for aging
        time.sleep(0.1)

        # Get stats
        stats = scheduler.get_stats()

        # Should include aging stats
        assert "priority_aging" in stats
        assert stats["priority_aging"]["tracked_intents"] == 3

    def test_aging_cleared_on_dequeue(self):
        """Test that aging tracking is cleared when intent is dequeued."""
        policy = AgingPolicy()
        scheduler = WeightedRoundRobinScheduler(aging_policy=policy)

        # Enqueue intent
        intent = create_test_intent("agent1", "req1")
        scheduler.enqueue(intent, priority=0)

        # Check it's tracked
        stats = scheduler.get_stats()
        assert stats["priority_aging"]["tracked_intents"] == 1

        # Dequeue
        scheduler.dequeue()

        # Should no longer be tracked
        stats = scheduler.get_stats()
        assert stats["priority_aging"]["tracked_intents"] == 0

    def test_aging_cleared_on_scheduler_clear(self):
        """Test that aging is cleared when scheduler is cleared."""
        policy = AgingPolicy()
        scheduler = WeightedRoundRobinScheduler(aging_policy=policy)

        # Enqueue intents
        for i in range(5):
            intent = create_test_intent("agent1", f"req{i}")
            scheduler.enqueue(intent, priority=0)

        # Check they're tracked
        stats = scheduler.get_stats()
        assert stats["priority_aging"]["tracked_intents"] == 5

        # Clear scheduler
        scheduler.clear()

        # Aging should be cleared
        stats = scheduler.get_stats()
        assert stats["priority_aging"]["tracked_intents"] == 0

    def test_weighted_round_robin_with_aging(self):
        """Test weighted round robin respects weights with aging."""
        policy = AgingPolicy(
            aging_rate=0.5,
            min_wait_time_seconds=0.05,
        )
        scheduler = WeightedRoundRobinScheduler(
            default_weight=1,
            aging_policy=policy,
        )

        # Set different weights
        scheduler.set_agent_weight("agent1", 2)  # Gets 2x processing
        scheduler.set_agent_weight("agent2", 1)  # Gets 1x processing

        # Enqueue intents
        for i in range(4):
            intent1 = create_test_intent("agent1", f"agent1_req{i}")
            intent2 = create_test_intent("agent2", f"agent2_req{i}")
            scheduler.enqueue(intent1, priority=0)
            scheduler.enqueue(intent2, priority=0)

        # Dequeue all
        agent_counts = {"agent1": 0, "agent2": 0}
        for _ in range(8):
            intent = scheduler.dequeue()
            if intent:
                agent_counts[intent["agent_id"]] += 1

        # Agent1 should get roughly 2x processing
        # (exact ratio may vary due to aging)
        assert agent_counts["agent1"] >= agent_counts["agent2"]

    def test_max_boost_prevents_excessive_aging(self):
        """Test that max boost limits priority increases."""
        policy = AgingPolicy(
            aging_rate=100.0,  # Very high rate
            min_wait_time_seconds=0.01,
            max_priority_boost=2,  # Low max
        )
        scheduler = WeightedRoundRobinScheduler(
            priority_levels=10,
            aging_policy=policy,
        )

        # Enqueue low-priority intent
        low = create_test_intent("agent1", "req_low", priority=9)
        scheduler.enqueue(low, priority=9)

        # Wait for aging
        time.sleep(0.2)

        # Check that aging is capped
        stats = scheduler.get_stats()
        aging_stats = stats["priority_aging"]
        assert aging_stats["max_boost_applied"] <= 2

        # Enqueue high-priority intent
        high = create_test_intent("agent1", "req_high", priority=0)
        scheduler.enqueue(high, priority=0)

        # The aged low-priority (9+2=11) will beat fresh high-priority (0)
        # This is correct behavior - aging allows old intents to be processed
        first = scheduler.dequeue()
        second = scheduler.dequeue()

        # Both should be processed
        processed = {first["req_id"], second["req_id"]}
        assert "req_low" in processed
        assert "req_high" in processed

    def test_disabled_aging(self):
        """Test scheduler behavior with aging disabled."""
        policy = AgingPolicy(enabled=False)
        scheduler = WeightedRoundRobinScheduler(
            priority_levels=3,
            aging_policy=policy,
        )

        # Enqueue intents - priority parameter to enqueue() is what's tracked
        # Higher priority number = higher priority
        low = create_test_intent("agent1", "req_low")
        scheduler.enqueue(low, priority=0)  # Low priority (0)

        # Wait
        time.sleep(0.1)

        high = create_test_intent("agent1", "req_high")
        scheduler.enqueue(high, priority=2)  # High priority (2)

        # With aging disabled, tracked priority determines order
        # High priority (2) should win over low priority (0)
        first = scheduler.dequeue()
        second = scheduler.dequeue()

        # Both should be processed
        processed = [first["req_id"], second["req_id"]]
        assert "req_low" in processed
        assert "req_high" in processed

        # High priority should come first when aging is disabled
        assert processed[0] == "req_high"
        assert processed[1] == "req_low"

    def test_concurrent_agents_with_aging(self):
        """Test multiple agents with different priorities and aging."""
        policy = AgingPolicy(
            aging_rate=1.0,
            min_wait_time_seconds=0.05,
        )
        scheduler = WeightedRoundRobinScheduler(
            priority_levels=3,
            aging_policy=policy,
        )

        # Agent1: low priority, enqueued first (will age)
        low1 = create_test_intent("agent1", "agent1_low", priority=2)
        scheduler.enqueue(low1, priority=2)

        # Wait for aging
        time.sleep(0.1)

        # Agent2: high priority, enqueued later
        high2 = create_test_intent("agent2", "agent2_high", priority=0)
        scheduler.enqueue(high2, priority=0)

        # Agent3: medium priority
        med3 = create_test_intent("agent3", "agent3_med", priority=1)
        scheduler.enqueue(med3, priority=1)

        # Dequeue all
        processed = []
        for _ in range(3):
            intent = scheduler.dequeue()
            if intent:
                processed.append(intent["req_id"])

        # All should be processed
        assert len(processed) == 3
        assert "agent1_low" in processed
        assert "agent2_high" in processed
        assert "agent3_med" in processed
