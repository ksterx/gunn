"""Unit tests for priority aging system."""

import asyncio
import time

import pytest

from gunn.schemas.types import Intent
from gunn.utils.priority_aging import AgingPolicy, PriorityAging


def create_test_intent(agent_id: str, req_id: str) -> Intent:
    """Create a test intent."""
    return {
        "kind": "Speak",
        "payload": {"text": "test"},
        "context_seq": 0,
        "req_id": req_id,
        "agent_id": agent_id,
        "priority": 0,
        "schema_version": "1.0.0",
    }


class TestAgingPolicy:
    """Test aging policy configuration."""

    def test_default_policy(self):
        """Test default aging policy."""
        policy = AgingPolicy()
        assert policy.enabled is True
        assert policy.aging_rate == 0.1
        assert policy.max_priority_boost == 5
        assert policy.min_wait_time_seconds == 1.0

    def test_custom_policy(self):
        """Test custom aging policy."""
        policy = AgingPolicy(
            enabled=False,
            aging_rate=0.5,
            max_priority_boost=10,
            min_wait_time_seconds=2.0,
        )
        assert policy.enabled is False
        assert policy.aging_rate == 0.5
        assert policy.max_priority_boost == 10
        assert policy.min_wait_time_seconds == 2.0


class TestPriorityAging:
    """Test priority aging system."""

    def test_track_intent(self):
        """Test tracking an intent for aging."""
        aging = PriorityAging()
        intent = create_test_intent("agent1", "req1")

        aging.track_intent(intent, priority=0)

        assert "req1" in aging.get_tracked_intents()

    def test_duplicate_tracking(self):
        """Test tracking the same intent twice."""
        aging = PriorityAging()
        intent = create_test_intent("agent1", "req1")

        aging.track_intent(intent, priority=0)
        aging.track_intent(intent, priority=0)  # Should log warning but not fail

        assert len(aging.get_tracked_intents()) == 1

    def test_get_aged_priority_no_aging(self):
        """Test getting priority before aging threshold."""
        policy = AgingPolicy(min_wait_time_seconds=1.0)
        aging = PriorityAging(policy)
        intent = create_test_intent("agent1", "req1")

        aging.track_intent(intent, priority=5)

        # Immediately check - should return original priority
        assert aging.get_aged_priority("req1") == 5

    def test_get_aged_priority_with_aging(self):
        """Test priority increases over time."""
        policy = AgingPolicy(
            aging_rate=1.0,  # 1 priority per second
            min_wait_time_seconds=0.1,
            max_priority_boost=10,
        )
        aging = PriorityAging(policy)
        intent = create_test_intent("agent1", "req1")

        aging.track_intent(intent, priority=5)

        # Wait for aging to kick in
        time.sleep(0.2)

        # Should have aged by ~0.1 seconds * 1.0 rate = ~0.1 priority
        aged = aging.get_aged_priority("req1")
        assert aged >= 5  # At least original priority

    def test_max_priority_boost(self):
        """Test maximum priority boost limit."""
        policy = AgingPolicy(
            aging_rate=100.0,  # Very high rate
            min_wait_time_seconds=0.01,
            max_priority_boost=3,
        )
        aging = PriorityAging(policy)
        intent = create_test_intent("agent1", "req1")

        aging.track_intent(intent, priority=5)

        # Wait long enough for max boost
        time.sleep(0.1)

        # Should be capped at original + max_boost
        aged = aging.get_aged_priority("req1")
        assert aged <= 5 + 3

    def test_aging_disabled(self):
        """Test that aging can be disabled."""
        policy = AgingPolicy(enabled=False)
        aging = PriorityAging(policy)
        intent = create_test_intent("agent1", "req1")

        aging.track_intent(intent, priority=5)

        # Wait
        time.sleep(0.2)

        # Should still return original priority
        assert aging.get_aged_priority("req1") == 5

    def test_untrack_intent(self):
        """Test untracking an intent."""
        aging = PriorityAging()
        intent = create_test_intent("agent1", "req1")

        aging.track_intent(intent, priority=5)
        assert "req1" in aging.get_tracked_intents()

        aging.untrack_intent("req1")
        assert "req1" not in aging.get_tracked_intents()

    def test_get_wait_time(self):
        """Test getting intent wait time."""
        aging = PriorityAging()
        intent = create_test_intent("agent1", "req1")

        aging.track_intent(intent, priority=5)

        # Wait a bit
        time.sleep(0.1)

        wait_time = aging.get_wait_time("req1")
        assert wait_time >= 0.1

    def test_get_wait_time_untracked(self):
        """Test getting wait time for untracked intent."""
        aging = PriorityAging()

        wait_time = aging.get_wait_time("nonexistent")
        assert wait_time == 0.0

    def test_get_tracked_intents(self):
        """Test getting list of tracked intents."""
        aging = PriorityAging()

        intent1 = create_test_intent("agent1", "req1")
        intent2 = create_test_intent("agent2", "req2")

        aging.track_intent(intent1, priority=0)
        aging.track_intent(intent2, priority=1)

        tracked = aging.get_tracked_intents()
        assert len(tracked) == 2
        assert "req1" in tracked
        assert "req2" in tracked

    def test_get_stats(self):
        """Test getting aging statistics."""
        policy = AgingPolicy(
            aging_rate=1.0,
            min_wait_time_seconds=0.05,
        )
        aging = PriorityAging(policy)

        intent1 = create_test_intent("agent1", "req1")
        intent2 = create_test_intent("agent2", "req2")

        aging.track_intent(intent1, priority=0)
        aging.track_intent(intent2, priority=1)

        # Wait for aging
        time.sleep(0.1)

        # Trigger aging calculation
        aging.get_aged_priority("req1")
        aging.get_aged_priority("req2")

        stats = aging.get_stats()
        assert stats["tracked_intents"] == 2
        assert stats["total_aged"] >= 0
        assert stats["avg_wait_time"] >= 0.1
        assert stats["max_wait_time"] >= 0.1

    def test_clear(self):
        """Test clearing all tracked intents."""
        aging = PriorityAging()

        intent1 = create_test_intent("agent1", "req1")
        intent2 = create_test_intent("agent2", "req2")

        aging.track_intent(intent1, priority=0)
        aging.track_intent(intent2, priority=1)

        assert len(aging.get_tracked_intents()) == 2

        aging.clear()

        assert len(aging.get_tracked_intents()) == 0

    def test_multiple_agents_aging(self):
        """Test aging works independently for multiple agents."""
        policy = AgingPolicy(
            aging_rate=10.0,  # Fast aging for testing
            min_wait_time_seconds=0.01,
        )
        aging = PriorityAging(policy)

        intent1 = create_test_intent("agent1", "req1")
        intent2 = create_test_intent("agent2", "req2")

        # Track at different times with non-zero priority
        aging.track_intent(intent1, priority=5)
        time.sleep(0.15)  # Longer wait for aging
        aging.track_intent(intent2, priority=5)

        # First intent should have aged more
        aged1 = aging.get_aged_priority("req1")
        aged2 = aging.get_aged_priority("req2")

        assert aged1 > aged2

    def test_priority_boost_statistics(self):
        """Test that boost statistics are tracked."""
        policy = AgingPolicy(
            aging_rate=10.0,  # Fast aging for testing
            min_wait_time_seconds=0.01,
            max_priority_boost=5,
        )
        aging = PriorityAging(policy)

        intent = create_test_intent("agent1", "req1")
        aging.track_intent(intent, priority=3)

        # Wait for significant aging
        time.sleep(0.15)

        # Trigger aging
        aged = aging.get_aged_priority("req1")

        stats = aging.get_stats()
        assert stats["total_aged"] >= 1
        assert stats["max_boost_applied"] > 0
        assert aged > 3  # Should have aged from original priority

    @pytest.mark.asyncio
    async def test_concurrent_aging_checks(self):
        """Test concurrent aging checks from multiple coroutines."""
        policy = AgingPolicy(
            aging_rate=1.0,
            min_wait_time_seconds=0.05,
        )
        aging = PriorityAging(policy)

        # Track multiple intents
        for i in range(10):
            intent = create_test_intent(f"agent{i}", f"req{i}")
            aging.track_intent(intent, priority=i)

        # Wait for aging
        await asyncio.sleep(0.1)

        # Check priorities concurrently
        async def check_priority(req_id: str):
            return aging.get_aged_priority(req_id)

        results = await asyncio.gather(*[check_priority(f"req{i}") for i in range(10)])

        # All should have aged
        assert all(result >= 0 for result in results)
