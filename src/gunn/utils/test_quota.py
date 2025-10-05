"""Unit tests for quota management and rate limiting."""

import asyncio

import pytest

from gunn.utils.errors import QuotaExceededError
from gunn.utils.quota import QuotaController, QuotaPolicy, TokenBucket


class TestTokenBucket:
    """Test token bucket rate limiting."""

    @pytest.mark.asyncio
    async def test_basic_consumption(self):
        """Test basic token consumption."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, refill_interval=1.0)

        # Should be able to consume up to capacity
        assert await bucket.consume(5.0)
        assert await bucket.consume(5.0)
        assert not await bucket.consume(1.0)  # Bucket empty

    @pytest.mark.asyncio
    async def test_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10.0, refill_rate=5.0, refill_interval=0.1)

        # Consume all tokens
        assert await bucket.consume(10.0)
        assert not await bucket.consume(1.0)

        # Wait for refill
        await asyncio.sleep(0.15)

        # Should have ~5 tokens now
        assert await bucket.consume(4.0)

    @pytest.mark.asyncio
    async def test_wait_for_tokens(self):
        """Test waiting for tokens to become available."""
        bucket = TokenBucket(capacity=10.0, refill_rate=10.0, refill_interval=0.1)

        # Consume all tokens
        assert await bucket.consume(10.0)

        # Wait for tokens (should succeed after refill)
        assert await bucket.wait_for_tokens(5.0, timeout=1.0)

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        """Test timeout when waiting for tokens."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, refill_interval=1.0)

        # Consume all tokens
        assert await bucket.consume(10.0)

        # Wait with short timeout (should fail)
        assert not await bucket.wait_for_tokens(20.0, timeout=0.1)

    @pytest.mark.asyncio
    async def test_get_available_tokens(self):
        """Test getting available token count."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, refill_interval=1.0)

        # Initial capacity
        available = await bucket.get_available_tokens()
        assert available == 10.0

        # After consumption
        await bucket.consume(3.0)
        available = await bucket.get_available_tokens()
        assert available == 7.0

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test bucket reset."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, refill_interval=1.0)

        # Consume tokens
        await bucket.consume(8.0)
        assert await bucket.get_available_tokens() == 2.0

        # Reset
        await bucket.reset()
        assert await bucket.get_available_tokens() == 10.0


class TestQuotaPolicy:
    """Test quota policy configuration."""

    def test_default_burst_size(self):
        """Test default burst size calculation."""
        policy = QuotaPolicy(intents_per_minute=60)
        assert policy.burst_size == 10  # 60 / 6

        policy = QuotaPolicy(intents_per_minute=120)
        assert policy.burst_size == 20  # 120 / 6

    def test_custom_burst_size(self):
        """Test custom burst size."""
        policy = QuotaPolicy(intents_per_minute=60, burst_size=20)
        assert policy.burst_size == 20


class TestQuotaController:
    """Test quota controller."""

    @pytest.mark.asyncio
    async def test_intent_quota_enforcement(self):
        """Test intent quota enforcement."""
        policy = QuotaPolicy(intents_per_minute=60, burst_size=5)
        controller = QuotaController(policy)

        # Should allow up to burst size
        for _ in range(5):
            await controller.check_intent_quota("agent1")

        # Should exceed quota
        with pytest.raises(QuotaExceededError) as exc_info:
            await controller.check_intent_quota("agent1")

        assert exc_info.value.agent_id == "agent1"
        assert exc_info.value.quota_type == "intents_per_minute"

    @pytest.mark.asyncio
    async def test_token_quota_enforcement(self):
        """Test token quota enforcement."""
        policy = QuotaPolicy(tokens_per_minute=600)
        controller = QuotaController(policy)

        # Should allow tokens up to limit
        await controller.check_token_quota("agent1", 50)
        await controller.check_token_quota("agent1", 50)

        # Should exceed quota
        with pytest.raises(QuotaExceededError) as exc_info:
            await controller.check_token_quota("agent1", 1000)

        assert exc_info.value.agent_id == "agent1"
        assert exc_info.value.quota_type == "tokens_per_minute"

    @pytest.mark.asyncio
    async def test_per_agent_isolation(self):
        """Test that quotas are isolated per agent."""
        policy = QuotaPolicy(intents_per_minute=60, burst_size=3)
        controller = QuotaController(policy)

        # Agent 1 uses quota
        for _ in range(3):
            await controller.check_intent_quota("agent1")

        # Agent 1 should be blocked
        with pytest.raises(QuotaExceededError):
            await controller.check_intent_quota("agent1")

        # Agent 2 should still have quota
        await controller.check_intent_quota("agent2")

    @pytest.mark.asyncio
    async def test_quota_refill(self):
        """Test quota refills over time."""
        policy = QuotaPolicy(
            intents_per_minute=60,
            burst_size=2,
            refill_interval_seconds=0.1,
        )
        controller = QuotaController(policy)

        # Use up quota
        await controller.check_intent_quota("agent1")
        await controller.check_intent_quota("agent1")

        # Should be blocked
        with pytest.raises(QuotaExceededError):
            await controller.check_intent_quota("agent1")

        # Wait for refill
        await asyncio.sleep(0.15)

        # Should have quota again
        await controller.check_intent_quota("agent1")

    @pytest.mark.asyncio
    async def test_wait_for_quota(self):
        """Test waiting for quota to become available."""
        policy = QuotaPolicy(
            intents_per_minute=60,
            burst_size=1,
            refill_interval_seconds=0.1,
        )
        controller = QuotaController(policy)

        # Use quota
        await controller.check_intent_quota("agent1")

        # Wait for quota (should succeed after refill)
        await controller.check_intent_quota("agent1", wait=True)

    @pytest.mark.asyncio
    async def test_get_available_quotas(self):
        """Test getting available quota amounts."""
        policy = QuotaPolicy(intents_per_minute=60, burst_size=5)
        controller = QuotaController(policy)

        # Check initial quota
        available = await controller.get_available_intents("agent1")
        assert available == 5.0

        # Use some quota
        await controller.check_intent_quota("agent1")
        available = await controller.get_available_intents("agent1")
        assert available == 4.0

    @pytest.mark.asyncio
    async def test_reset_agent_quota(self):
        """Test resetting agent quota."""
        policy = QuotaPolicy(intents_per_minute=60, burst_size=3)
        controller = QuotaController(policy)

        # Use quota
        for _ in range(3):
            await controller.check_intent_quota("agent1")

        # Should be blocked
        with pytest.raises(QuotaExceededError):
            await controller.check_intent_quota("agent1")

        # Reset quota
        await controller.reset_agent_quota("agent1")

        # Should have quota again
        await controller.check_intent_quota("agent1")

    @pytest.mark.asyncio
    async def test_quota_statistics(self):
        """Test quota statistics collection."""
        policy = QuotaPolicy(intents_per_minute=60, tokens_per_minute=1000)
        controller = QuotaController(policy)

        # Generate some activity
        await controller.check_intent_quota("agent1")
        await controller.check_intent_quota("agent2")
        await controller.check_token_quota("agent1", 100)

        # Check stats
        stats = controller.get_stats()
        assert stats["total_intents"]["agent1"] == 1
        assert stats["total_intents"]["agent2"] == 1
        assert stats["total_tokens"]["agent1"] == 100
        assert stats["tracked_agents"] == 2

    @pytest.mark.asyncio
    async def test_remove_agent(self):
        """Test removing agent from quota controller."""
        policy = QuotaPolicy(intents_per_minute=60)
        controller = QuotaController(policy)

        # Use quota
        await controller.check_intent_quota("agent1")

        # Remove agent
        await controller.remove_agent("agent1")

        # Agent should have fresh quota
        stats = controller.get_stats()
        assert "agent1" not in stats["total_intents"]

    @pytest.mark.asyncio
    async def test_concurrent_quota_checks(self):
        """Test concurrent quota checks from multiple agents."""
        policy = QuotaPolicy(intents_per_minute=60, burst_size=10)
        controller = QuotaController(policy)

        async def check_quota(agent_id: str, count: int):
            for _ in range(count):
                await controller.check_intent_quota(agent_id)

        # Run concurrent checks
        await asyncio.gather(
            check_quota("agent1", 5),
            check_quota("agent2", 5),
            check_quota("agent3", 5),
        )

        # Verify stats
        stats = controller.get_stats()
        assert stats["total_intents"]["agent1"] == 5
        assert stats["total_intents"]["agent2"] == 5
        assert stats["total_intents"]["agent3"] == 5
