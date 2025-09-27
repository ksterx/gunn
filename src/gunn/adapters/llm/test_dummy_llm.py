"""Unit tests for DummyLLMAdapter cancellation behavior."""

import asyncio
import time

import pytest

from gunn.adapters.llm.dummy_llm import DummyLLMAdapter
from gunn.schemas.types import CancelToken


class TestDummyLLMAdapter:
    """Test suite for DummyLLMAdapter."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        adapter = DummyLLMAdapter(
            generation_time_ms=1000.0,
            token_count=50,
            yield_interval_ms=25.0,
            tokens_per_yield=2,
        )

        assert adapter.generation_time_ms == 1000.0
        assert adapter.token_count == 50
        assert adapter.yield_interval_ms == 25.0
        assert adapter.tokens_per_yield == 2
        assert adapter.total_yields == 25  # 50 tokens / 2 tokens per yield
        assert (
            abs(adapter.actual_yield_interval - 0.04) < 0.001
        )  # 1000ms / 25 yields / 1000

    def test_init_invalid_yield_interval(self):
        """Test initialization with invalid yield interval."""
        with pytest.raises(
            ValueError, match="yield_interval_ms should be between 10-50ms"
        ):
            DummyLLMAdapter(yield_interval_ms=5.0)

        with pytest.raises(
            ValueError, match="yield_interval_ms should be between 10-50ms"
        ):
            DummyLLMAdapter(yield_interval_ms=60.0)

    def test_init_invalid_generation_time(self):
        """Test initialization with invalid generation time."""
        with pytest.raises(ValueError, match="generation_time_ms must be positive"):
            DummyLLMAdapter(generation_time_ms=0)

        with pytest.raises(ValueError, match="generation_time_ms must be positive"):
            DummyLLMAdapter(generation_time_ms=-100)

    def test_init_invalid_token_count(self):
        """Test initialization with invalid token count."""
        with pytest.raises(ValueError, match="token_count must be positive"):
            DummyLLMAdapter(token_count=0)

        with pytest.raises(ValueError, match="token_count must be positive"):
            DummyLLMAdapter(token_count=-10)

    @pytest.mark.asyncio
    async def test_generate_tokens_complete(self):
        """Test complete token generation without cancellation."""
        adapter = DummyLLMAdapter(
            generation_time_ms=100.0,  # Short time for fast test
            token_count=10,
            yield_interval_ms=20.0,
            tokens_per_yield=2,
        )

        cancel_token = CancelToken("req_1", "agent_1")
        tokens = []

        start_time = time.perf_counter()
        async for token in adapter.generate_tokens("test prompt", cancel_token):
            tokens.append(token)
        end_time = time.perf_counter()

        # Verify token generation
        assert len(tokens) == 10
        assert tokens[0] == "token_1"
        assert tokens[-1] == "token_10"

        # Verify timing (should be close to 100ms, allow some tolerance)
        elapsed_ms = (end_time - start_time) * 1000
        assert 80 <= elapsed_ms <= 150  # Allow 50% tolerance for test timing

    @pytest.mark.asyncio
    async def test_generate_tokens_with_max_tokens(self):
        """Test token generation with max_tokens limit."""
        adapter = DummyLLMAdapter(
            generation_time_ms=100.0,
            token_count=20,  # Default count
            yield_interval_ms=20.0,
        )

        cancel_token = CancelToken("req_1", "agent_1")
        tokens = []

        # Generate only 5 tokens despite default of 20
        async for token in adapter.generate_tokens(
            "test prompt", cancel_token, max_tokens=5
        ):
            tokens.append(token)

        assert len(tokens) == 5
        assert tokens == ["token_1", "token_2", "token_3", "token_4", "token_5"]

    @pytest.mark.asyncio
    async def test_cancellation_timing_accuracy(self):
        """Test cancellation timing accuracy within ±5ms tolerance."""
        adapter = DummyLLMAdapter(
            generation_time_ms=1000.0,  # Long generation
            token_count=50,
            yield_interval_ms=25.0,  # Yield every 25ms
        )

        cancel_token = CancelToken("req_1", "agent_1")
        tokens = []

        # Start generation
        generation_task = asyncio.create_task(
            self._collect_tokens(adapter, "test prompt", cancel_token, tokens)
        )

        # Wait 100ms then cancel
        await asyncio.sleep(0.1)
        cancel_start = time.perf_counter()
        cancel_token.cancel("test_cancellation")

        # Wait for cancellation to take effect
        with pytest.raises(asyncio.CancelledError):
            await generation_task

        cancel_end = time.perf_counter()
        cancel_latency_ms = (cancel_end - cancel_start) * 1000

        # Verify cancellation latency is within 100ms SLO
        assert (
            cancel_latency_ms <= 100
        ), f"Cancellation took {cancel_latency_ms}ms, exceeds 100ms SLO"

        # Should have generated some tokens but not all
        assert 0 < len(tokens) < 50

        # Verify cancel token state
        assert cancel_token.cancelled
        assert cancel_token.reason == "test_cancellation"

    @pytest.mark.asyncio
    async def test_immediate_cancellation(self):
        """Test immediate cancellation before generation starts."""
        adapter = DummyLLMAdapter(
            generation_time_ms=1000.0,
            token_count=50,
        )

        cancel_token = CancelToken("req_1", "agent_1")
        cancel_token.cancel("immediate_cancel")  # Cancel before starting

        tokens = []
        with pytest.raises(asyncio.CancelledError, match="immediate_cancel"):
            async for token in adapter.generate_tokens("test prompt", cancel_token):
                tokens.append(token)

        # Should not generate any tokens
        assert len(tokens) == 0

    @pytest.mark.asyncio
    async def test_cancellation_at_token_boundary(self):
        """Test that cancellation is checked at token boundaries."""
        adapter = DummyLLMAdapter(
            generation_time_ms=200.0,
            token_count=10,
            yield_interval_ms=25.0,
            tokens_per_yield=1,  # One token per yield for precise control
        )

        cancel_token = CancelToken("req_1", "agent_1")
        tokens = []

        # Start generation and cancel after a few tokens
        async def cancel_after_delay():
            await asyncio.sleep(0.06)  # Wait for ~2-3 tokens
            cancel_token.cancel("boundary_test")

        cancel_task = asyncio.create_task(cancel_after_delay())

        with pytest.raises(asyncio.CancelledError):
            async for token in adapter.generate_tokens("test prompt", cancel_token):
                tokens.append(token)

        await cancel_task

        # Should have generated some tokens (at least 1, but not all 10)
        assert 1 <= len(tokens) < 10

        # Verify tokens are sequential
        for i, token in enumerate(tokens):
            assert token == f"token_{i + 1}"

    @pytest.mark.asyncio
    async def test_generate_response_complete(self):
        """Test complete response generation."""
        adapter = DummyLLMAdapter(
            generation_time_ms=50.0,
            token_count=5,
        )

        cancel_token = CancelToken("req_1", "agent_1")
        response = await adapter.generate_response("test prompt", cancel_token)

        expected = "token_1 token_2 token_3 token_4 token_5"
        assert response == expected

    @pytest.mark.asyncio
    async def test_generate_response_cancelled(self):
        """Test response generation with cancellation."""
        adapter = DummyLLMAdapter(
            generation_time_ms=1000.0,
            token_count=50,
        )

        cancel_token = CancelToken("req_1", "agent_1")

        # Cancel after short delay
        async def cancel_after_delay():
            await asyncio.sleep(0.05)
            cancel_token.cancel("response_test")

        cancel_task = asyncio.create_task(cancel_after_delay())

        with pytest.raises(asyncio.CancelledError):
            await adapter.generate_response("test prompt", cancel_token)

        await cancel_task

    def test_estimate_generation_time(self):
        """Test generation time estimation."""
        adapter = DummyLLMAdapter(
            generation_time_ms=1000.0,
            token_count=50,
            tokens_per_yield=2,
        )

        # Default token count
        estimated = adapter.estimate_generation_time()
        assert abs(estimated - 1.0) < 0.01  # Should be ~1 second

        # Custom token count
        estimated_custom = adapter.estimate_generation_time(100)
        assert abs(estimated_custom - 2.0) < 0.01  # Should be ~2 seconds

    def test_get_yield_interval(self):
        """Test yield interval getter."""
        adapter = DummyLLMAdapter(
            generation_time_ms=1000.0,
            token_count=50,
            tokens_per_yield=2,
        )

        interval = adapter.get_yield_interval()
        expected = 1000.0 / 25 / 1000.0  # 1000ms / 25 yields / 1000ms per second
        assert abs(interval - expected) < 0.001

    def test_configure_timing(self):
        """Test timing reconfiguration."""
        adapter = DummyLLMAdapter(
            generation_time_ms=1000.0,
            token_count=50,
            yield_interval_ms=25.0,
        )

        # Reconfigure
        adapter.configure_timing(
            generation_time_ms=500.0,
            token_count=25,
            yield_interval_ms=20.0,
        )

        assert adapter.generation_time_ms == 500.0
        assert adapter.token_count == 25
        assert adapter.yield_interval_ms == 20.0

        # Verify recalculated values
        assert adapter.total_yields == 25  # 25 tokens / 1 token per yield
        expected_interval = 500.0 / 25 / 1000.0  # 500ms / 25 yields / 1000ms per second
        assert abs(adapter.actual_yield_interval - expected_interval) < 0.001

    @pytest.mark.asyncio
    async def test_yield_interval_responsiveness(self):
        """Test that yield interval allows responsive cancellation."""
        # Test with 25ms yield interval (within 20-30ms requirement)
        adapter = DummyLLMAdapter(
            generation_time_ms=500.0,
            token_count=20,
            yield_interval_ms=25.0,
            tokens_per_yield=1,
        )

        cancel_token = CancelToken("req_1", "agent_1")
        tokens = []

        start_time = time.perf_counter()

        # Start generation
        generation_task = asyncio.create_task(
            self._collect_tokens(adapter, "test prompt", cancel_token, tokens)
        )

        # Wait for a few yields, then cancel
        await asyncio.sleep(0.08)  # Wait ~80ms (should allow 3-4 tokens)
        cancel_token.cancel("responsiveness_test")

        with pytest.raises(asyncio.CancelledError):
            await generation_task

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Should have cancelled quickly after the cancel signal
        # Total time should be close to 80ms + one yield interval
        assert total_time_ms < 150  # Allow some tolerance

        # Should have generated some tokens but not all
        assert 2 <= len(tokens) <= 6  # Expect 3-4 tokens ±1 for timing variance

    async def _collect_tokens(self, adapter, prompt, cancel_token, tokens_list):
        """Helper to collect tokens into a list."""
        async for token in adapter.generate_tokens(prompt, cancel_token):
            tokens_list.append(token)


@pytest.mark.asyncio
async def test_cancellation_slo_validation():
    """Integration test for 100ms cancellation SLO validation.

    This test validates the requirement that cancellation should halt
    token generation within 100ms of the cancel signal.
    """
    adapter = DummyLLMAdapter(
        generation_time_ms=2000.0,  # Long generation to ensure cancellation
        token_count=100,
        yield_interval_ms=25.0,  # Yield every 25ms for responsiveness
    )

    cancel_token = CancelToken("slo_test", "agent_slo")
    tokens = []

    # Start generation
    async def collect_tokens():
        async for token in adapter.generate_tokens("SLO test prompt", cancel_token):
            tokens.append(token)

    generation_task = asyncio.create_task(collect_tokens())

    # Let it run for a bit, then measure cancellation latency
    await asyncio.sleep(0.2)  # Let some tokens generate

    cancel_start = time.perf_counter()
    cancel_token.cancel("slo_validation")

    # Wait for cancellation to complete
    with pytest.raises(asyncio.CancelledError):
        await generation_task

    cancel_end = time.perf_counter()
    cancellation_latency_ms = (cancel_end - cancel_start) * 1000

    # Validate 100ms SLO with ±5ms tolerance as specified
    assert cancellation_latency_ms <= 105, (
        f"Cancellation latency {cancellation_latency_ms:.1f}ms exceeds "
        f"100ms SLO + 5ms tolerance"
    )

    # Should have generated some tokens but stopped before completion
    assert 5 <= len(tokens) < 100

    print(
        f"✓ Cancellation SLO validated: {cancellation_latency_ms:.1f}ms (≤100ms required)"
    )
