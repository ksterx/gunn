"""Unit tests for dummy LLM adapter cancellation behavior.

Tests the 100ms cancellation SLO and token yield timing accuracy
within ±5ms tolerance as specified in the requirements.
"""

import asyncio
import time

import pytest

from gunn.adapters.llm import CancellationTestHelper, DummyLLMAdapter
from gunn.schemas.types import CancelToken


class TestDummyLLMAdapter:
    """Test suite for DummyLLMAdapter cancellation behavior."""

    def test_init_validation(self):
        """Test initialization parameter validation."""
        # Valid parameters
        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        assert adapter.token_interval_ms == 25.0

        # Invalid token interval (too low)
        with pytest.raises(
            ValueError, match="token_interval_ms should be between 20-30ms"
        ):
            DummyLLMAdapter(token_interval_ms=15.0)

        # Invalid token interval (too high)
        with pytest.raises(
            ValueError, match="token_interval_ms should be between 20-30ms"
        ):
            DummyLLMAdapter(token_interval_ms=35.0)

    @pytest.mark.asyncio
    async def test_basic_token_generation(self):
        """Test basic token generation without cancellation."""
        adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=200.0,
        )
        cancel_token = CancelToken("req_1", "agent_1")

        tokens = []
        start_time = time.perf_counter()

        async for token in adapter.generate_stream(
            "test prompt", cancel_token, max_tokens=5
        ):
            tokens.append(token)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should generate exactly 5 tokens
        assert len(tokens) == 5
        assert tokens == ["token_1", "token_2", "token_3", "token_4", "token_5"]

        # Should take approximately 5 * 25ms = 125ms
        assert 100 <= elapsed_ms <= 200  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_generation_time_limit(self):
        """Test that generation stops after configured time limit."""
        adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=100.0,  # Short time limit
        )
        cancel_token = CancelToken("req_2", "agent_2")

        tokens = []
        start_time = time.perf_counter()

        async for token in adapter.generate_stream(
            "test prompt", cancel_token, max_tokens=1000
        ):
            tokens.append(token)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should stop due to time limit, not token limit
        assert len(tokens) < 1000
        assert 90 <= elapsed_ms <= 150  # Should be close to 100ms limit

    @pytest.mark.asyncio
    async def test_immediate_cancellation(self):
        """Test cancellation before any tokens are generated."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        cancel_token = CancelToken("req_3", "agent_3")

        # Cancel immediately
        cancel_token.cancel("immediate_test")

        tokens = []
        with pytest.raises(
            asyncio.CancelledError, match="Generation cancelled: immediate_test"
        ):
            async for token in adapter.generate_stream(
                "test prompt", cancel_token, max_tokens=10
            ):
                tokens.append(token)

        # Should not generate any tokens
        assert len(tokens) == 0

    @pytest.mark.asyncio
    async def test_cancellation_during_generation(self):
        """Test cancellation during token generation."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        cancel_token = CancelToken("req_4", "agent_4")

        # Start generation and cancel after 60ms
        async def cancel_after_delay() -> None:
            await asyncio.sleep(0.06)  # 60ms
            cancel_token.cancel("mid_generation_test")

        cancel_task = asyncio.create_task(cancel_after_delay())

        tokens = []
        start_time = time.perf_counter()

        with pytest.raises(
            asyncio.CancelledError, match="Generation cancelled: mid_generation_test"
        ):
            async for token in adapter.generate_stream(
                "test prompt", cancel_token, max_tokens=100
            ):
                tokens.append(token)

        await cancel_task
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should have generated some tokens but not all
        assert 0 < len(tokens) < 100
        # Should be cancelled around 60ms + token_interval
        assert 50 <= elapsed_ms <= 100

    @pytest.mark.asyncio
    async def test_cancellation_timing_accuracy(self):
        """Test that cancellation response time is within ±5ms tolerance."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        cancel_token = CancelToken("req_5", "agent_5")

        # Test cancellation timing with helper
        (
            within_tolerance,
            actual_time,
            expected_tolerance,
        ) = await CancellationTestHelper.test_cancellation_timing(
            adapter, cancel_token, cancel_after_ms=50.0, tolerance_ms=5.0
        )

        assert within_tolerance, (
            f"Cancellation took {actual_time:.2f}ms, expected ≤{expected_tolerance:.2f}ms"
        )
        assert actual_time <= 5.0  # Should be within 5ms tolerance

    @pytest.mark.asyncio
    async def test_100ms_cancellation_slo(self):
        """Test the 100ms cancellation SLO requirement."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0)

        # Test multiple cancellation scenarios
        test_cases = [
            (30.0, "early_cancellation"),
            (50.0, "mid_cancellation"),
            (75.0, "late_cancellation"),
        ]

        for cancel_after_ms, test_name in test_cases:
            cancel_token = CancelToken(f"req_{test_name}", "agent_slo")

            (
                within_tolerance,
                actual_time,
                _,
            ) = await CancellationTestHelper.test_cancellation_timing(
                adapter,
                cancel_token,
                cancel_after_ms=cancel_after_ms,
                tolerance_ms=100.0,
            )

            assert within_tolerance, (
                f"{test_name}: Cancellation took {actual_time:.2f}ms, exceeds 100ms SLO"
            )
            assert actual_time <= 100.0, (
                f"{test_name}: Failed 100ms SLO with {actual_time:.2f}ms"
            )

    @pytest.mark.asyncio
    async def test_token_yield_intervals(self):
        """Test that tokens are yielded at consistent intervals."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        cancel_token = CancelToken("req_6", "agent_6")

        # Measure token yield intervals
        intervals = await CancellationTestHelper.measure_token_yield_interval(
            adapter, cancel_token, measurement_duration_ms=150.0
        )

        # Should have multiple intervals
        assert len(intervals) >= 3

        # Each interval should be close to 25ms (±5ms tolerance)
        for interval in intervals:
            assert 20.0 <= interval <= 30.0, (
                f"Token interval {interval:.2f}ms outside 20-30ms range"
            )

        # Average should be close to configured interval (allow some system scheduling variance)
        avg_interval = sum(intervals) / len(intervals)
        assert 20.0 <= avg_interval <= 30.0, (
            f"Average interval {avg_interval:.2f}ms not close to 25ms"
        )

    @pytest.mark.asyncio
    async def test_generate_with_timing(self):
        """Test the generate_with_timing helper method."""
        adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=150.0,
        )
        cancel_token = CancelToken("req_7", "agent_7")

        # Test normal completion
        (
            tokens,
            generation_time_ms,
            cancellation_time_ms,
        ) = await adapter.generate_with_timing(
            "test prompt", cancel_token, max_tokens=3
        )

        assert len(tokens) == 3
        assert tokens == ["token_1", "token_2", "token_3"]
        assert 60 <= generation_time_ms <= 120  # ~75ms for 3 tokens
        assert cancellation_time_ms is None  # Not cancelled

    @pytest.mark.asyncio
    async def test_generate_with_timing_cancelled(self):
        """Test generate_with_timing with cancellation."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        cancel_token = CancelToken("req_8", "agent_8")

        # Cancel after 40ms
        async def cancel_after_delay() -> None:
            await asyncio.sleep(0.04)
            cancel_token.cancel("timing_test_cancellation")

        cancel_task = asyncio.create_task(cancel_after_delay())

        with pytest.raises(asyncio.CancelledError):
            await adapter.generate_with_timing(
                "test prompt", cancel_token, max_tokens=100
            )

        await cancel_task

    @pytest.mark.asyncio
    async def test_cancellation_during_sleep(self):
        """Test that cancellation is detected during token interval sleep."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        cancel_token = CancelToken("req_9", "agent_9")

        tokens: list[str] = []

        # Start generation
        generation_task = asyncio.create_task(
            self._collect_tokens(adapter, cancel_token, tokens)
        )

        # Wait for first token, then cancel during sleep
        await asyncio.sleep(0.03)  # Wait for first token
        assert len(tokens) >= 1  # Should have at least one token

        # Cancel during the sleep period
        cancel_token.cancel("sleep_cancellation_test")

        # Should raise CancelledError quickly
        start_time = time.perf_counter()
        with pytest.raises(asyncio.CancelledError):
            await generation_task

        cancellation_response_time = (time.perf_counter() - start_time) * 1000

        # Should respond to cancellation within a few milliseconds
        assert cancellation_response_time <= 10.0, (
            f"Cancellation during sleep took {cancellation_response_time:.2f}ms"
        )

    async def _collect_tokens(
        self, adapter: DummyLLMAdapter, cancel_token: CancelToken, tokens: list[str]
    ) -> None:
        """Helper to collect tokens from generation stream."""
        async for token in adapter.generate_stream(
            "test", cancel_token, max_tokens=100
        ):
            tokens.append(token)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_generations(self):
        """Test multiple concurrent generations with different cancellation patterns."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0)

        # Create multiple cancel tokens
        tokens_1 = CancelToken("req_10a", "agent_10a")
        tokens_2 = CancelToken("req_10b", "agent_10b")
        tokens_3 = CancelToken("req_10c", "agent_10c")

        # Start multiple generations
        task_1 = asyncio.create_task(
            self._collect_tokens_list(adapter, tokens_1, max_tokens=10)
        )
        task_2 = asyncio.create_task(
            self._collect_tokens_list(adapter, tokens_2, max_tokens=10)
        )
        task_3 = asyncio.create_task(
            self._collect_tokens_list(adapter, tokens_3, max_tokens=10)
        )

        # Cancel one after 40ms
        await asyncio.sleep(0.04)
        tokens_2.cancel("concurrent_test")

        # Wait for all tasks
        results = await asyncio.gather(task_1, task_2, task_3, return_exceptions=True)

        # Task 1 and 3 should complete normally
        assert isinstance(results[0], list)
        assert len(results[0]) == 10

        assert isinstance(results[2], list)
        assert len(results[2]) == 10

        # Task 2 should be cancelled
        assert isinstance(results[1], asyncio.CancelledError)

    async def _collect_tokens_list(
        self, adapter: DummyLLMAdapter, cancel_token: CancelToken, max_tokens: int
    ) -> list[str]:
        """Helper to collect tokens into a list."""
        tokens = []
        async for token in adapter.generate_stream(
            "test", cancel_token, max_tokens=max_tokens
        ):
            tokens.append(token)
        return tokens

    def test_cancel_token_integration(self):
        """Test that CancelToken is properly integrated."""
        cancel_token = CancelToken("req_11", "agent_11")

        # Test initial state
        assert not cancel_token.cancelled
        assert cancel_token.reason is None
        assert cancel_token.req_id == "req_11"
        assert cancel_token.agent_id == "agent_11"

        # Test cancellation
        cancel_token.cancel("integration_test")
        assert cancel_token.cancelled
        assert cancel_token.reason == "integration_test"  # type: ignore

    @pytest.mark.asyncio
    async def test_edge_case_zero_tokens(self):
        """Test generation with max_tokens=0."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        cancel_token = CancelToken("req_12", "agent_12")

        tokens = []
        async for token in adapter.generate_stream("test", cancel_token, max_tokens=0):
            tokens.append(token)

        assert len(tokens) == 0

    @pytest.mark.asyncio
    async def test_edge_case_very_short_generation_time(self):
        """Test with very short generation time."""
        adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=10.0,  # Very short
        )
        cancel_token = CancelToken("req_13", "agent_13")

        tokens = []
        start_time = time.perf_counter()

        async for token in adapter.generate_stream(
            "test", cancel_token, max_tokens=100
        ):
            tokens.append(token)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should generate very few or no tokens due to time limit
        assert len(tokens) <= 1
        assert elapsed_ms <= 50  # Should stop quickly
