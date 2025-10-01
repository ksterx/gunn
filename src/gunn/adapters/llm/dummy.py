"""Dummy LLM adapter for cancellation testing.

This module provides a mock LLM adapter that simulates token generation
with configurable timing and proper cancellation support for testing
the 100ms cancellation SLO.
"""

import asyncio
import math
import time
from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable

from gunn.schemas.types import CancelToken
from gunn.utils.telemetry import get_logger


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for LLM adapters with streaming support."""

    async def generate_stream(
        self,
        prompt: str,
        cancel_token: CancelToken,
        max_tokens: int = 100,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming tokens with cancellation support.

        Args:
            prompt: Input prompt for generation
            cancel_token: Token for cancellation signaling
            max_tokens: Maximum number of tokens to generate

        Yields:
            Generated tokens as strings

        Raises:
            asyncio.CancelledError: If generation is cancelled
        """
        ...


class DummyLLMAdapter:
    """Mock LLM adapter for testing cancellation behavior.

    This adapter simulates token generation with configurable timing
    to test the 100ms cancellation SLO requirement. It yields control
    every 20-30ms to ensure responsive cancellation.

    Requirements addressed:
    - 6.2: Monitor for cancellation signals at token boundaries
    - 6.3: Immediately halt token generation within 100ms
    - 11.2: Cancel-to-halt latency ≤ 100ms at token boundaries
    """

    def __init__(
        self,
        token_interval_ms: float = 25.0,
        generation_time_ms: float = 1000.0,
        tokens_per_second: float = 40.0,
    ):
        """Initialize dummy LLM adapter.

        Args:
            token_interval_ms: Time between token yields (20-30ms recommended)
            generation_time_ms: Total generation time for testing
            tokens_per_second: Rate of token generation
        """
        if not 20.0 <= token_interval_ms <= 30.0:
            raise ValueError(
                "token_interval_ms should be between 20-30ms for responsive cancellation"
            )

        self.token_interval_ms = token_interval_ms
        self.generation_time_ms = generation_time_ms
        self.tokens_per_second = tokens_per_second
        self._logger = get_logger("gunn.adapters.llm.dummy")

    async def generate_stream(
        self,
        prompt: str,
        cancel_token: CancelToken,
        max_tokens: int = 100,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming tokens with cancellation support.

        Yields control every token_interval_ms to check for cancellation.
        Ensures cancellation response within 100ms SLO.

        Args:
            prompt: Input prompt for generation
            cancel_token: Token for cancellation signaling
            max_tokens: Maximum number of tokens to generate

        Yields:
            Generated tokens as strings

        Raises:
            asyncio.CancelledError: If generation is cancelled via cancel_token
        """
        start_time = time.perf_counter()
        tokens_generated = 0

        # Calculate timing parameters
        token_interval_s = self.token_interval_ms / 1000.0
        max_generation_time_s = self.generation_time_ms / 1000.0

        self._logger.info(
            "Starting token generation",
            req_id=cancel_token.req_id,
            agent_id=cancel_token.agent_id,
            max_tokens=max_tokens,
            token_interval_ms=self.token_interval_ms,
            generation_time_ms=self.generation_time_ms,
        )

        try:
            while tokens_generated < max_tokens:
                # Check cancellation at token boundary
                if cancel_token.cancelled:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    self._logger.info(
                        "Generation cancelled",
                        req_id=cancel_token.req_id,
                        agent_id=cancel_token.agent_id,
                        tokens_generated=tokens_generated,
                        elapsed_ms=elapsed_ms,
                        reason=cancel_token.reason,
                    )
                    raise asyncio.CancelledError(
                        f"Generation cancelled: {cancel_token.reason}"
                    )

                # Check if we've exceeded max generation time
                elapsed_time = time.perf_counter() - start_time
                if elapsed_time >= max_generation_time_s:
                    self._logger.info(
                        "Generation completed (time limit)",
                        req_id=cancel_token.req_id,
                        agent_id=cancel_token.agent_id,
                        tokens_generated=tokens_generated,
                        elapsed_ms=elapsed_time * 1000,
                    )
                    break

                # Generate next token
                token = f"token_{tokens_generated + 1}"
                tokens_generated += 1

                # Yield the token
                yield token

                # Sleep for token interval, but check cancellation periodically
                # Split the sleep into smaller chunks for more responsive cancellation
                sleep_chunks = max(
                    1, math.ceil(token_interval_s / 0.002)
                )  # ≤2ms chunks
                chunk_sleep = token_interval_s / sleep_chunks

                for _ in range(sleep_chunks):
                    if cancel_token.cancelled:
                        # Final cancellation check during sleep
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        self._logger.info(
                            "Generation cancelled during sleep",
                            req_id=cancel_token.req_id,
                            agent_id=cancel_token.agent_id,
                            tokens_generated=tokens_generated,
                            elapsed_ms=elapsed_ms,
                            reason=cancel_token.reason,
                        )
                        raise asyncio.CancelledError(
                            f"Generation cancelled: {cancel_token.reason}"
                        )

                    await asyncio.sleep(chunk_sleep)

        except asyncio.CancelledError:
            # Re-raise cancellation
            raise
        except Exception as e:
            self._logger.error(
                "Generation failed",
                req_id=cancel_token.req_id,
                agent_id=cancel_token.agent_id,
                tokens_generated=tokens_generated,
                error=str(e),
            )
            raise

        # Log completion
        final_elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._logger.info(
            "Generation completed",
            req_id=cancel_token.req_id,
            agent_id=cancel_token.agent_id,
            tokens_generated=tokens_generated,
            elapsed_ms=final_elapsed_ms,
        )

    async def generate_with_timing(
        self,
        prompt: str,
        cancel_token: CancelToken,
        max_tokens: int = 100,
    ) -> tuple[list[str], float, float | None]:
        """Generate tokens and return timing information for testing.

        Args:
            prompt: Input prompt for generation
            cancel_token: Token for cancellation signaling
            max_tokens: Maximum number of tokens to generate

        Returns:
            Tuple of (tokens, generation_time_ms, cancellation_time_ms)
            cancellation_time_ms is None if not cancelled

        Raises:
            asyncio.CancelledError: If generation is cancelled
        """
        tokens = []
        start_time = time.perf_counter()
        cancellation_time_ms = None

        try:
            async for token in self.generate_stream(prompt, cancel_token, max_tokens):
                tokens.append(token)
        except asyncio.CancelledError:
            cancellation_time_ms = (time.perf_counter() - start_time) * 1000
            raise

        generation_time_ms = (time.perf_counter() - start_time) * 1000
        return tokens, generation_time_ms, cancellation_time_ms


class CancellationTestHelper:
    """Helper class for testing cancellation timing accuracy."""

    @staticmethod
    async def test_cancellation_timing(
        adapter: DummyLLMAdapter,
        cancel_token: CancelToken,
        cancel_after_ms: float,
        tolerance_ms: float = 5.0,
    ) -> tuple[bool, float, float]:
        """Test cancellation timing accuracy.

        Args:
            adapter: LLM adapter to test
            cancel_token: Token to cancel
            cancel_after_ms: Time to wait before cancelling
            tolerance_ms: Acceptable timing tolerance

        Returns:
            Tuple of (within_tolerance, actual_cancel_time_ms, expected_cancel_time_ms)
        """
        # Start generation task
        generation_task = asyncio.create_task(
            adapter.generate_with_timing("test prompt", cancel_token, max_tokens=1000)
        )

        # Schedule cancellation
        async def cancel_after_delay() -> float:
            await asyncio.sleep(cancel_after_ms / 1000.0)
            cancel_start = time.perf_counter()
            cancel_token.cancel("test_cancellation")
            return cancel_start

        cancel_task = asyncio.create_task(cancel_after_delay())

        try:
            # Wait for both tasks
            _ = await cancel_task
            await generation_task

            # Should not reach here if cancellation worked
            return False, 0.0, cancel_after_ms

        except asyncio.CancelledError:
            # Calculate actual cancellation response time
            cancel_response_time = time.perf_counter()
            actual_cancel_time_ms = (cancel_response_time - await cancel_task) * 1000

            # Check if within tolerance
            within_tolerance = actual_cancel_time_ms <= tolerance_ms

            return within_tolerance, actual_cancel_time_ms, tolerance_ms

    @staticmethod
    async def measure_token_yield_interval(
        adapter: DummyLLMAdapter,
        cancel_token: CancelToken,
        measurement_duration_ms: float = 200.0,
    ) -> list[float]:
        """Measure actual token yield intervals.

        Args:
            adapter: LLM adapter to test
            cancel_token: Token for generation
            measurement_duration_ms: How long to measure

        Returns:
            List of intervals between token yields in milliseconds
        """
        intervals: list[float] = []
        last_token_time = None
        start_time = time.perf_counter()

        try:
            async for _ in adapter.generate_stream(
                "test", cancel_token, max_tokens=1000
            ):
                current_time = time.perf_counter()

                if last_token_time is not None:
                    interval_ms = (current_time - last_token_time) * 1000
                    intervals.append(interval_ms)

                last_token_time = current_time

                # Stop after measurement duration
                if (current_time - start_time) * 1000 >= measurement_duration_ms:
                    cancel_token.cancel("measurement_complete")
                    break

        except asyncio.CancelledError:
            pass

        return intervals
