"""Dummy LLM adapter for testing cancellation behavior.

This module provides a mock LLM adapter that simulates token generation
with configurable timing and proper cancel token integration for testing
the 100ms cancellation SLO.
"""

import asyncio
import time
from collections.abc import AsyncGenerator

from gunn.schemas.types import CancelToken
from gunn.utils.telemetry import get_logger


class DummyLLMAdapter:
    """Mock LLM adapter for testing cancellation behavior.

    This adapter simulates token generation with configurable timing
    and yields control every 20-30ms for responsive cancellation testing.

    Requirements addressed:
    - 6.2: Monitor for cancellation signals at token boundaries
    - 6.3: Immediately halt token generation within 100ms
    - 11.2: 100ms cancellation SLO validation
    """

    def __init__(
        self,
        generation_time_ms: float = 1000.0,
        token_count: int = 50,
        yield_interval_ms: float = 25.0,
        tokens_per_yield: int = 1,
    ):
        """Initialize dummy LLM adapter.

        Args:
            generation_time_ms: Total generation time in milliseconds
            token_count: Number of tokens to generate
            yield_interval_ms: Time between yields (20-30ms recommended)
            tokens_per_yield: Number of tokens to generate per yield
        """
        if yield_interval_ms < 10 or yield_interval_ms > 50:
            raise ValueError(
                "yield_interval_ms should be between 10-50ms for responsive cancellation"
            )

        if generation_time_ms <= 0:
            raise ValueError("generation_time_ms must be positive")

        if token_count <= 0:
            raise ValueError("token_count must be positive")

        self.generation_time_ms = generation_time_ms
        self.token_count = token_count
        self.yield_interval_ms = yield_interval_ms
        self.tokens_per_yield = tokens_per_yield

        # Calculate timing
        self.total_yields = max(1, token_count // tokens_per_yield)
        self.actual_yield_interval = (
            generation_time_ms / self.total_yields / 1000.0
        )  # Convert to seconds

        self._logger = get_logger("gunn.adapters.llm.dummy")

        self._logger.debug(
            "DummyLLM initialized",
            generation_time_ms=generation_time_ms,
            token_count=token_count,
            yield_interval_ms=yield_interval_ms,
            tokens_per_yield=tokens_per_yield,
            total_yields=self.total_yields,
            actual_yield_interval_s=self.actual_yield_interval,
        )

    async def generate_tokens(
        self, prompt: str, cancel_token: CancelToken, max_tokens: int | None = None
    ) -> AsyncGenerator[str, None]:
        """Generate tokens with cancellation support.

        Args:
            prompt: Input prompt (ignored in dummy implementation)
            cancel_token: Token for cancellation monitoring
            max_tokens: Maximum tokens to generate (overrides default if provided)

        Yields:
            Generated tokens as strings

        Raises:
            asyncio.CancelledError: If generation is cancelled via cancel_token
        """
        effective_token_count = min(max_tokens or self.token_count, self.token_count)
        start_time = time.perf_counter()

        self._logger.info(
            "Starting token generation",
            prompt_length=len(prompt),
            max_tokens=effective_token_count,
            req_id=cancel_token.req_id,
            agent_id=cancel_token.agent_id,
        )

        tokens_generated = 0
        yield_count = 0

        try:
            while tokens_generated < effective_token_count:
                # Check cancellation at token boundary
                if cancel_token.cancelled:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    self._logger.info(
                        "Generation cancelled",
                        tokens_generated=tokens_generated,
                        elapsed_ms=elapsed_ms,
                        reason=cancel_token.reason,
                        req_id=cancel_token.req_id,
                        agent_id=cancel_token.agent_id,
                    )
                    raise asyncio.CancelledError(
                        f"Generation cancelled: {cancel_token.reason}"
                    )

                # Generate batch of tokens
                batch_size = min(
                    self.tokens_per_yield, effective_token_count - tokens_generated
                )

                for i in range(batch_size):
                    token = f"token_{tokens_generated + i + 1}"
                    yield token

                tokens_generated += batch_size
                yield_count += 1

                # Yield control for the specified interval (unless we're done)
                if tokens_generated < effective_token_count:
                    await asyncio.sleep(self.actual_yield_interval)

        except asyncio.CancelledError:
            # Re-raise cancellation
            raise
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.error(
                "Generation failed",
                error=str(e),
                tokens_generated=tokens_generated,
                elapsed_ms=elapsed_ms,
                req_id=cancel_token.req_id,
                agent_id=cancel_token.agent_id,
            )
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._logger.info(
            "Generation completed",
            tokens_generated=tokens_generated,
            yield_count=yield_count,
            elapsed_ms=elapsed_ms,
            req_id=cancel_token.req_id,
            agent_id=cancel_token.agent_id,
        )

    async def generate_response(
        self, prompt: str, cancel_token: CancelToken, max_tokens: int | None = None
    ) -> str:
        """Generate complete response with cancellation support.

        Args:
            prompt: Input prompt
            cancel_token: Token for cancellation monitoring
            max_tokens: Maximum tokens to generate

        Returns:
            Complete generated response as a single string

        Raises:
            asyncio.CancelledError: If generation is cancelled
        """
        tokens = []
        async for token in self.generate_tokens(prompt, cancel_token, max_tokens):
            tokens.append(token)

        return " ".join(tokens)

    def estimate_generation_time(self, token_count: int | None = None) -> float:
        """Estimate generation time for given token count.

        Args:
            token_count: Number of tokens (uses default if None)

        Returns:
            Estimated generation time in seconds
        """
        effective_count = token_count or self.token_count
        yields_needed = max(1, effective_count // self.tokens_per_yield)
        return yields_needed * self.actual_yield_interval

    def get_yield_interval(self) -> float:
        """Get the actual yield interval in seconds.

        Returns:
            Yield interval in seconds
        """
        return self.actual_yield_interval

    def configure_timing(
        self,
        generation_time_ms: float | None = None,
        token_count: int | None = None,
        yield_interval_ms: float | None = None,
    ) -> None:
        """Reconfigure timing parameters.

        Args:
            generation_time_ms: New total generation time
            token_count: New token count
            yield_interval_ms: New yield interval
        """
        if generation_time_ms is not None:
            self.generation_time_ms = generation_time_ms
        if token_count is not None:
            self.token_count = token_count
        if yield_interval_ms is not None:
            self.yield_interval_ms = yield_interval_ms

        # Recalculate timing
        self.total_yields = max(1, self.token_count // self.tokens_per_yield)
        self.actual_yield_interval = (
            self.generation_time_ms / self.total_yields / 1000.0
        )

        self._logger.debug(
            "Timing reconfigured",
            generation_time_ms=self.generation_time_ms,
            token_count=self.token_count,
            yield_interval_ms=self.yield_interval_ms,
            actual_yield_interval_s=self.actual_yield_interval,
        )
