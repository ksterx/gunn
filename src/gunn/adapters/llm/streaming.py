"""Streaming LLM adapter with cancellation support.

This module provides a production-ready LLM adapter that supports streaming
token generation with proper cancellation handling and error recovery.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import anthropic
import openai

from gunn.schemas.types import CancelToken
from gunn.utils.errors import LLMGenerationError, LLMTimeoutError
from gunn.utils.telemetry import get_logger


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """Configuration for LLM adapter."""

    provider: LLMProvider
    model: str
    api_key: str | None = None
    api_base: str | None = None
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout_seconds: float = 30.0
    token_yield_interval_ms: float = 25.0  # 20-30ms for responsive cancellation
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class GenerationRequest:
    """Request for LLM generation."""

    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class GenerationResponse:
    """Response from LLM generation."""

    tokens: list[str]
    total_tokens: int
    generation_time_ms: float
    cancelled: bool = False
    cancellation_time_ms: float | None = None
    error: str | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._logger = get_logger(f"gunn.adapters.llm.{config.provider.value}")

    @abstractmethod
    async def generate_stream(
        self,
        request: GenerationRequest,
        cancel_token: CancelToken,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming tokens with cancellation support.

        Args:
            request: Generation request parameters
            cancel_token: Token for cancellation signaling

        Yields:
            Generated tokens as strings

        Raises:
            asyncio.CancelledError: If generation is cancelled
            LLMGenerationError: If generation fails
            LLMTimeoutError: If generation times out
        """
        ...

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connection to LLM provider.

        Returns:
            True if connection is valid, False otherwise
        """
        ...


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout_seconds,
            )
        except ImportError:
            self._logger.warning(
                "OpenAI client not available. Install with: pip install openai"
            )
            self._client = None

    async def generate_stream(
        self,
        request: GenerationRequest,
        cancel_token: CancelToken,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming tokens using OpenAI API."""
        if not self._client:
            raise LLMGenerationError("OpenAI client not initialized")

        start_time = time.perf_counter()
        token_count = 0
        last_yield_time = start_time

        try:
            # Prepare request parameters
            max_tokens = request.max_tokens or self.config.max_tokens
            temperature = request.temperature or self.config.temperature

            self._logger.info(
                "Starting OpenAI generation",
                req_id=cancel_token.req_id,
                agent_id=cancel_token.agent_id,
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Create streaming completion
            stream = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stop=request.stop_sequences,
            )

            async for chunk in stream:
                # Check cancellation at token boundary
                if cancel_token.cancelled:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    self._logger.info(
                        "OpenAI generation cancelled",
                        req_id=cancel_token.req_id,
                        agent_id=cancel_token.agent_id,
                        tokens_generated=token_count,
                        elapsed_ms=elapsed_ms,
                        reason=cancel_token.reason,
                    )
                    raise asyncio.CancelledError(
                        f"Generation cancelled: {cancel_token.reason}"
                    )

                # Extract token from chunk
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    token_count += 1
                    yield token

                    # Yield control periodically for responsive cancellation
                    current_time = time.perf_counter()
                    if (
                        current_time - last_yield_time
                    ) * 1000 >= self.config.token_yield_interval_ms:
                        await asyncio.sleep(0)  # Yield control
                        last_yield_time = current_time

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.error(
                "OpenAI generation failed",
                req_id=cancel_token.req_id,
                agent_id=cancel_token.agent_id,
                error=str(e),
                tokens_generated=token_count,
            )
            raise LLMGenerationError(f"OpenAI generation failed: {e}") from e

        # Log completion
        final_elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._logger.info(
            "OpenAI generation completed",
            req_id=cancel_token.req_id,
            agent_id=cancel_token.agent_id,
            tokens_generated=token_count,
            elapsed_ms=final_elapsed_ms,
        )

    async def validate_connection(self) -> bool:
        """Validate OpenAI connection."""
        if not self._client:
            return False

        try:
            # Test with a minimal request
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return bool(response.choices)
        except Exception as e:
            self._logger.error("OpenAI connection validation failed", error=str(e))
            return False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout_seconds,
            )
        except ImportError:
            self._logger.warning(
                "Anthropic client not available. Install with: pip install anthropic"
            )
            self._client = None

    async def generate_stream(
        self,
        request: GenerationRequest,
        cancel_token: CancelToken,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming tokens using Anthropic API."""
        if not self._client:
            raise LLMGenerationError("Anthropic client not initialized")

        start_time = time.perf_counter()
        token_count = 0
        last_yield_time = start_time

        try:
            max_tokens = request.max_tokens or self.config.max_tokens
            temperature = request.temperature or self.config.temperature

            self._logger.info(
                "Starting Anthropic generation",
                req_id=cancel_token.req_id,
                agent_id=cancel_token.agent_id,
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Create streaming completion
            async with self._client.messages.stream(
                model=self.config.model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=request.stop_sequences,
            ) as stream:
                async for text in stream.text_stream:
                    # Check cancellation at token boundary
                    if cancel_token.cancelled:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        self._logger.info(
                            "Anthropic generation cancelled",
                            req_id=cancel_token.req_id,
                            agent_id=cancel_token.agent_id,
                            tokens_generated=token_count,
                            elapsed_ms=elapsed_ms,
                            reason=cancel_token.reason,
                        )
                        raise asyncio.CancelledError(
                            f"Generation cancelled: {cancel_token.reason}"
                        )

                    token_count += 1
                    yield text

                    # Yield control periodically for responsive cancellation
                    current_time = time.perf_counter()
                    if (
                        current_time - last_yield_time
                    ) * 1000 >= self.config.token_yield_interval_ms:
                        await asyncio.sleep(0)  # Yield control
                        last_yield_time = current_time

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.error(
                "Anthropic generation failed",
                req_id=cancel_token.req_id,
                agent_id=cancel_token.agent_id,
                error=str(e),
                tokens_generated=token_count,
            )
            raise LLMGenerationError(f"Anthropic generation failed: {e}") from e

        # Log completion
        final_elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._logger.info(
            "Anthropic generation completed",
            req_id=cancel_token.req_id,
            agent_id=cancel_token.agent_id,
            tokens_generated=token_count,
            elapsed_ms=final_elapsed_ms,
        )

    async def validate_connection(self) -> bool:
        """Validate Anthropic connection."""
        if not self._client:
            return False

        try:
            # Test with a minimal request
            response = await self._client.messages.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return bool(response.content)
        except Exception as e:
            self._logger.error("Anthropic connection validation failed", error=str(e))
            return False


class StreamingLLMAdapter:
    """Production LLM adapter with streaming support and cancellation.

    This adapter provides a unified interface for multiple LLM providers
    with proper error handling, retry logic, and cancellation support.

    Requirements addressed:
    - 6.1: Stream tokens incrementally
    - 6.2: Monitor for cancellation signals at token boundaries
    - 6.3: Immediately halt token generation within 100ms
    - 11.2: Cancel-to-halt latency ≤ 100ms at token boundaries
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._logger = get_logger("gunn.adapters.llm.streaming")
        self._provider = self._create_provider()

    def _create_provider(self) -> BaseLLMProvider:
        """Create appropriate provider based on configuration."""
        if self.config.provider == LLMProvider.OPENAI:
            return OpenAIProvider(self.config)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return AnthropicProvider(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    async def generate_stream(
        self,
        request: GenerationRequest,
        cancel_token: CancelToken,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming tokens with cancellation support.

        Args:
            request: Generation request parameters
            cancel_token: Token for cancellation signaling

        Yields:
            Generated tokens as strings

        Raises:
            asyncio.CancelledError: If generation is cancelled
            LLMGenerationError: If generation fails after retries
            LLMTimeoutError: If generation times out
        """
        attempt = 0
        last_error = None

        while attempt < self.config.retry_attempts:
            try:
                # Check cancellation before starting attempt
                if cancel_token.cancelled:
                    raise asyncio.CancelledError(
                        f"Generation cancelled before attempt {attempt + 1}: {cancel_token.reason}"
                    )

                self._logger.info(
                    "Starting generation attempt",
                    req_id=cancel_token.req_id,
                    agent_id=cancel_token.agent_id,
                    attempt=attempt + 1,
                    max_attempts=self.config.retry_attempts,
                )

                # Generate with timeout using manual timeout tracking
                start_time = time.perf_counter()

                try:
                    async for token in self._provider.generate_stream(
                        request, cancel_token
                    ):
                        # Check timeout
                        elapsed = time.perf_counter() - start_time
                        if elapsed > self.config.timeout_seconds:
                            raise TimeoutError()

                        yield token

                    # Success - return
                    return

                except TimeoutError as e:
                    raise LLMTimeoutError(
                        f"Generation timed out after {self.config.timeout_seconds}s",
                        timeout_seconds=self.config.timeout_seconds,
                        provider=self.config.provider.value,
                    ) from e

            except asyncio.CancelledError:
                # Don't retry cancellations
                raise
            except (LLMTimeoutError, LLMGenerationError) as e:
                last_error = e
                attempt += 1

                if attempt < self.config.retry_attempts:
                    self._logger.warning(
                        "Generation attempt failed, retrying",
                        req_id=cancel_token.req_id,
                        agent_id=cancel_token.agent_id,
                        attempt=attempt,
                        error=str(e),
                        retry_delay=self.config.retry_delay_seconds,
                    )
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    self._logger.error(
                        "All generation attempts failed",
                        req_id=cancel_token.req_id,
                        agent_id=cancel_token.agent_id,
                        total_attempts=attempt,
                        final_error=str(e),
                    )
                    raise e

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        else:
            raise LLMGenerationError("Generation failed with unknown error")

    async def generate_with_timing(
        self,
        request: GenerationRequest,
        cancel_token: CancelToken,
    ) -> GenerationResponse:
        """Generate tokens and return comprehensive response with timing.

        Args:
            request: Generation request parameters
            cancel_token: Token for cancellation signaling

        Returns:
            GenerationResponse with tokens, timing, and metadata

        Raises:
            asyncio.CancelledError: If generation is cancelled
            LLMGenerationError: If generation fails
            LLMTimeoutError: If generation times out
        """
        tokens = []
        start_time = time.perf_counter()
        cancellation_time_ms = None
        error_message = None

        try:
            async for token in self.generate_stream(request, cancel_token):
                tokens.append(token)

        except asyncio.CancelledError:
            cancellation_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(
                "Generation cancelled",
                req_id=cancel_token.req_id,
                agent_id=cancel_token.agent_id,
                tokens_generated=len(tokens),
                cancellation_time_ms=cancellation_time_ms,
                reason=cancel_token.reason,
            )
            raise
        except Exception as e:
            error_message = str(e)
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            raise
        else:
            generation_time_ms = (time.perf_counter() - start_time) * 1000

        return GenerationResponse(
            tokens=tokens,
            total_tokens=len(tokens),
            generation_time_ms=generation_time_ms,
            cancelled=cancellation_time_ms is not None,
            cancellation_time_ms=cancellation_time_ms,
            error=error_message,
        )

    async def validate_connection(self) -> bool:
        """Validate connection to LLM provider.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            return await self._provider.validate_connection()
        except Exception as e:
            self._logger.error("Connection validation failed", error=str(e))
            return False

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check.

        Returns:
            Dictionary with health check results
        """
        start_time = time.perf_counter()

        try:
            connection_valid = await self.validate_connection()
            response_time_ms = (time.perf_counter() - start_time) * 1000

            return {
                "status": "healthy" if connection_valid else "unhealthy",
                "provider": self.config.provider.value,
                "model": self.config.model,
                "connection_valid": connection_valid,
                "response_time_ms": response_time_ms,
                "config": {
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "timeout_seconds": self.config.timeout_seconds,
                    "token_yield_interval_ms": self.config.token_yield_interval_ms,
                    "retry_attempts": self.config.retry_attempts,
                },
            }
        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            return {
                "status": "error",
                "provider": self.config.provider.value,
                "model": self.config.model,
                "connection_valid": False,
                "response_time_ms": response_time_ms,
                "error": str(e),
            }
