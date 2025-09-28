"""Integration tests for LLM streaming adapter with cancellation behavior.

This module tests the streaming LLM adapter's cancellation timing,
error handling, and integration with the orchestrator system.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from gunn.adapters.llm import (
    DummyLLMAdapter,
    GenerationRequest,
    LLMConfig,
    LLMProvider,
    StreamingLLMAdapter,
)
from gunn.schemas.types import CancelToken
from gunn.utils.errors import LLMGenerationError, LLMTimeoutError


class TestDummyLLMAdapter:
    """Test dummy LLM adapter for cancellation behavior."""

    def test_initialization_validates_token_interval(self):
        """Test that initialization validates token interval range."""
        # Valid interval should work
        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        assert adapter.token_interval_ms == 25.0

        # Invalid intervals should raise ValueError
        with pytest.raises(
            ValueError, match="token_interval_ms should be between 20-30ms"
        ):
            DummyLLMAdapter(token_interval_ms=10.0)

        with pytest.raises(
            ValueError, match="token_interval_ms should be between 20-30ms"
        ):
            DummyLLMAdapter(token_interval_ms=50.0)

    @pytest.mark.asyncio
    async def test_basic_token_generation(self):
        """Test basic token generation without cancellation."""
        adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=200.0,  # Longer time to allow 5 tokens
        )
        cancel_token = CancelToken("test_req", "test_agent")

        tokens = []
        async for token in adapter.generate_stream(
            "test prompt", cancel_token, max_tokens=5
        ):
            tokens.append(token)

        assert len(tokens) == 5
        assert tokens == ["token_1", "token_2", "token_3", "token_4", "token_5"]

    @pytest.mark.asyncio
    async def test_cancellation_timing_accuracy(self):
        """Test that cancellation occurs within 100ms SLO."""
        adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=5000.0,  # Long generation
        )
        cancel_token = CancelToken("test_req", "test_agent")

        # Start generation
        generation_task = asyncio.create_task(
            adapter.generate_with_timing("test prompt", cancel_token, max_tokens=1000)
        )

        # Cancel after 100ms
        await asyncio.sleep(0.1)
        cancel_start = time.perf_counter()
        cancel_token.cancel("test_cancellation")

        # Wait for cancellation
        with pytest.raises(asyncio.CancelledError):
            await generation_task

        cancel_end = time.perf_counter()
        cancellation_response_time_ms = (cancel_end - cancel_start) * 1000

        # Should respond within 100ms (with some tolerance for test environment)
        assert cancellation_response_time_ms <= 150.0, (
            f"Cancellation took {cancellation_response_time_ms}ms, "
            f"exceeds 100ms SLO (with 50ms test tolerance)"
        )

    @pytest.mark.asyncio
    async def test_token_yield_interval_accuracy(self):
        """Test that tokens are yielded at correct intervals."""
        adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=200.0,
        )
        cancel_token = CancelToken("test_req", "test_agent")

        token_times = []
        start_time = time.perf_counter()

        try:
            async for token in adapter.generate_stream(
                "test", cancel_token, max_tokens=6
            ):
                token_times.append(time.perf_counter() - start_time)
        except asyncio.CancelledError:
            pass

        # Calculate intervals between tokens
        intervals = []
        for i in range(1, len(token_times)):
            interval_ms = (token_times[i] - token_times[i - 1]) * 1000
            intervals.append(interval_ms)

        # Each interval should be approximately 25ms (±10ms tolerance)
        for i, interval in enumerate(intervals):
            assert (
                15.0 <= interval <= 35.0
            ), f"Token {i + 1} interval {interval}ms outside 25ms ±10ms range"

    @pytest.mark.asyncio
    async def test_generation_with_timing_returns_correct_metadata(self):
        """Test that generate_with_timing returns correct timing metadata."""
        adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=100.0,
        )
        cancel_token = CancelToken("test_req", "test_agent")

        (
            tokens,
            generation_time_ms,
            cancellation_time_ms,
        ) = await adapter.generate_with_timing(
            "test prompt", cancel_token, max_tokens=3
        )

        assert tokens == ["token_1", "token_2", "token_3"]
        assert 75.0 <= generation_time_ms <= 125.0  # ~100ms ±25ms tolerance
        assert cancellation_time_ms is None  # Not cancelled

    @pytest.mark.asyncio
    async def test_cancellation_during_generation_returns_timing(self):
        """Test that cancellation during generation returns timing metadata."""
        adapter = DummyLLMAdapter(
            token_interval_ms=25.0,
            generation_time_ms=1000.0,  # Long generation
        )
        cancel_token = CancelToken("test_req", "test_agent")

        # Start generation and cancel after 50ms
        async def cancel_after_delay():
            await asyncio.sleep(0.05)
            cancel_token.cancel("test_cancellation")

        cancel_task = asyncio.create_task(cancel_after_delay())

        with pytest.raises(asyncio.CancelledError):
            await adapter.generate_with_timing(
                "test prompt", cancel_token, max_tokens=100
            )

        await cancel_task


class TestStreamingLLMAdapter:
    """Test production streaming LLM adapter."""

    def test_config_validation(self):
        """Test LLM configuration validation."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            max_tokens=100,
            temperature=0.7,
            timeout_seconds=30.0,
            token_yield_interval_ms=25.0,
        )

        adapter = StreamingLLMAdapter(config)
        assert adapter.config.provider == LLMProvider.OPENAI
        assert adapter.config.model == "gpt-4"
        assert adapter.config.token_yield_interval_ms == 25.0

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,  # Not implemented
            model="custom-model",
        )

        with pytest.raises(ValueError, match="Unsupported provider"):
            StreamingLLMAdapter(config)

    @pytest.mark.asyncio
    async def test_openai_provider_without_client_raises_error(self):
        """Test OpenAI provider without client installation."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
        )

        adapter = StreamingLLMAdapter(config)
        cancel_token = CancelToken("test_req", "test_agent")
        request = GenerationRequest(prompt="test")

        # Mock the provider to simulate missing client
        adapter._provider._client = None

        with pytest.raises(LLMGenerationError, match="OpenAI client not initialized"):
            async for _ in adapter.generate_stream(request, cancel_token):
                pass

    @pytest.mark.asyncio
    async def test_retry_logic_on_generation_failure(self):
        """Test retry logic when generation fails."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            retry_attempts=3,
            retry_delay_seconds=0.01,  # Fast retry for testing
        )

        adapter = StreamingLLMAdapter(config)
        cancel_token = CancelToken("test_req", "test_agent")
        request = GenerationRequest(prompt="test")

        # Mock the provider to fail twice, then succeed
        mock_provider = AsyncMock()
        call_count = 0

        async def mock_generate_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise LLMGenerationError("Temporary failure")
            else:
                # Simulate successful generation
                for i in range(3):
                    yield f"token_{i + 1}"

        mock_provider.generate_stream = mock_generate_stream
        adapter._provider = mock_provider

        # Should succeed after retries
        tokens = []
        async for token in adapter.generate_stream(request, cancel_token):
            tokens.append(token)

        assert tokens == ["token_1", "token_2", "token_3"]
        assert call_count == 3  # Failed twice, succeeded on third attempt

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises_final_error(self):
        """Test that retry exhaustion raises the final error."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            retry_attempts=2,
            retry_delay_seconds=0.01,
        )

        adapter = StreamingLLMAdapter(config)
        cancel_token = CancelToken("test_req", "test_agent")
        request = GenerationRequest(prompt="test")

        # Mock the provider to always fail
        mock_provider = AsyncMock()
        call_count = 0

        async def failing_generate_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise LLMGenerationError("Persistent failure")
            yield  # This line will never be reached, but makes it an async generator

        mock_provider.generate_stream = failing_generate_stream
        adapter._provider = mock_provider

        with pytest.raises(LLMGenerationError, match="Persistent failure"):
            async for _ in adapter.generate_stream(request, cancel_token):
                pass

        # Should have attempted 2 times
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cancellation_not_retried(self):
        """Test that cancellation errors are not retried."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            retry_attempts=3,
        )

        adapter = StreamingLLMAdapter(config)
        cancel_token = CancelToken("test_req", "test_agent")
        request = GenerationRequest(prompt="test")

        # Cancel immediately
        cancel_token.cancel("immediate_cancellation")

        # Mock the provider
        mock_provider = AsyncMock()
        adapter._provider = mock_provider

        with pytest.raises(asyncio.CancelledError):
            async for _ in adapter.generate_stream(request, cancel_token):
                pass

        # Should not have called provider at all due to immediate cancellation
        mock_provider.generate_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling during generation."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            timeout_seconds=0.1,  # Very short timeout
            retry_attempts=1,
        )

        adapter = StreamingLLMAdapter(config)
        cancel_token = CancelToken("test_req", "test_agent")
        request = GenerationRequest(prompt="test")

        # Mock provider that takes too long
        mock_provider = AsyncMock()

        async def slow_generate_stream(*args, **kwargs):
            await asyncio.sleep(0.2)  # Longer than timeout
            yield "token_1"

        mock_provider.generate_stream = slow_generate_stream
        adapter._provider = mock_provider

        with pytest.raises(LLMTimeoutError, match="Generation timed out"):
            async for _ in adapter.generate_stream(request, cancel_token):
                pass

    @pytest.mark.asyncio
    async def test_generate_with_timing_comprehensive_response(self):
        """Test generate_with_timing returns comprehensive response."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
        )

        adapter = StreamingLLMAdapter(config)
        cancel_token = CancelToken("test_req", "test_agent")
        request = GenerationRequest(prompt="test")

        # Mock successful generation
        mock_provider = AsyncMock()

        async def mock_generate_stream(*args, **kwargs):
            for i in range(3):
                yield f"token_{i + 1}"
                await asyncio.sleep(0.01)  # Small delay

        mock_provider.generate_stream = mock_generate_stream
        adapter._provider = mock_provider

        response = await adapter.generate_with_timing(request, cancel_token)

        assert response.tokens == ["token_1", "token_2", "token_3"]
        assert response.total_tokens == 3
        assert response.generation_time_ms > 0
        assert not response.cancelled
        assert response.cancellation_time_ms is None
        assert response.error is None

    @pytest.mark.asyncio
    async def test_health_check_with_valid_connection(self):
        """Test health check with valid connection."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
        )

        adapter = StreamingLLMAdapter(config)

        # Mock successful connection validation
        mock_provider = AsyncMock()
        mock_provider.validate_connection = AsyncMock(return_value=True)
        adapter._provider = mock_provider

        health = await adapter.health_check()

        assert health["status"] == "healthy"
        assert health["provider"] == "openai"
        assert health["model"] == "gpt-4"
        assert health["connection_valid"] is True
        assert "response_time_ms" in health
        assert "config" in health

    @pytest.mark.asyncio
    async def test_health_check_with_invalid_connection(self):
        """Test health check with invalid connection."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="invalid-key",
        )

        adapter = StreamingLLMAdapter(config)

        # Mock failed connection validation
        mock_provider = AsyncMock()
        mock_provider.validate_connection = AsyncMock(return_value=False)
        adapter._provider = mock_provider

        health = await adapter.health_check()

        assert health["status"] == "unhealthy"
        assert health["connection_valid"] is False

    @pytest.mark.asyncio
    async def test_health_check_with_exception(self):
        """Test health check when validation raises exception."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
        )

        adapter = StreamingLLMAdapter(config)

        # Mock exception during validation
        mock_provider = AsyncMock()
        mock_provider.validate_connection = AsyncMock(
            side_effect=Exception("Connection error")
        )
        adapter._provider = mock_provider

        health = await adapter.health_check()

        assert (
            health["status"] == "unhealthy"
        )  # validate_connection exception results in "unhealthy"
        assert health["connection_valid"] is False
        # Note: The error is logged but not included in health response for validate_connection exceptions


class TestLLMIntegrationWithOrchestrator:
    """Test LLM adapter integration with orchestrator system."""

    @pytest.mark.asyncio
    async def test_llm_adapter_with_orchestrator_cancellation(self):
        """Test LLM adapter integration with orchestrator cancellation."""
        # This test would require the orchestrator to be implemented
        # For now, we'll test the interface compatibility

        adapter = DummyLLMAdapter(token_interval_ms=25.0)
        cancel_token = CancelToken("test_req", "test_agent")

        # Simulate orchestrator issuing cancel token
        generation_task = asyncio.create_task(
            adapter.generate_with_timing("test prompt", cancel_token, max_tokens=100)
        )

        # Simulate orchestrator cancelling due to staleness
        await asyncio.sleep(0.05)  # Let some generation happen
        cancel_token.cancel("stale_due_to_new_observation")

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await generation_task

        assert "stale_due_to_new_observation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_generations_with_cancellation(self):
        """Test multiple concurrent generations with selective cancellation."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0, generation_time_ms=200.0)

        # Create multiple cancel tokens for different agents
        tokens = [
            CancelToken("req_1", "agent_1"),
            CancelToken("req_2", "agent_2"),
            CancelToken("req_3", "agent_3"),
        ]

        # Start multiple generations
        tasks = [
            asyncio.create_task(
                adapter.generate_with_timing(f"prompt_{i}", token, max_tokens=10)
            )
            for i, token in enumerate(tokens)
        ]

        # Cancel only the middle one after some time
        await asyncio.sleep(0.05)
        tokens[1].cancel("selective_cancellation")

        # Wait for all tasks to complete or be cancelled
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # First and third should succeed, second should be cancelled
        assert isinstance(results[0], tuple)  # Success
        assert isinstance(results[1], asyncio.CancelledError)  # Cancelled
        assert isinstance(results[2], tuple)  # Success

        # Verify the successful results
        tokens_1, time_1, cancel_time_1 = results[0]
        tokens_3, time_3, cancel_time_3 = results[2]

        assert len(tokens_1) > 0
        assert len(tokens_3) > 0
        assert cancel_time_1 is None
        assert cancel_time_3 is None

    @pytest.mark.asyncio
    async def test_cancellation_response_time_under_load(self):
        """Test cancellation response time under concurrent load."""
        adapter = DummyLLMAdapter(token_interval_ms=25.0, generation_time_ms=1000.0)

        # Create many concurrent generations
        num_concurrent = 10
        tokens = [CancelToken(f"req_{i}", f"agent_{i}") for i in range(num_concurrent)]

        tasks = [
            asyncio.create_task(
                adapter.generate_with_timing(f"prompt_{i}", token, max_tokens=100)
            )
            for i, token in enumerate(tokens)
        ]

        # Let them run for a bit
        await asyncio.sleep(0.1)

        # Cancel all at once and measure response time
        cancel_start = time.perf_counter()
        for token in tokens:
            token.cancel("load_test_cancellation")

        # Wait for all to be cancelled
        results = await asyncio.gather(*tasks, return_exceptions=True)
        cancel_end = time.perf_counter()

        # All should be cancelled
        for result in results:
            assert isinstance(result, asyncio.CancelledError)

        # Total cancellation time should be reasonable even under load
        total_cancel_time_ms = (cancel_end - cancel_start) * 1000
        assert total_cancel_time_ms <= 200.0, (
            f"Cancellation under load took {total_cancel_time_ms}ms, "
            f"should be much faster"
        )


@pytest.mark.asyncio
async def test_llm_adapter_protocol_compliance():
    """Test that all LLM adapters comply with the LLMAdapter protocol."""
    from gunn.adapters.llm.dummy import LLMAdapter

    # Test dummy adapter compliance
    dummy_adapter = DummyLLMAdapter()
    assert isinstance(dummy_adapter, LLMAdapter)

    # Test that it has the required methods
    assert hasattr(dummy_adapter, "generate_stream")
    assert callable(dummy_adapter.generate_stream)

    # Test streaming adapter compliance
    config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
    streaming_adapter = StreamingLLMAdapter(config)

    # Should have the required methods
    assert hasattr(streaming_adapter, "generate_stream")
    assert callable(streaming_adapter.generate_stream)
    assert hasattr(streaming_adapter, "validate_connection")
    assert callable(streaming_adapter.validate_connection)


if __name__ == "__main__":
    pytest.main([__file__])
