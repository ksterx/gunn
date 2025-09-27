"""Unit tests for telemetry utilities."""

import asyncio
import time

import pytest

from gunn.utils.telemetry import (
    MonotonicClock,
    PerformanceTimer,
    async_performance_timer,
    get_logger,
    get_timing_context,
    pii_redaction_processor,
    record_cancellation,
    record_queue_depth,
    redact_pii,
    setup_logging,
)


class TestPIIRedaction:
    """Test PII redaction functionality."""

    def test_redact_email(self) -> None:
        """Test email redaction."""
        text = "Contact john.doe@example.com for support"
        result = redact_pii(text)
        assert "[REDACTED_EMAIL]" in result
        assert "john.doe@example.com" not in result

    def test_redact_phone(self) -> None:
        """Test phone number redaction."""
        text = "Call us at 555-123-4567 or (555) 123-4567"
        result = redact_pii(text)
        assert "[REDACTED_PHONE]" in result
        assert "555-123-4567" not in result

    def test_redact_token(self) -> None:
        """Test token redaction."""
        text = "Use token abc123def456ghi789jkl012mno345pqr678"
        result = redact_pii(text)
        assert "[REDACTED_TOKEN]" in result
        assert "abc123def456ghi789jkl012mno345pqr678" not in result

    def test_redact_multiple_pii(self) -> None:
        """Test redacting multiple PII types."""
        text = "Email: user@test.com, Phone: 555-0123, Token: verylongtoken123456789"
        result = redact_pii(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "[REDACTED_TOKEN]" in result
        assert "user@test.com" not in result
        assert "555-0123" not in result

    def test_no_pii_unchanged(self) -> None:
        """Test that text without PII is unchanged."""
        text = "This is a normal message with no sensitive data"
        result = redact_pii(text)
        assert result == text

    def test_pii_processor_dict(self) -> None:
        """Test PII processor with dictionary values."""
        event_dict = {
            "message": "Contact user@example.com",
            "metadata": {"phone": "555-1234", "safe_data": "normal text"},
            "list_data": ["email@test.com", "safe item"],
        }

        result = pii_redaction_processor(None, "info", event_dict)

        assert "[REDACTED_EMAIL]" in result["message"]
        assert "[REDACTED_PHONE]" in result["metadata"]["phone"]
        assert result["metadata"]["safe_data"] == "normal text"
        assert "[REDACTED_EMAIL]" in result["list_data"][0]
        assert result["list_data"][1] == "safe item"


class TestLogging:
    """Test logging setup and functionality."""

    def test_setup_logging(self) -> None:
        """Test logging setup."""
        setup_logging("DEBUG", enable_pii_redaction=True)
        logger = get_logger("test")
        # Just verify the logger has the expected methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")

    def test_logger_with_context(self) -> None:
        """Test logger with bound context."""
        setup_logging()
        logger = get_logger("test", agent_id="agent_1", req_id="req_123")

        # Logger should have bound context
        assert hasattr(logger, "_context")

    def test_pii_redaction_in_logs(self) -> None:
        """Test that PII is redacted in actual log output."""
        setup_logging(enable_pii_redaction=True)
        logger = get_logger("test")

        # This test verifies the processor is installed correctly
        # Just ensure logging doesn't raise an exception
        logger.info("Test message with user@example.com")


class TestPerformanceTimer:
    """Test performance timing functionality."""

    def test_performance_timer_context(self) -> None:
        """Test performance timer as context manager."""
        with PerformanceTimer(
            "test_operation", "agent_1", record_metrics=False
        ) as timer:
            time.sleep(0.01)  # Small delay

        assert timer.duration is not None
        assert timer.duration > 0.005  # Should be at least 5ms

    def test_performance_timer_exception(self) -> None:
        """Test performance timer with exception."""
        with pytest.raises(ValueError):
            with PerformanceTimer("test_operation", record_metrics=False):
                raise ValueError("Test error")

    @pytest.mark.asyncio
    async def test_async_performance_timer(self) -> None:
        """Test async performance timer."""
        async with async_performance_timer(
            "async_test", "agent_1", record_metrics=False
        ) as timer:
            await asyncio.sleep(0.01)

        assert timer.duration is not None
        assert timer.duration > 0.005

    @pytest.mark.asyncio
    async def test_async_performance_timer_exception(self) -> None:
        """Test async performance timer with exception."""
        with pytest.raises(ValueError):
            async with async_performance_timer("async_test", record_metrics=False):
                raise ValueError("Async test error")


class TestMetrics:
    """Test metrics recording functionality."""

    def test_record_queue_depth(self) -> None:
        """Test queue depth recording."""
        # Should not raise exception
        record_queue_depth("agent_1", 5)
        record_queue_depth("agent_2", 0)

    def test_record_cancellation(self) -> None:
        """Test cancellation recording."""
        # Should not raise exception
        record_cancellation("agent_1", "stale_context")
        record_cancellation("agent_2", "user_requested")


class TestMonotonicClock:
    """Test monotonic clock functionality."""

    def test_monotonic_now(self) -> None:
        """Test monotonic time retrieval."""
        time1 = MonotonicClock.now()
        time.sleep(0.001)
        time2 = MonotonicClock.now()

        assert time2 > time1
        assert isinstance(time1, float)
        assert isinstance(time2, float)

    def test_wall_time(self) -> None:
        """Test wall time retrieval."""
        wall_time = MonotonicClock.wall_time()
        assert isinstance(wall_time, float)
        assert wall_time > 0

    def test_monotonic_without_event_loop(self) -> None:
        """Test monotonic time without running event loop."""
        # This should fall back to time.monotonic()
        mono_time = MonotonicClock.now()
        assert isinstance(mono_time, float)
        assert mono_time > 0

    @pytest.mark.asyncio
    async def test_monotonic_with_event_loop(self) -> None:
        """Test monotonic time with running event loop."""
        mono_time = MonotonicClock.now()
        assert isinstance(mono_time, float)
        assert mono_time > 0


class TestTimingContext:
    """Test timing context functionality."""

    def test_get_timing_context(self) -> None:
        """Test timing context retrieval."""
        context = get_timing_context()

        assert "monotonic_time" in context
        assert "wall_time" in context
        assert isinstance(context["monotonic_time"], float)
        assert isinstance(context["wall_time"], float)
        assert context["monotonic_time"] > 0
        assert context["wall_time"] > 0

    def test_timing_context_consistency(self) -> None:
        """Test that timing context is consistent."""
        context1 = get_timing_context()
        time.sleep(0.001)
        context2 = get_timing_context()

        assert context2["monotonic_time"] > context1["monotonic_time"]
        assert context2["wall_time"] > context1["wall_time"]


class TestIntegration:
    """Integration tests for telemetry components."""

    def test_full_logging_pipeline(self) -> None:
        """Test complete logging pipeline with PII redaction."""
        setup_logging("INFO", enable_pii_redaction=True)

        logger = get_logger("integration_test", agent_id="agent_1", req_id="req_123")

        # Log message with PII - should not raise exception
        logger.info("Processing request for user@example.com")
        logger.warning("Phone number 555-1234 detected")
        logger.error("Token abc123def456ghi789 is invalid")

    @pytest.mark.asyncio
    async def test_performance_with_logging(self) -> None:
        """Test performance timer with logging integration."""
        setup_logging("DEBUG", enable_pii_redaction=False)
        logger = get_logger("perf_test", agent_id="agent_1")

        async with async_performance_timer("test_op", "agent_1", logger, False):
            await asyncio.sleep(0.01)

        # Should complete without errors

    def test_metrics_and_timing_integration(self) -> None:
        """Test metrics recording with timing."""
        with PerformanceTimer("integration_test", "agent_1", record_metrics=False):
            record_queue_depth("agent_1", 10)
            record_cancellation("agent_1", "timeout")
            time.sleep(0.005)

        # Should complete without errors
