"""Unit tests for telemetry utilities."""

import asyncio
import time
<<<<<<< HEAD

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
=======
from unittest.mock import patch

import pytest
import structlog

from gunn.utils.telemetry import (
    OperationTimer,
    PIIRedactionProcessor,
    create_logger,
    get_monotonic_time,
    get_wall_time,
    redact_dict,
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7
    redact_pii,
    setup_logging,
)


class TestPIIRedaction:
    """Test PII redaction functionality."""

<<<<<<< HEAD
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
=======
    def test_redact_email(self):
        """Test email redaction."""
        text = "Contact me at john.doe@example.com for more info"
        result = redact_pii(text)
        assert "john.doe@example.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_redact_phone(self):
        """Test phone number redaction."""
        text = "Call me at 555-123-4567 or 555.987.6543"
        result = redact_pii(text)
        assert "555-123-4567" not in result
        assert "555.987.6543" not in result
        assert result.count("[REDACTED_PHONE]") == 2

    def test_redact_token(self):
        """Test token redaction."""
        text = "Use token abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
        result = redact_pii(text)
        assert "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz" not in result
        assert "[REDACTED_TOKEN]" in result

    def test_redact_multiple_pii_types(self):
        """Test redacting multiple PII types in one text."""
        text = "Email: user@test.com, Phone: 555-1234, Token: abcdef123456789012345678"
        result = redact_pii(text)
        assert "user@test.com" not in result
        assert "555-1234" not in result
        assert "abcdef123456789012345678" not in result
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "[REDACTED_TOKEN]" in result

    def test_redact_non_string(self):
        """Test that non-string inputs are returned unchanged."""
        assert redact_pii(123) == 123
        assert redact_pii(None) is None
        assert redact_pii([1, 2, 3]) == [1, 2, 3]

    def test_redact_dict_simple(self):
        """Test dictionary PII redaction."""
        data = {
            "email": "test@example.com",
            "phone": "555-1234",
            "name": "John Doe",
            "count": 42,
        }
        result = redact_dict(data)
        assert result["email"] == "[REDACTED_EMAIL]"
        assert result["phone"] == "[REDACTED_PHONE]"
        assert result["name"] == "John Doe"  # No PII pattern match
        assert result["count"] == 42  # Non-string unchanged

    def test_redact_dict_nested(self):
        """Test nested dictionary PII redaction."""
        data = {
            "user": {
                "contact": {
                    "email": "nested@test.com",
                    "phone": "555-9876",
                },
                "id": 123,
            },
            "tokens": ["short", "verylongtoken123456789012345"],
        }
        result = redact_dict(data)
        assert result["user"]["contact"]["email"] == "[REDACTED_EMAIL]"
        assert result["user"]["contact"]["phone"] == "[REDACTED_PHONE]"
        assert result["user"]["id"] == 123
        assert result["tokens"][0] == "short"  # Too short to be a token
        assert result["tokens"][1] == "[REDACTED_TOKEN]"

    def test_redact_dict_non_dict(self):
        """Test that non-dict inputs are returned unchanged."""
        assert redact_dict("string") == "string"
        assert redact_dict(123) == 123
        assert redact_dict(None) is None


class TestPIIRedactionProcessor:
    """Test the structlog PII redaction processor."""

    def test_processor_redacts_event(self):
        """Test that the processor redacts PII from event messages."""
        processor = PIIRedactionProcessor()
        event_dict = {
            "event": "User login: user@example.com",
            "level": "info",
        }
        result = processor(None, "info", event_dict)
        assert result["event"] == "User login: [REDACTED_EMAIL]"
        assert result["level"] == "info"

    def test_processor_redacts_fields(self):
        """Test that the processor redacts PII from other fields."""
        processor = PIIRedactionProcessor()
        event_dict = {
            "event": "User action",
            "user_email": "test@domain.com",
            "phone_number": "555-1234",
            "metadata": {"contact": "admin@site.com"},
        }
        result = processor(None, "info", event_dict)
        assert result["user_email"] == "[REDACTED_EMAIL]"
        assert result["phone_number"] == "[REDACTED_PHONE]"
        assert result["metadata"]["contact"] == "[REDACTED_EMAIL]"


class TestLoggingSetup:
    """Test logging setup functionality."""

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        setup_logging("test-service", "INFO")
        logger = structlog.get_logger("test")
        # Should not raise an exception
        logger.info("Test message")

    def test_setup_logging_debug(self):
        """Test debug logging setup."""
        setup_logging("test-service", "DEBUG")
        logger = structlog.get_logger("test")
        # Should not raise an exception
        logger.debug("Debug message")


class TestOperationTimer:
    """Test operation timing functionality."""

    @pytest.mark.asyncio
    async def test_operation_timer_success(self):
        """Test successful operation timing."""
        with patch("gunn.utils.telemetry.OPERATION_COUNTER") as mock_counter, patch(
            "gunn.utils.telemetry.OPERATION_DURATION"
        ) as mock_duration:
            mock_counter.labels.return_value.inc = lambda: None
            mock_duration.labels.return_value.observe = lambda x: None

            with OperationTimer("test_op", "agent_1"):
                await asyncio.sleep(0.01)  # Small delay

            # Verify metrics were recorded
            mock_counter.labels.assert_called_with(
                operation="test_op", status="success", agent_id="agent_1"
            )
            mock_duration.labels.assert_called_with(
                operation="test_op", agent_id="agent_1"
            )

    @pytest.mark.asyncio
    async def test_operation_timer_error(self):
        """Test operation timing with error."""
        with patch("gunn.utils.telemetry.OPERATION_COUNTER") as mock_counter, patch(
            "gunn.utils.telemetry.OPERATION_DURATION"
        ) as mock_duration:
            mock_counter.labels.return_value.inc = lambda: None
            mock_duration.labels.return_value.observe = lambda x: None

            with pytest.raises(ValueError):
                with OperationTimer("test_op", "agent_1"):
                    raise ValueError("Test error")

            # Verify error status was recorded
            mock_counter.labels.assert_called_with(
                operation="test_op", status="error", agent_id="agent_1"
            )


class TestTimingUtilities:
    """Test timing utility functions."""

    @pytest.mark.asyncio
    async def test_get_monotonic_time_in_loop(self):
        """Test monotonic time within event loop."""
        time1 = get_monotonic_time()
        await asyncio.sleep(0.001)
        time2 = get_monotonic_time()
        assert time2 > time1

    def test_get_monotonic_time_outside_loop(self):
        """Test monotonic time outside event loop."""
        # Should not raise an exception and should return a float
        result = get_monotonic_time()
        assert isinstance(result, float)
        assert result > 0

    def test_get_wall_time(self):
        """Test wall clock time."""
        wall_time = get_wall_time()
        assert isinstance(wall_time, float)
        assert wall_time > 1600000000  # After 2020


class TestLoggerCreation:
    """Test logger creation utilities."""

    def test_create_logger_basic(self):
        """Test basic logger creation."""
        logger = create_logger("test_logger")
        assert logger is not None
        # Should not raise an exception
        logger.info("Test message")

    def test_create_logger_with_context(self):
        """Test logger creation with context."""
        logger = create_logger("test_logger", agent_id="agent_1", req_id="req_123")
        assert logger is not None
        # Should not raise an exception
        logger.info("Test message with context")


class TestPerformanceImpact:
    """Test performance impact of telemetry operations."""

    def test_redaction_performance(self):
        """Test that PII redaction doesn't significantly impact performance."""
        text = "This is a test message with user@example.com and 555-1234"

        # Time the redaction operation
        start_time = time.perf_counter()
        for _ in range(1000):
            redact_pii(text)
        end_time = time.perf_counter()

        # Should complete 1000 redactions in reasonable time (< 1 second)
        duration = end_time - start_time
        assert duration < 1.0, f"PII redaction took too long: {duration:.3f}s"

    def test_dict_redaction_performance(self):
        """Test that dictionary redaction doesn't significantly impact performance."""
        data = {
            "user": {
                "email": "test@example.com",
                "phone": "555-1234",
                "metadata": {
                    "tokens": ["token123456789012345678", "short"],
                    "count": 42,
                },
            }
        }

        # Time the redaction operation
        start_time = time.perf_counter()
        for _ in range(100):
            redact_dict(data)
        end_time = time.perf_counter()

        # Should complete 100 redactions in reasonable time (< 1 second)
        duration = end_time - start_time
        assert duration < 1.0, f"Dict redaction took too long: {duration:.3f}s"
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7
