"""Unit tests for telemetry utilities."""

import asyncio
import time
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
    redact_pii,
    setup_logging,
)


class TestPIIRedaction:
    """Test PII redaction functionality."""

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
