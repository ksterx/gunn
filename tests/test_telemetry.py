"""Unit tests for telemetry utilities."""

import time

import structlog

from gunn.utils.telemetry import (
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


class TestLoggingSetup:
    """Test logging setup functionality."""

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        setup_logging("INFO")
        logger = structlog.get_logger("test")
        # Should not raise an exception
        logger.info("Test message")

    def test_setup_logging_debug(self):
        """Test debug logging setup."""
        setup_logging("DEBUG")
        logger = structlog.get_logger("test")
        # Should not raise an exception
        logger.debug("Debug message")


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
