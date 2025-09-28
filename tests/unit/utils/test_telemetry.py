"""Unit tests for telemetry utilities."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
import structlog
from prometheus_client import REGISTRY

from gunn.utils.telemetry import (
    BandwidthMonitor,
    PerformanceTimer,
    SystemMonitor,
    async_performance_timer,
    get_tracer,
    log_operation,
    record_conflict,
    record_intent_throughput,
    record_observation_delivery_latency,
    redact_pii,
    setup_logging,
    setup_tracing,
    update_active_agents_count,
    update_global_seq,
    update_view_seq,
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


class TestEnhancedLogging:
    """Test enhanced logging functionality."""

    def test_log_operation_basic(self):
        """Test basic operation logging."""
        logger = MagicMock()

        log_operation(
            logger,
            operation="test_operation",
            status="success",
            agent_id="agent_1",
            req_id="req_123",
            latency_ms=50.0,
        )

        # Verify logger was called with correct data
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert call_args[0][0] == "Operation completed"

        # Check that required fields are present
        kwargs = call_args[1]
        assert kwargs["operation"] == "test_operation"
        assert kwargs["status"] == "success"
        assert kwargs["agent_id"] == "agent_1"
        assert kwargs["req_id"] == "req_123"
        assert kwargs["latency_ms"] == 50.0

    def test_log_operation_error(self):
        """Test error operation logging."""
        logger = MagicMock()

        log_operation(
            logger,
            operation="test_operation",
            status="error",
            error_message="Something went wrong",
        )

        # Should use error level for error status
        logger.error.assert_called_once()

    def test_log_operation_optional_fields(self):
        """Test logging with optional fields."""
        logger = MagicMock()

        log_operation(
            logger,
            operation="test_operation",
            global_seq=100,
            view_seq=50,
        )

        # Check that optional fields are included
        call_args = logger.info.call_args
        kwargs = call_args[1]
        assert kwargs["global_seq"] == 100
        assert kwargs["view_seq"] == 50


class TestMetricsRecording:
    """Test metrics recording functions."""

    def test_record_intent_throughput(self):
        """Test intent throughput recording."""
        # Clear any existing metrics
        REGISTRY._collector_to_names.clear()
        REGISTRY._names_to_collectors.clear()

        record_intent_throughput("agent_1", "Speak", "success")
        record_intent_throughput("agent_1", "Move", "error")

        # Metrics should be recorded (we can't easily verify values without
        # accessing internal Prometheus state, but we can verify no exceptions)

    def test_record_conflict(self):
        """Test conflict recording."""
        record_conflict("agent_1", "staleness")
        record_conflict("agent_2", "validation")

    def test_record_observation_delivery_latency(self):
        """Test observation delivery latency recording."""
        record_observation_delivery_latency("agent_1", 0.025)  # 25ms
        record_observation_delivery_latency("agent_2", 0.100)  # 100ms

    def test_update_sequence_metrics(self):
        """Test sequence number metric updates."""
        update_global_seq(1000)
        update_view_seq("agent_1", 500)
        update_view_seq("agent_2", 750)
        update_active_agents_count(2)


class TestSystemMonitor:
    """Test system monitoring functionality."""

    def test_system_monitor_creation(self):
        """Test system monitor can be created."""
        monitor = SystemMonitor()
        assert monitor is not None

    @patch("psutil.Process")
    def test_record_memory_usage(self, mock_process_class):
        """Test memory usage recording."""
        # Mock process and memory info
        mock_process = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        monitor = SystemMonitor()
        memory_bytes = monitor.record_memory_usage("test_component")

        assert memory_bytes == 1024 * 1024 * 100
        mock_process.memory_info.assert_called_once()

    @patch("psutil.Process")
    def test_record_cpu_usage(self, mock_process_class):
        """Test CPU usage recording."""
        # Mock process and CPU info
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 25.5
        mock_process_class.return_value = mock_process

        monitor = SystemMonitor()
        cpu_percent = monitor.record_cpu_usage("test_component")

        assert cpu_percent == 25.5
        mock_process.cpu_percent.assert_called_once_with(interval=None)

    @patch("psutil.Process")
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    def test_get_system_stats(
        self, mock_cpu_count, mock_cpu_percent, mock_virtual_memory, mock_process_class
    ):
        """Test comprehensive system stats."""
        # Mock all the psutil calls
        mock_process = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 50
        mock_memory_info.vms = 1024 * 1024 * 100
        mock_process.memory_info.return_value = mock_memory_info

        mock_cpu_times = MagicMock()
        mock_cpu_times.user = 10.5
        mock_cpu_times.system = 5.2
        mock_process.cpu_times.return_value = mock_cpu_times
        mock_process.cpu_percent.return_value = 15.0
        mock_process.num_threads.return_value = 8
        mock_process.num_fds.return_value = 25

        mock_process_class.return_value = mock_process

        mock_system_memory = MagicMock()
        mock_system_memory.total = 1024 * 1024 * 1024 * 8  # 8GB
        mock_system_memory.available = 1024 * 1024 * 1024 * 4  # 4GB
        mock_system_memory.used = 1024 * 1024 * 1024 * 4  # 4GB
        mock_system_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_system_memory

        mock_cpu_percent.return_value = 20.0
        mock_cpu_count.return_value = 4

        monitor = SystemMonitor()
        stats = monitor.get_system_stats()

        assert "process" in stats
        assert "system" in stats
        assert stats["process"]["memory_rss_bytes"] == 1024 * 1024 * 50
        assert stats["process"]["cpu_user_seconds"] == 10.5
        assert stats["system"]["memory_total_bytes"] == 1024 * 1024 * 1024 * 8
        assert stats["system"]["cpu_count"] == 4


class TestBandwidthMonitor:
    """Test bandwidth monitoring functionality."""

    def test_bandwidth_monitor_creation(self):
        """Test bandwidth monitor can be created."""
        monitor = BandwidthMonitor()
        assert monitor is not None

    def test_record_patch_bandwidth(self):
        """Test patch bandwidth recording."""
        monitor = BandwidthMonitor()

        # Test normal patch
        monitor.record_patch_bandwidth("agent_1", 1024, 5, False)

        # Test fallback patch
        monitor.record_patch_bandwidth("agent_2", 10240, 1000, True)

    def test_record_data_transfer(self):
        """Test general data transfer recording."""
        monitor = BandwidthMonitor()

        monitor.record_data_transfer("inbound", "web_adapter", 2048)
        monitor.record_data_transfer("outbound", "llm_adapter", 4096)


class TestPerformanceTimer:
    """Test performance timer functionality."""

    def test_performance_timer_basic(self):
        """Test basic performance timer usage."""
        with PerformanceTimer("test_operation", agent_id="agent_1") as timer:
            time.sleep(0.01)  # 10ms

        assert timer.duration is not None
        assert timer.duration >= 0.01
        assert timer.operation == "test_operation"
        assert timer.agent_id == "agent_1"

    def test_performance_timer_with_context(self):
        """Test performance timer with full context."""
        with PerformanceTimer(
            "test_operation",
            agent_id="agent_1",
            req_id="req_123",
            global_seq=100,
            view_seq=50,
        ) as timer:
            time.sleep(0.005)  # 5ms

        assert timer.req_id == "req_123"
        assert timer.global_seq == 100
        assert timer.view_seq == 50

    def test_performance_timer_exception(self):
        """Test performance timer with exception."""
        with pytest.raises(ValueError):
            with PerformanceTimer("test_operation") as timer:
                raise ValueError("Test error")

        assert timer.duration is not None

    @pytest.mark.asyncio
    async def test_async_performance_timer(self):
        """Test async performance timer."""
        async with async_performance_timer(
            "async_test_operation",
            agent_id="agent_1",
            req_id="req_456",
        ) as timer:
            await asyncio.sleep(0.01)  # 10ms

        assert timer.duration is not None
        assert timer.duration >= 0.01

    @pytest.mark.asyncio
    async def test_async_performance_timer_exception(self):
        """Test async performance timer with exception."""
        with pytest.raises(RuntimeError):
            async with async_performance_timer("async_test_operation") as timer:
                raise RuntimeError("Async test error")

        assert timer.duration is not None


class TestTracingSetup:
    """Test OpenTelemetry tracing setup."""

    def test_setup_tracing_basic(self):
        """Test basic tracing setup."""
        setup_tracing("test_service", enable_fastapi_instrumentation=False)

        # Should not raise an exception
        tracer = get_tracer("test_module")
        assert tracer is not None

    def test_setup_tracing_with_endpoint(self):
        """Test tracing setup with OTLP endpoint."""
        # This would normally connect to a real OTLP endpoint
        # For testing, we just verify it doesn't crash
        setup_tracing(
            "test_service",
            otlp_endpoint="http://localhost:4317",
            enable_fastapi_instrumentation=False,
        )

        tracer = get_tracer("test_module")
        assert tracer is not None

    def test_get_tracer(self):
        """Test tracer retrieval."""
        tracer1 = get_tracer("module1")
        tracer2 = get_tracer("module2")

        assert tracer1 is not None
        assert tracer2 is not None
        # Different modules should get different tracer instances
        assert tracer1 != tracer2


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

    def test_metrics_recording_performance(self):
        """Test that metrics recording doesn't significantly impact performance."""
        start_time = time.perf_counter()

        for i in range(1000):
            record_intent_throughput(f"agent_{i % 10}", "Speak", "success")
            record_conflict(f"agent_{i % 10}", "staleness")
            update_global_seq(i)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Should complete 3000 metric operations in reasonable time (< 1 second)
        assert duration < 1.0, f"Metrics recording took too long: {duration:.3f}s"

    def test_logging_performance(self):
        """Test that enhanced logging doesn't significantly impact performance."""
        logger = MagicMock()

        start_time = time.perf_counter()

        for i in range(1000):
            log_operation(
                logger,
                "test_operation",
                status="success",
                global_seq=i,
                view_seq=i // 2,
                agent_id=f"agent_{i % 10}",
                req_id=f"req_{i}",
                latency_ms=float(i % 100),
            )

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Should complete 1000 log operations in reasonable time (< 1 second)
        assert duration < 1.0, f"Enhanced logging took too long: {duration:.3f}s"
