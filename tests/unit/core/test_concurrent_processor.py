"""Tests for concurrent intent processing."""

import asyncio
import time

import pytest

from gunn.core.concurrent_processor import (
    BatchResult,
    ConcurrentIntentProcessor,
    ConcurrentProcessingConfig,
    ProcessingMode,
)
from gunn.schemas.types import Effect, Intent


class TestConcurrentProcessingConfig:
    """Test concurrent processing configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConcurrentProcessingConfig()

        assert config.max_concurrent_intents == 100
        assert config.default_timeout == 30.0
        assert config.enable_deterministic_mode is True
        assert config.processing_mode == ProcessingMode.SEQUENTIAL
        assert config.batch_size_threshold == 5
        assert config.semaphore_size == 50

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConcurrentProcessingConfig(
            max_concurrent_intents=200,
            default_timeout=60.0,
            enable_deterministic_mode=False,
            processing_mode=ProcessingMode.CONCURRENT,
            batch_size_threshold=10,
            semaphore_size=100,
        )

        assert config.max_concurrent_intents == 200
        assert config.default_timeout == 60.0
        assert config.enable_deterministic_mode is False
        assert config.processing_mode == ProcessingMode.CONCURRENT
        assert config.batch_size_threshold == 10
        assert config.semaphore_size == 100


class TestConcurrentIntentProcessor:
    """Test concurrent intent processor."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ConcurrentProcessingConfig(
            max_concurrent_intents=10, default_timeout=5.0, semaphore_size=5
        )

    @pytest.fixture
    def processor(self, config):
        """Create test processor."""
        return ConcurrentIntentProcessor(config)

    @pytest.fixture
    def mock_intent_processor(self):
        """Create mock intent processor."""

        async def process_intent(intent: Intent) -> list[Effect]:
            # Simulate processing time
            await asyncio.sleep(0.01)

            # Return mock effect
            import uuid

            return [
                {
                    "uuid": str(uuid.uuid4()),
                    "kind": intent["kind"],
                    "payload": intent["payload"],
                    "source_id": intent["agent_id"],
                    "schema_version": "1.0.0",
                    "global_seq": 1,
                    "sim_time": 1.0,
                }
            ]

        return process_intent

    @pytest.fixture
    def sample_intents(self):
        """Create sample intents for testing."""
        return [
            {
                "req_id": "req_1",
                "agent_id": "agent_1",
                "kind": "Move",
                "payload": {"target_position": [10, 10]},
                "schema_version": "1.0.0",
            },
            {
                "req_id": "req_2",
                "agent_id": "agent_2",
                "kind": "Attack",
                "payload": {"target_id": "agent_3"},
                "schema_version": "1.0.0",
            },
            {
                "req_id": "req_3",
                "agent_id": "agent_3",
                "kind": "Heal",
                "payload": {"target_id": "agent_3"},
                "schema_version": "1.0.0",
            },
        ]

    def test_processor_initialization(self, config):
        """Test processor initialization."""
        processor = ConcurrentIntentProcessor(config)

        assert processor.config == config
        assert processor._intent_processor is None

    def test_set_intent_processor(self, processor, mock_intent_processor):
        """Test setting intent processor."""
        processor.set_intent_processor(mock_intent_processor)
        assert processor._intent_processor == mock_intent_processor

    async def test_process_empty_batch(self, processor, mock_intent_processor):
        """Test processing empty batch."""
        processor.set_intent_processor(mock_intent_processor)

        result = await processor.process_batch([])

        assert isinstance(result, BatchResult)
        assert result.effects == []
        assert result.errors == {}
        assert result.processing_time == 0.0
        assert result.metadata["mode"] == "empty"
        assert result.metadata["intent_count"] == 0

    async def test_process_batch_without_processor(self, processor, sample_intents):
        """Test processing batch without setting processor."""
        with pytest.raises(ValueError, match="Intent processor not set"):
            await processor.process_batch(sample_intents)

    async def test_sequential_processing(
        self, processor, mock_intent_processor, sample_intents
    ):
        """Test sequential processing mode."""
        processor.set_intent_processor(mock_intent_processor)

        start_time = time.time()
        result = await processor.process_batch(
            sample_intents, mode=ProcessingMode.SEQUENTIAL
        )
        end_time = time.time()

        assert isinstance(result, BatchResult)
        assert len(result.effects) == 3  # One effect per intent
        assert len(result.errors) == 0
        assert result.processing_time > 0
        assert result.metadata["mode"] == "sequential"
        assert result.metadata["intent_count"] == 3
        assert result.metadata["success_count"] == 3
        assert result.metadata["error_count"] == 0

        # Sequential should take at least 3 * 0.01 seconds
        assert (end_time - start_time) >= 0.025

    async def test_concurrent_processing(
        self, processor, mock_intent_processor, sample_intents
    ):
        """Test concurrent processing mode."""
        processor.set_intent_processor(mock_intent_processor)

        start_time = time.time()
        result = await processor.process_batch(
            sample_intents, mode=ProcessingMode.CONCURRENT
        )
        end_time = time.time()

        assert isinstance(result, BatchResult)
        assert len(result.effects) == 3
        assert len(result.errors) == 0
        assert result.processing_time > 0
        assert result.metadata["mode"] == "concurrent"
        assert result.metadata["intent_count"] == 3

        # Concurrent should be faster than sequential
        assert (end_time - start_time) < 0.025

    async def test_deterministic_concurrent_processing(
        self, processor, mock_intent_processor, sample_intents
    ):
        """Test deterministic concurrent processing mode."""
        processor.set_intent_processor(mock_intent_processor)

        result = await processor.process_batch(
            sample_intents, mode=ProcessingMode.DETERMINISTIC_CONCURRENT
        )

        assert isinstance(result, BatchResult)
        assert len(result.effects) == 3
        assert len(result.errors) == 0
        assert result.metadata["mode"] == "deterministic_concurrent"
        assert result.metadata["deterministic_ordering"] is True

        # Effects should be sorted by source_id
        source_ids = [effect["source_id"] for effect in result.effects]
        assert source_ids == sorted(source_ids)

    async def test_error_handling(self, processor, sample_intents):
        """Test error handling in concurrent processing."""

        async def failing_processor(intent: Intent) -> list[Effect]:
            if intent["agent_id"] == "agent_2":
                raise ValueError("Simulated processing error")

            import uuid

            return [
                {
                    "uuid": str(uuid.uuid4()),
                    "kind": intent["kind"],
                    "payload": intent["payload"],
                    "source_id": intent["agent_id"],
                    "schema_version": "1.0.0",
                    "global_seq": 1,
                    "sim_time": 1.0,
                }
            ]

        processor.set_intent_processor(failing_processor)

        result = await processor.process_batch(
            sample_intents, mode=ProcessingMode.CONCURRENT
        )

        assert len(result.effects) == 2  # Two successful intents
        assert len(result.errors) == 1  # One failed intent
        assert "req_2" in result.errors
        assert isinstance(result.errors["req_2"], ValueError)

    async def test_timeout_handling(self, processor, sample_intents):
        """Test timeout handling in concurrent processing."""

        async def slow_processor(intent: Intent) -> list[Effect]:
            await asyncio.sleep(1.0)  # Longer than timeout
            import uuid

            return [
                {
                    "uuid": str(uuid.uuid4()),
                    "kind": intent["kind"],
                    "source_id": intent["agent_id"],
                    "payload": {},
                    "schema_version": "1.0.0",
                    "global_seq": 1,
                    "sim_time": 1.0,
                }
            ]

        processor.set_intent_processor(slow_processor)

        with pytest.raises(asyncio.TimeoutError):
            await processor.process_batch(
                sample_intents, mode=ProcessingMode.CONCURRENT, timeout=0.1
            )

    def test_should_use_concurrent_processing(self, processor):
        """Test concurrent processing recommendation."""
        # Below threshold
        assert not processor.should_use_concurrent_processing(3)

        # At threshold
        assert processor.should_use_concurrent_processing(5)

        # Above threshold
        assert processor.should_use_concurrent_processing(10)

    def test_get_recommended_mode(self, processor):
        """Test recommended processing mode."""
        # Small batch - sequential
        mode = processor.get_recommended_mode(3)
        assert mode == ProcessingMode.SEQUENTIAL

        # Large batch - deterministic concurrent (default)
        mode = processor.get_recommended_mode(10)
        assert mode == ProcessingMode.DETERMINISTIC_CONCURRENT

        # Large batch with deterministic mode disabled
        processor.config.enable_deterministic_mode = False
        mode = processor.get_recommended_mode(10)
        assert mode == ProcessingMode.CONCURRENT

    def test_get_processing_stats(self, processor):
        """Test processing statistics."""
        stats = processor.get_processing_stats()

        assert "config" in stats
        assert "semaphore_available" in stats
        assert "semaphore_total" in stats
        assert "semaphore_utilization" in stats

        assert stats["semaphore_total"] == processor.config.semaphore_size
        assert 0.0 <= stats["semaphore_utilization"] <= 1.0

    async def test_semaphore_limiting(self, processor, sample_intents):
        """Test that semaphore limits concurrent operations."""
        # Create more intents than semaphore allows
        many_intents = []
        for i in range(10):
            many_intents.append(
                {
                    "req_id": f"req_{i}",
                    "agent_id": f"agent_{i}",
                    "kind": "Move",
                    "payload": {"target_position": [i, i]},
                    "schema_version": "1.0.0",
                }
            )

        concurrent_count = 0
        max_concurrent = 0

        async def tracking_processor(intent: Intent) -> list[Effect]:
            nonlocal concurrent_count, max_concurrent

            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.1)  # Simulate work

            concurrent_count -= 1

            import uuid

            return [
                {
                    "uuid": str(uuid.uuid4()),
                    "kind": intent["kind"],
                    "source_id": intent["agent_id"],
                    "payload": {},
                    "schema_version": "1.0.0",
                    "global_seq": 1,
                    "sim_time": 1.0,
                }
            ]

        processor.set_intent_processor(tracking_processor)

        result = await processor.process_batch(
            many_intents, mode=ProcessingMode.CONCURRENT
        )

        assert len(result.effects) == 10
        assert len(result.errors) == 0
        # Should not exceed semaphore size
        assert max_concurrent <= processor.config.semaphore_size


class TestProcessingMode:
    """Test ProcessingMode enum."""

    def test_processing_mode_values(self):
        """Test processing mode enum values."""
        assert ProcessingMode.SEQUENTIAL == "sequential"
        assert ProcessingMode.CONCURRENT == "concurrent"
        assert ProcessingMode.DETERMINISTIC_CONCURRENT == "deterministic_concurrent"

    def test_processing_mode_iteration(self):
        """Test iterating over processing modes."""
        modes = list(ProcessingMode)
        assert len(modes) == 3
        assert ProcessingMode.SEQUENTIAL in modes
        assert ProcessingMode.CONCURRENT in modes
        assert ProcessingMode.DETERMINISTIC_CONCURRENT in modes


class TestBatchResult:
    """Test BatchResult model."""

    def test_batch_result_creation(self):
        """Test creating BatchResult."""
        import uuid

        effects = [
            {
                "uuid": str(uuid.uuid4()),
                "kind": "Move",
                "source_id": "agent_1",
                "payload": {},
                "schema_version": "1.0.0",
                "global_seq": 1,
                "sim_time": 1.0,
            }
        ]
        errors = {"req_1": ValueError("test error")}
        processing_time = 1.5
        metadata = {"mode": "concurrent", "intent_count": 2}

        result = BatchResult(
            effects=effects,
            errors=errors,
            processing_time=processing_time,
            metadata=metadata,
        )

        assert result.effects == effects
        assert result.errors == errors
        assert result.processing_time == processing_time
        assert result.metadata == metadata

    def test_batch_result_defaults(self):
        """Test BatchResult with default values."""
        result = BatchResult(effects=[], errors={}, processing_time=0.0, metadata={})

        assert result.effects == []
        assert result.errors == {}
        assert result.processing_time == 0.0
        assert result.metadata == {}
