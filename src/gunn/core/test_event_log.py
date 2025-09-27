"""Unit tests for EventLog implementation."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from gunn.core.event_log import EventLog
from gunn.schemas.types import Effect


class TestEventLog:
    """Test EventLog functionality."""

    @pytest.fixture
    def sample_effect(self) -> Effect:
        """Create a sample effect for testing."""
        return {
            "uuid": "test-uuid-123",
            "kind": "Move",
            "payload": {"x": 10, "y": 20},
            "global_seq": 1,
            "sim_time": 123.456,
            "source_id": "agent_001",
            "schema_version": "1.0.0",
        }

    @pytest.fixture
    def event_log(self) -> EventLog:
        """Create an EventLog instance for testing."""
        return EventLog(world_seed=12345)

    @pytest.mark.asyncio
    async def test_append_effect(self, event_log: EventLog, sample_effect: Effect):
        """Test appending an effect to the log."""
        global_seq = await event_log.append(sample_effect, "req_123")

        assert global_seq == sample_effect["global_seq"]
        assert event_log.entry_count == 1
        assert event_log.get_latest_seq() == sample_effect["global_seq"]

    @pytest.mark.asyncio
    async def test_append_invalid_effect(self, event_log: EventLog):
        """Test appending an invalid effect raises ValueError."""
        invalid_effect = {
            "uuid": "test-uuid",
            "kind": "Move",
            # Missing required fields
        }

        with pytest.raises(ValueError, match="Effect must contain"):
            await event_log.append(invalid_effect)

    @pytest.mark.asyncio
    async def test_get_entries_since(self, event_log: EventLog):
        """Test retrieving entries since a sequence number."""
        # Add multiple effects
        effects = []
        for i in range(5):
            effect = {
                "uuid": f"uuid-{i}",
                "kind": "Move",
                "payload": {"x": i, "y": i},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": f"agent_{i}",
                "schema_version": "1.0.0",
            }
            effects.append(effect)
            await event_log.append(effect)

        # Get entries since seq 2
        entries = await event_log.get_entries_since(2)
        assert len(entries) == 2  # seq 3 and 4
        assert entries[0].global_seq == 3
        assert entries[1].global_seq == 4

    @pytest.mark.asyncio
    async def test_get_entries_range(self, event_log: EventLog):
        """Test retrieving entries in a specific range."""
        # Add multiple effects
        for i in range(10):
            effect = {
                "uuid": f"uuid-{i}",
                "kind": "Move",
                "payload": {"x": i, "y": i},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": f"agent_{i}",
                "schema_version": "1.0.0",
            }
            await event_log.append(effect)

        # Test range query
        entries = await event_log.get_entries_range(3, 7)
        assert len(entries) == 5  # seq 3, 4, 5, 6, 7
        assert entries[0].global_seq == 3
        assert entries[-1].global_seq == 7

        # Test range with no end
        entries = await event_log.get_entries_range(8)
        assert len(entries) == 2  # seq 8, 9
        assert entries[0].global_seq == 8
        assert entries[-1].global_seq == 9

    @pytest.mark.asyncio
    async def test_validate_integrity_valid(self, event_log: EventLog):
        """Test integrity validation with valid log."""
        # Add multiple effects
        for i in range(3):
            effect = {
                "uuid": f"uuid-{i}",
                "kind": "Move",
                "payload": {"x": i, "y": i},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": f"agent_{i}",
                "schema_version": "1.0.0",
            }
            await event_log.append(effect)

        # Validate integrity
        is_valid = await event_log.validate_integrity()
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_integrity_corrupted(self, event_log: EventLog):
        """Test integrity validation with corrupted log."""
        # Add an effect
        effect = {
            "uuid": "uuid-1",
            "kind": "Move",
            "payload": {"x": 1, "y": 1},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "agent_1",
            "schema_version": "1.0.0",
        }
        await event_log.append(effect)

        # Corrupt the checksum
        event_log._entries[0].checksum = "corrupted_checksum"

        # Validate integrity
        is_valid = await event_log.validate_integrity()
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_concurrent_appends(self, event_log: EventLog):
        """Test concurrent append operations."""

        async def append_effect(i: int):
            effect = {
                "uuid": f"uuid-{i}",
                "kind": "Move",
                "payload": {"x": i, "y": i},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": f"agent_{i}",
                "schema_version": "1.0.0",
            }
            return await event_log.append(effect, f"req_{i}")

        # Run concurrent appends
        tasks = [append_effect(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all appends succeeded
        assert len(results) == 10
        assert event_log.entry_count == 10

        # Verify integrity is maintained
        is_valid = await event_log.validate_integrity()
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_save_and_load_file(self, event_log: EventLog):
        """Test saving and loading log from file."""
        # Add some effects
        for i in range(3):
            effect = {
                "uuid": f"uuid-{i}",
                "kind": "Move",
                "payload": {"x": i, "y": i},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": f"agent_{i}",
                "schema_version": "1.0.0",
            }
            await event_log.append(effect)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            await event_log.save_to_file(temp_path)

            # Create new log and load from file
            new_log = EventLog()
            await new_log.load_from_file(temp_path)

            # Verify loaded data
            assert new_log.entry_count == event_log.entry_count
            assert new_log.world_seed == event_log.world_seed
            assert new_log.get_latest_seq() == event_log.get_latest_seq()

            # Verify integrity
            is_valid = await new_log.validate_integrity()
            assert is_valid is True

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises FileNotFoundError."""
        event_log = EventLog()
        nonexistent_path = Path("/nonexistent/file.json")

        with pytest.raises(FileNotFoundError):
            await event_log.load_from_file(nonexistent_path)

    @pytest.mark.asyncio
    async def test_load_invalid_file(self):
        """Test loading from invalid file raises ValueError."""
        # Create invalid JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = Path(f.name)

        try:
            event_log = EventLog()
            with pytest.raises(ValueError, match="Invalid log file format"):
                await event_log.load_from_file(temp_path)
        finally:
            temp_path.unlink()

    def test_world_seed_property(self):
        """Test world seed property."""
        seed = 98765
        event_log = EventLog(world_seed=seed)
        assert event_log.world_seed == seed

    def test_empty_log_properties(self):
        """Test properties of empty log."""
        event_log = EventLog()
        assert event_log.entry_count == 0
        assert event_log.get_latest_seq() == -1
        assert len(event_log) == 0

    def test_repr(self):
        """Test string representation."""
        event_log = EventLog(world_seed=12345)
        repr_str = repr(event_log)
        assert "EventLog" in repr_str
        assert "entries=0" in repr_str
        assert "latest_seq=-1" in repr_str
        assert "world_seed=12345" in repr_str


class TestEventLogPerformance:
    """Test EventLog performance characteristics."""

    @pytest.mark.asyncio
    async def test_large_log_performance(self):
        """Test performance with large number of entries."""
        import time

        event_log = EventLog()

        # Add many effects
        start_time = time.perf_counter()
        for i in range(1000):
            effect = {
                "uuid": f"uuid-{i}",
                "kind": "Move",
                "payload": {"x": i, "y": i},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": f"agent_{i % 10}",  # 10 different agents
                "schema_version": "1.0.0",
            }
            await event_log.append(effect)

        append_time = time.perf_counter() - start_time

        # Test retrieval performance
        start_time = time.perf_counter()
        entries = await event_log.get_entries_range(0, 999)
        retrieval_time = time.perf_counter() - start_time

        # Test integrity validation performance
        start_time = time.perf_counter()
        is_valid = await event_log.validate_integrity()
        validation_time = time.perf_counter() - start_time

        # Verify results
        assert len(entries) == 1000
        assert is_valid is True

        # Performance assertions (generous limits for CI)
        assert append_time < 5.0, f"Append took too long: {append_time:.3f}s"
        assert retrieval_time < 1.0, f"Retrieval took too long: {retrieval_time:.3f}s"
        assert (
            validation_time < 2.0
        ), f"Validation took too long: {validation_time:.3f}s"
