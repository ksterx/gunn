"""Unit tests for EventLog implementation."""

import asyncio
<<<<<<< HEAD
import time

import pytest

from gunn.core.event_log import EventLog, EventLogEntry
from gunn.schemas.types import Effect
from gunn.utils.hashing import chain_checksum


class TestEventLog:
    """Test EventLog basic functionality."""

    @pytest.fixture
    def event_log(self) -> EventLog:
        """Create a fresh EventLog for testing."""
        return EventLog("test_world")
=======
import tempfile
from pathlib import Path

import pytest

from gunn.core.event_log import EventLog
from gunn.schemas.types import Effect


class TestEventLog:
    """Test EventLog functionality."""
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7

    @pytest.fixture
    def sample_effect(self) -> Effect:
        """Create a sample effect for testing."""
<<<<<<< HEAD
        return Effect(
            uuid="test-uuid-123",
            kind="TestEffect",
            payload={"data": "test"},
            global_seq=0,  # Will be set by EventLog
            sim_time=1.0,
            source_id="test_source",
            schema_version="1.0.0",
        )

    @pytest.mark.asyncio
    async def test_append_single_effect(
        self, event_log: EventLog, sample_effect: Effect
    ) -> None:
        """Test appending a single effect."""
        global_seq = await event_log.append(sample_effect)

        assert global_seq == 1
        assert event_log.get_entry_count() == 1
        assert event_log.get_latest_seq() == 1

        entries = event_log.get_all_entries()
        assert len(entries) == 1
        assert entries[0].global_seq == 1
        assert entries[0].effect["uuid"] == "test-uuid-123"

    @pytest.mark.asyncio
    async def test_append_multiple_effects(self, event_log: EventLog) -> None:
        """Test appending multiple effects in sequence."""
        effects = []
        for i in range(5):
            effect = Effect(
                uuid=f"uuid-{i}",
                kind="TestEffect",
                payload={"index": i},
                global_seq=0,
                sim_time=float(i),
                source_id="test_source",
                schema_version="1.0.0",
            )
            effects.append(effect)

        # Append all effects
        sequences = []
        for effect in effects:
            seq = await event_log.append(effect)
            sequences.append(seq)

        # Verify sequences are monotonic
        assert sequences == [1, 2, 3, 4, 5]
        assert event_log.get_entry_count() == 5
        assert event_log.get_latest_seq() == 5

    @pytest.mark.asyncio
    async def test_append_with_req_id(
        self, event_log: EventLog, sample_effect: Effect
    ) -> None:
        """Test appending with request ID."""
        req_id = "req-123"
        global_seq = await event_log.append(sample_effect, req_id=req_id)

        entries = event_log.find_entries_by_req_id(req_id)
        assert len(entries) == 1
        assert entries[0].req_id == req_id
        assert entries[0].global_seq == global_seq

    @pytest.mark.asyncio
    async def test_append_invalid_effect(self, event_log: EventLog) -> None:
        """Test appending invalid effects raises errors."""
        # Missing uuid
        with pytest.raises(ValueError, match="Effect must have uuid field"):
            await event_log.append(
                Effect(
                    uuid="",
                    kind="TestEffect",
                    payload={},
                    global_seq=0,
                    sim_time=1.0,
                    source_id="test",
                    schema_version="1.0.0",
                )
            )

        # Missing kind
        with pytest.raises(ValueError, match="Effect must have kind field"):
            await event_log.append(
                Effect(
                    uuid="test-uuid",
                    kind="",
                    payload={},
                    global_seq=0,
                    sim_time=1.0,
                    source_id="test",
                    schema_version="1.0.0",
                )
            )

    @pytest.mark.asyncio
    async def test_concurrent_appends(self, event_log: EventLog) -> None:
        """Test concurrent append operations are thread-safe."""

        async def append_effect(index: int) -> int:
            effect = Effect(
                uuid=f"concurrent-{index}",
                kind="ConcurrentTest",
                payload={"index": index},
                global_seq=0,
                sim_time=float(index),
                source_id=f"source-{index}",
                schema_version="1.0.0",
            )
            return await event_log.append(effect)

        # Run 10 concurrent appends
        tasks = [append_effect(i) for i in range(10)]
        sequences = await asyncio.gather(*tasks)

        # All sequences should be unique and in range 1-10
        assert len(set(sequences)) == 10
        assert min(sequences) == 1
        assert max(sequences) == 10
        assert event_log.get_entry_count() == 10


class TestEventLogRetrieval:
    """Test EventLog retrieval methods."""

    @pytest.fixture
    async def populated_log(self) -> EventLog:
        """Create an EventLog with test data."""
        log = EventLog("test_world")

        # Add 10 test effects
        for i in range(10):
            effect = Effect(
                uuid=f"uuid-{i}",
                kind="TestEffect" if i % 2 == 0 else "OtherEffect",
                payload={"index": i, "data": f"test-{i}"},
                global_seq=0,
                sim_time=float(i * 10),  # 0, 10, 20, ...
                source_id="source_a" if i < 5 else "source_b",
                schema_version="1.0.0",
            )
            await log.append(effect, req_id=f"req-{i}")

        return log

    def test_get_entries_since(self, populated_log: EventLog) -> None:
        """Test retrieving entries since a sequence number."""
        # Get entries since sequence 5
        entries = populated_log.get_entries_since(5)

        assert len(entries) == 5  # Sequences 6, 7, 8, 9, 10
        assert all(entry.global_seq > 5 for entry in entries)
        assert entries[0].global_seq == 6
        assert entries[-1].global_seq == 10

    def test_get_entries_since_empty(self, populated_log: EventLog) -> None:
        """Test getting entries since latest sequence returns empty."""
        latest_seq = populated_log.get_latest_seq()
        entries = populated_log.get_entries_since(latest_seq)
        assert len(entries) == 0

    def test_find_entry_by_uuid(self, populated_log: EventLog) -> None:
        """Test finding entry by UUID."""
        entry = populated_log.find_entry_by_uuid("uuid-5")
        assert entry is not None
        assert entry.effect["uuid"] == "uuid-5"
        assert entry.effect["payload"]["index"] == 5

        # Non-existent UUID
        entry = populated_log.find_entry_by_uuid("non-existent")
        assert entry is None

    def test_find_entries_by_req_id(self, populated_log: EventLog) -> None:
        """Test finding entries by request ID."""
        entries = populated_log.find_entries_by_req_id("req-3")
        assert len(entries) == 1
        assert entries[0].req_id == "req-3"
        assert entries[0].effect["payload"]["index"] == 3

    def test_get_entries_by_source(self, populated_log: EventLog) -> None:
        """Test getting entries by source ID."""
        # Source A has first 5 entries (indices 0-4)
        entries_a = populated_log.get_entries_by_source("source_a")
        assert len(entries_a) == 5
        assert all(entry.effect["source_id"] == "source_a" for entry in entries_a)

        # Source B has last 5 entries (indices 5-9)
        entries_b = populated_log.get_entries_by_source("source_b")
        assert len(entries_b) == 5
        assert all(entry.effect["source_id"] == "source_b" for entry in entries_b)

    def test_get_entries_in_time_range(self, populated_log: EventLog) -> None:
        """Test getting entries within time range."""
        # Get entries with sim_time between 20 and 60 (inclusive)
        entries = populated_log.get_entries_in_time_range(20.0, 60.0, use_sim_time=True)

        # Should include indices 2, 3, 4, 5, 6 (sim_times 20, 30, 40, 50, 60)
        assert len(entries) == 5
        assert all(20.0 <= entry.sim_time <= 60.0 for entry in entries)

    def test_get_entries_time_range_no_bounds(self, populated_log: EventLog) -> None:
        """Test time range with no bounds returns all entries."""
        entries = populated_log.get_entries_in_time_range()
        assert len(entries) == 10


class TestEventLogIntegrity:
    """Test EventLog integrity validation."""

    @pytest.fixture
    async def log_with_integrity(self) -> tuple[EventLog, list[Effect]]:
        """Create log with known integrity."""
        log = EventLog("integrity_test")

        effects = []
        for i in range(5):
            effect = Effect(
                uuid=f"integrity-{i}",
                kind="IntegrityTest",
                payload={"value": i * 2},
                global_seq=0,
                sim_time=float(i),
                source_id="integrity_source",
                schema_version="1.0.0",
            )
            effects.append(effect)
            await log.append(effect)

        return log, effects

    def test_validate_integrity_valid_log(
        self, log_with_integrity: tuple[EventLog, list[Effect]]
    ) -> None:
        """Test integrity validation on valid log."""
        log, _ = log_with_integrity

        result = log.validate_integrity()

        assert result["valid"] is True
        assert len(result["corrupted_entries"]) == 0
        assert len(result["missing_sequences"]) == 0
        assert result["total_entries"] == 5

    def test_validate_integrity_corrupted_checksum(
        self, log_with_integrity: tuple[EventLog, list[Effect]]
    ) -> None:
        """Test integrity validation detects corrupted checksums."""
        log, _ = log_with_integrity

        # Corrupt the checksum of the third entry
        log._entries[2].checksum = "corrupted_checksum"

        result = log.validate_integrity()

        assert result["valid"] is False
        assert 2 in result["corrupted_entries"]  # Index 2 is corrupted
        # Subsequent entries will also be corrupted due to hash chain
        assert len(result["corrupted_entries"]) >= 1

    def test_validate_integrity_empty_log(self) -> None:
        """Test integrity validation on empty log."""
        log = EventLog("empty_test")

        result = log.validate_integrity()

        assert result["valid"] is True
        assert len(result["corrupted_entries"]) == 0
        assert len(result["missing_sequences"]) == 0
        assert result["total_entries"] == 0

    def test_hash_chain_consistency(
        self, log_with_integrity: tuple[EventLog, list[Effect]]
    ) -> None:
        """Test that hash chain is consistent."""
        log, _effects = log_with_integrity

        entries = log.get_all_entries()

        # Verify hash chain
        prev_checksum = None
        for entry in entries:
            expected_checksum = chain_checksum(dict(entry.effect), prev_checksum)
            assert entry.checksum == expected_checksum
            prev_checksum = entry.checksum


class TestEventLogStats:
    """Test EventLog statistics and metadata."""

    @pytest.fixture
    async def stats_log(self) -> EventLog:
        """Create log for statistics testing."""
        log = EventLog("stats_test")

        # Add effects with varying times
        for i in range(3):
            effect = Effect(
                uuid=f"stats-{i}",
                kind="StatsTest",
                payload={"index": i},
                global_seq=0,
                sim_time=float(i * 5),  # 0, 5, 10
                source_id="stats_source",
                schema_version="1.0.0",
            )
            await log.append(effect)
            # Small delay to ensure different wall times
            await asyncio.sleep(0.001)

        return log

    def test_get_stats_populated(self, stats_log: EventLog) -> None:
        """Test statistics on populated log."""
        stats = stats_log.get_stats()

        assert stats["total_entries"] == 3
        assert stats["latest_seq"] == 3
        assert stats["seq_range"]["first"] == 1
        assert stats["seq_range"]["last"] == 3
        assert stats["time_range"]["sim_time"]["first"] == 0.0
        assert stats["time_range"]["sim_time"]["last"] == 10.0
        assert stats["time_range"]["sim_time"]["duration"] == 10.0
        assert stats["world_id"] == "stats_test"

    def test_get_stats_empty(self) -> None:
        """Test statistics on empty log."""
        log = EventLog("empty_stats")
        stats = log.get_stats()

        assert stats["total_entries"] == 0
        assert stats["latest_seq"] == 0
        assert stats["time_range"] is None
        assert stats["world_id"] == "empty_stats"

    @pytest.mark.asyncio
    async def test_compact_log(self, stats_log: EventLog) -> None:
        """Test log compaction."""
        # Add more entries to make compaction meaningful
        for i in range(3, 8):  # Add 5 more entries
            effect = Effect(
                uuid=f"compact-{i}",
                kind="CompactTest",
                payload={"index": i},
                global_seq=0,
                sim_time=float(i * 5),
                source_id="compact_source",
                schema_version="1.0.0",
            )
            await stats_log.append(effect)

        # Now we have 8 entries total
        assert stats_log.get_entry_count() == 8

        # Compact to keep only 5 entries
        removed_count = await stats_log.compact(keep_entries=5)

        assert removed_count == 3
        assert stats_log.get_entry_count() == 5

        # Verify remaining entries are the most recent ones
        entries = stats_log.get_all_entries()
        assert entries[0].global_seq == 4  # First remaining entry
        assert entries[-1].global_seq == 8  # Last entry

    @pytest.mark.asyncio
    async def test_compact_no_effect(self, stats_log: EventLog) -> None:
        """Test compaction when no entries need to be removed."""
        # Log has 3 entries, try to keep 5
        removed_count = await stats_log.compact(keep_entries=5)

        assert removed_count == 0
        assert stats_log.get_entry_count() == 3
=======
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
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7


class TestEventLogPerformance:
    """Test EventLog performance characteristics."""

    @pytest.mark.asyncio
<<<<<<< HEAD
    async def test_append_performance(self) -> None:
        """Test append operation performance."""
        log = EventLog("perf_test")

        # Measure time for 100 appends
        start_time = time.perf_counter()

        for i in range(100):
            effect = Effect(
                uuid=f"perf-{i}",
                kind="PerfTest",
                payload={"index": i},
                global_seq=0,
                sim_time=float(i),
                source_id="perf_source",
                schema_version="1.0.0",
            )
            await log.append(effect)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Should complete 100 appends in reasonable time (< 1 second)
        assert duration < 1.0
        assert log.get_entry_count() == 100

    def test_retrieval_performance(self) -> None:
        """Test retrieval operation performance."""
        # This would be an async fixture in real implementation
        # For now, just test that retrieval is fast
        log = EventLog("retrieval_perf")

        # Add some entries synchronously for this test
        for i in range(10):
            effect = Effect(
                uuid=f"retrieval-{i}",
                kind="RetrievalTest",
                payload={"index": i},
                global_seq=i + 1,
                sim_time=float(i),
                source_id="retrieval_source",
                schema_version="1.0.0",
            )
            # Manually add to bypass async append for this perf test
            entry = EventLogEntry(
                global_seq=i + 1,
                sim_time=float(i),
                wall_time=time.time(),
                effect=effect,
                source_metadata={"world_id": "retrieval_perf"},
                checksum=f"checksum-{i}",
                req_id=f"req-{i}",
            )
            log._entries.append(entry)
            log._seq_counter = i + 1

        # Test retrieval performance
        start_time = time.perf_counter()

        # Multiple retrieval operations
        for _ in range(100):
            entries = log.get_entries_since(5)
            assert len(entries) == 5

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Should complete 100 retrievals quickly
        assert duration < 0.1  # 100ms for 100 operations


class TestEventLogIntegration:
    """Integration tests for EventLog with other components."""

    @pytest.mark.asyncio
    async def test_with_telemetry_integration(self) -> None:
        """Test EventLog with telemetry integration."""
        from gunn.utils.telemetry import setup_logging

        # Setup logging
        setup_logging("DEBUG", enable_pii_redaction=False)

        log = EventLog("telemetry_test")

        # Append effect with telemetry
        effect = Effect(
            uuid="telemetry-test",
            kind="TelemetryTest",
            payload={"message": "test with telemetry"},
            global_seq=0,
            sim_time=1.0,
            source_id="telemetry_source",
            schema_version="1.0.0",
        )

        global_seq = await log.append(effect, req_id="telemetry-req")

        assert global_seq == 1

        # Validate integrity with telemetry
        result = log.validate_integrity()
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_replay_scenario(self) -> None:
        """Test a complete replay scenario."""
        log = EventLog("replay_test")

        # Simulate a sequence of events
        events = [
            ("AgentJoined", {"agent_id": "agent_1"}),
            ("MessageSent", {"from": "agent_1", "text": "Hello"}),
            ("AgentJoined", {"agent_id": "agent_2"}),
            ("MessageSent", {"from": "agent_2", "text": "Hi there"}),
            ("AgentLeft", {"agent_id": "agent_1"}),
        ]

        # Append all events
        for i, (kind, payload) in enumerate(events):
            effect = Effect(
                uuid=f"replay-{i}",
                kind=kind,
                payload=payload,
                global_seq=0,
                sim_time=float(i),
                source_id="replay_source",
                schema_version="1.0.0",
            )
            await log.append(effect, req_id=f"replay-req-{i}")

        # Simulate replay from sequence 2
        replay_entries = log.get_entries_since(2)

        assert len(replay_entries) == 3  # Last 3 events
        assert replay_entries[0].effect["kind"] == "AgentJoined"
        assert replay_entries[0].effect["payload"]["agent_id"] == "agent_2"
        assert replay_entries[-1].effect["kind"] == "AgentLeft"

        # Verify integrity throughout
        integrity = log.validate_integrity()
        assert integrity["valid"] is True
=======
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
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7
