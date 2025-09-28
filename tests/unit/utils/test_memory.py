"""Unit tests for memory management and compaction utilities."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gunn.core.event_log import EventLog
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Effect
from gunn.utils.memory import (
    MemoryConfig,
    MemoryManager,
    MemoryStats,
    SnapshotManager,
    ViewCache,
    WorldStateSnapshot,
)


class TestViewCache:
    """Test cases for ViewCache LRU implementation."""

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = ViewCache(max_size=3)

        # Test put and get
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ViewCache(max_size=2)

        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # New entry

    def test_update_existing_key(self):
        """Test updating existing key doesn't cause eviction."""
        cache = ViewCache(max_size=2)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Update existing key
        cache.put("key1", "new_value1")

        assert cache.get("key1") == "new_value1"
        assert cache.get("key2") == "value2"

    def test_evict_old_entries(self):
        """Test eviction of old entries by age."""
        cache = ViewCache(max_size=10)

        # Add entries with mocked access times
        with patch("time.time", return_value=1000.0):
            cache.put("old1", "value1")
            cache.put("old2", "value2")

        with patch("time.time", return_value=2000.0):
            cache.put("new1", "value3")
            cache.put("new2", "value4")

        # Evict entries older than 500 seconds
        with patch("time.time", return_value=2000.0):
            evicted_count = cache.evict_old_entries(max_age_seconds=500.0)

        assert evicted_count == 2
        assert cache.get("old1") is None
        assert cache.get("old2") is None
        assert cache.get("new1") == "value3"
        assert cache.get("new2") == "value4"

    def test_clear(self):
        """Test clearing all cache entries."""
        cache = ViewCache(max_size=10)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_get_stats(self):
        """Test cache statistics."""
        cache = ViewCache(max_size=5)

        with patch("time.time", return_value=1000.0):
            cache.put("key1", "value1")
            cache.put("key2", "value2")

        with patch("time.time", return_value=1100.0):
            stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 5
        assert stats["utilization"] == 0.4
        assert stats["oldest_access_age"] == 100.0
        assert stats["newest_access_age"] == 100.0


class TestSnapshotManager:
    """Test cases for SnapshotManager."""

    @pytest.fixture
    def config(self):
        """Memory configuration for testing."""
        return MemoryConfig(max_snapshots=3, snapshot_interval=100)

    @pytest.fixture
    def world_state(self):
        """Sample world state for testing."""
        return WorldState(
            entities={"entity1": {"x": 10, "y": 20}},
            relationships={"entity1": ["entity2"]},
            spatial_index={"entity1": (10, 20, 0)},
        )

    @pytest.mark.asyncio
    async def test_create_snapshot(self, config, world_state):
        """Test snapshot creation."""
        manager = SnapshotManager(config)

        with patch("time.time", return_value=1000.0):
            snapshot = await manager.create_snapshot(
                global_seq=100, world_state=world_state, sim_time=50.0
            )

        assert snapshot.global_seq == 100
        assert snapshot.sim_time == 50.0
        assert snapshot.wall_time == 1000.0
        assert snapshot.world_state == world_state
        assert len(snapshot.checksum) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_max_snapshots_limit(self, config, world_state):
        """Test that old snapshots are removed when limit is exceeded."""
        manager = SnapshotManager(config)

        # Create more snapshots than the limit
        for i in range(5):
            await manager.create_snapshot(
                global_seq=i * 100, world_state=world_state, sim_time=i * 10.0
            )

        snapshots = manager.get_all_snapshots()
        assert len(snapshots) == 3  # Should be limited to max_snapshots

        # Should keep the most recent ones
        assert snapshots[0].global_seq == 200
        assert snapshots[1].global_seq == 300
        assert snapshots[2].global_seq == 400

    def test_find_nearest_snapshot(self, config, world_state):
        """Test finding nearest snapshot."""
        manager = SnapshotManager(config)

        # Add snapshots manually for testing
        manager._snapshots = [
            WorldStateSnapshot(
                global_seq=100,
                sim_time=10.0,
                wall_time=1000.0,
                world_state=world_state,
                checksum="checksum1",
            ),
            WorldStateSnapshot(
                global_seq=200,
                sim_time=20.0,
                wall_time=2000.0,
                world_state=world_state,
                checksum="checksum2",
            ),
            WorldStateSnapshot(
                global_seq=300,
                sim_time=30.0,
                wall_time=3000.0,
                world_state=world_state,
                checksum="checksum3",
            ),
        ]

        # Test finding nearest snapshots
        assert manager.find_nearest_snapshot(50) is None
        assert manager.find_nearest_snapshot(150).global_seq == 100
        assert manager.find_nearest_snapshot(250).global_seq == 200
        assert manager.find_nearest_snapshot(350).global_seq == 300

    @pytest.mark.asyncio
    async def test_validate_snapshot_integrity(self, config, world_state):
        """Test snapshot integrity validation."""
        manager = SnapshotManager(config)

        # Create valid snapshot
        snapshot = await manager.create_snapshot(
            global_seq=100, world_state=world_state, sim_time=50.0
        )

        # Should validate successfully
        assert await manager.validate_snapshot_integrity(snapshot) is True

        # Corrupt the checksum
        snapshot.checksum = "invalid_checksum"
        assert await manager.validate_snapshot_integrity(snapshot) is False

    def test_get_stats(self, config, world_state):
        """Test snapshot statistics."""
        manager = SnapshotManager(config)

        # Empty manager
        stats = manager.get_stats()
        assert stats["total_snapshots"] == 0
        assert stats["oldest_seq"] is None

        # Add snapshots
        manager._snapshots = [
            WorldStateSnapshot(
                global_seq=100,
                sim_time=10.0,
                wall_time=1000.0,
                world_state=world_state,
                checksum="checksum1",
            ),
            WorldStateSnapshot(
                global_seq=300,
                sim_time=30.0,
                wall_time=3000.0,
                world_state=world_state,
                checksum="checksum2",
            ),
        ]

        stats = manager.get_stats()
        assert stats["total_snapshots"] == 2
        assert stats["oldest_seq"] == 100
        assert stats["newest_seq"] == 300
        assert stats["seq_range"] == 200


class TestMemoryManager:
    """Test cases for MemoryManager."""

    @pytest.fixture
    def config(self):
        """Memory configuration for testing."""
        return MemoryConfig(
            max_log_entries=1000,
            view_cache_size=100,
            compaction_threshold=500,
            snapshot_interval=100,
            max_snapshots=5,
            auto_compaction_enabled=True,
        )

    @pytest.fixture
    def world_state(self):
        """Sample world state for testing."""
        return WorldState(
            entities={"entity1": {"x": 10, "y": 20}},
            relationships={"entity1": ["entity2"]},
            spatial_index={"entity1": (10, 20, 0)},
        )

    @pytest.fixture
    def mock_event_log(self):
        """Mock event log for testing."""
        log = MagicMock(spec=EventLog)
        log.get_entry_count.return_value = 0
        log.get_all_entries.return_value = []
        log.compact = AsyncMock(return_value=0)
        return log

    def test_initialization(self, config):
        """Test memory manager initialization."""
        manager = MemoryManager(config)

        assert manager.config == config
        assert manager.view_cache.max_size == config.view_cache_size
        assert manager._last_snapshot_seq == 0
        assert not manager._compaction_in_progress

    @pytest.mark.asyncio
    async def test_check_and_create_snapshot(self, config, world_state):
        """Test automatic snapshot creation."""
        manager = MemoryManager(config)

        # Should not create snapshot initially
        snapshot = await manager.check_and_create_snapshot(50, world_state, 10.0)
        assert snapshot is None

        # Should create snapshot when interval is reached
        snapshot = await manager.check_and_create_snapshot(100, world_state, 20.0)
        assert snapshot is not None
        assert snapshot.global_seq == 100
        assert snapshot.sim_time == 20.0

        # Should not create another snapshot immediately
        snapshot = await manager.check_and_create_snapshot(150, world_state, 30.0)
        assert snapshot is None

        # Should create another snapshot after interval
        snapshot = await manager.check_and_create_snapshot(200, world_state, 40.0)
        assert snapshot is not None
        assert snapshot.global_seq == 200

    @pytest.mark.asyncio
    async def test_compact_log_no_snapshots(self, config, mock_event_log):
        """Test log compaction fails without snapshots."""
        manager = MemoryManager(config)
        mock_event_log.get_entry_count.return_value = 1000

        removed_count = await manager.compact_log(mock_event_log)

        assert removed_count == 0
        mock_event_log.compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_log_with_snapshots(
        self, config, mock_event_log, world_state
    ):
        """Test successful log compaction with snapshots."""
        manager = MemoryManager(config)

        # Create a snapshot
        await manager.snapshot_manager.create_snapshot(500, world_state, 50.0)

        # Mock log entries
        mock_entries = []
        for i in range(1000):
            entry = MagicMock()
            entry.global_seq = i + 1
            mock_entries.append(entry)

        mock_event_log.get_entry_count.return_value = 1000
        mock_event_log.get_all_entries.return_value = mock_entries
        mock_event_log.compact.return_value = 450  # Removed 450 entries

        removed_count = await manager.compact_log(mock_event_log)

        assert removed_count == 450
        mock_event_log.compact.assert_called_once()

    @pytest.mark.asyncio
    async def test_compact_log_below_threshold(self, config, mock_event_log):
        """Test that compaction is skipped when below threshold."""
        manager = MemoryManager(config)
        mock_event_log.get_entry_count.return_value = 100  # Below threshold

        removed_count = await manager.compact_log(mock_event_log)

        assert removed_count == 0
        mock_event_log.compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_compaction_prevention(self, config, mock_event_log):
        """Test that concurrent compaction is prevented."""
        manager = MemoryManager(config)
        manager._compaction_in_progress = True

        removed_count = await manager.compact_log(mock_event_log)

        assert removed_count == 0
        mock_event_log.compact.assert_not_called()

    def test_evict_old_views(self, config):
        """Test view cache eviction."""
        manager = MemoryManager(config)

        # Add some views to cache
        with patch("time.time", return_value=1000.0):
            manager.view_cache.put("old_view", {"data": "old"})

        with patch("time.time", return_value=2000.0):
            manager.view_cache.put("new_view", {"data": "new"})

        # Evict old views
        with patch("time.time", return_value=2000.0):
            evicted_count = manager.evict_old_views(max_age_seconds=500.0)

        assert evicted_count == 1
        assert manager.view_cache.get("old_view") is None
        assert manager.view_cache.get("new_view") is not None

    @pytest.mark.asyncio
    async def test_check_memory_limits_triggers_compaction(
        self, config, mock_event_log, world_state
    ):
        """Test that memory limit checking triggers compaction."""
        manager = MemoryManager(config)

        # Create snapshot for compaction
        await manager.snapshot_manager.create_snapshot(500, world_state, 50.0)

        # Mock log over limit
        mock_event_log.get_entry_count.return_value = 1500  # Over max_log_entries
        mock_entries = [MagicMock() for _ in range(1500)]
        for i, entry in enumerate(mock_entries):
            entry.global_seq = i + 1
        mock_event_log.get_all_entries.return_value = mock_entries
        mock_event_log.compact.return_value = 500

        # Reset last check time to force check
        manager._last_memory_check = 0.0

        result = await manager.check_memory_limits(mock_event_log)

        assert result is True
        mock_event_log.compact.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_memory_limits_rate_limiting(self, config, mock_event_log):
        """Test that memory checks are rate limited."""
        manager = MemoryManager(config)
        manager._last_memory_check = time.time()  # Recent check

        mock_event_log.get_entry_count.return_value = 1500  # Over limit

        result = await manager.check_memory_limits(mock_event_log)

        assert result is True
        mock_event_log.compact.assert_not_called()  # Should be rate limited

    def test_estimate_memory_usage(self, config, mock_event_log):
        """Test memory usage estimation."""
        manager = MemoryManager(config)

        # Mock log entries with timestamps
        mock_entries = []
        for i in range(100):
            entry = MagicMock()
            entry.wall_time = 1000.0 + i
            entry.global_seq = i + 1
            mock_entries.append(entry)

        mock_event_log.get_all_entries.return_value = mock_entries

        # Add some cache entries
        manager.view_cache.put("view1", {"data": "test"})
        manager.view_cache.put("view2", {"data": "test"})

        # Add snapshots with proper global_seq attributes
        snapshot1 = MagicMock()
        snapshot1.global_seq = 100
        snapshot2 = MagicMock()
        snapshot2.global_seq = 200
        manager.snapshot_manager._snapshots = [snapshot1, snapshot2]

        with patch("time.time", return_value=1200.0):
            stats = manager.estimate_memory_usage(mock_event_log)

        assert isinstance(stats, MemoryStats)
        assert stats.total_log_entries == 100
        assert stats.total_snapshots == 2
        assert stats.view_cache_size == 2
        assert stats.estimated_memory_mb > 0
        assert stats.oldest_entry_age_seconds == 200.0  # 1200 - 1000
        assert stats.newest_entry_age_seconds == 101.0  # 1200 - 1099

    def test_get_detailed_stats(self, config, mock_event_log):
        """Test detailed statistics collection."""
        manager = MemoryManager(config)

        mock_event_log.get_all_entries.return_value = []

        stats = manager.get_detailed_stats(mock_event_log)

        assert "memory_usage" in stats
        assert "view_cache" in stats
        assert "snapshots" in stats
        assert "config" in stats
        assert "status" in stats

        # Check config values
        assert stats["config"]["max_log_entries"] == config.max_log_entries
        assert stats["config"]["view_cache_size"] == config.view_cache_size

        # Check status values
        assert stats["status"]["compaction_in_progress"] is False
        assert stats["status"]["last_snapshot_seq"] == 0

    @pytest.mark.asyncio
    async def test_cleanup(self, config):
        """Test memory manager cleanup."""
        manager = MemoryManager(config)

        # Add some data
        manager.view_cache.put("test", {"data": "test"})

        await manager.cleanup()

        assert manager.view_cache.get("test") is None


@pytest.mark.asyncio
async def test_memory_manager_integration():
    """Integration test for memory manager with real event log."""
    config = MemoryConfig(
        max_log_entries=50,
        compaction_threshold=30,
        snapshot_interval=10,
        max_snapshots=3,
        auto_compaction_enabled=True,
    )

    manager = MemoryManager(config)
    event_log = EventLog("test_world")
    world_state = WorldState()

    # Add many effects to trigger compaction
    for i in range(60):
        effect: Effect = {
            "uuid": f"uuid_{i}",
            "kind": "TestEffect",
            "payload": {"index": i},
            "source_id": "test",
            "schema_version": "1.0.0",
            "sim_time": float(i),
            "global_seq": i + 1,
        }

        await event_log.append(effect)

        # Create snapshots periodically
        if (i + 1) % 10 == 0:
            await manager.check_and_create_snapshot(i + 1, world_state, float(i))

    # Check memory limits (should trigger compaction)
    manager._last_memory_check = 0.0  # Force check
    result = await manager.check_memory_limits(event_log)

    assert result is True

    # Verify compaction occurred
    remaining_entries = event_log.get_entry_count()
    assert remaining_entries < 60  # Some entries should be removed

    # Verify snapshots exist
    snapshots = manager.snapshot_manager.get_all_snapshots()
    assert len(snapshots) > 0

    # Get comprehensive stats
    stats = manager.get_detailed_stats(event_log)
    assert stats["memory_usage"]["total_log_entries"] == remaining_entries
    assert stats["snapshots"]["total_snapshots"] > 0

    # Cleanup
    await manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
