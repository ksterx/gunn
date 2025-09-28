"""Unit tests for deduplication store.

Tests SQLite and in-memory deduplication stores with TTL cleanup,
warmup guards, and concurrent access patterns.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from gunn.storage.dedup_store import DedupStore, InMemoryDedupStore


class TestInMemoryDedupStore:
    """Test suite for in-memory deduplication store."""

    @pytest.fixture
    async def dedup_store(self):
        """Create in-memory dedup store for testing."""
        store = InMemoryDedupStore(
            dedup_ttl_minutes=1,
            max_entries=100,
            warmup_duration_minutes=0,  # No warmup for most tests
        )
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_basic_deduplication(self, dedup_store):
        """Test basic deduplication functionality."""
        # First check should return None (new intent)
        result = await dedup_store.check_and_record("world1", "agent1", "req1", 1)
        assert result is None

        # Second check should return existing sequence
        result = await dedup_store.check_and_record("world1", "agent1", "req1", 2)
        assert result == 1

    @pytest.mark.asyncio
    async def test_different_keys(self, dedup_store):
        """Test that different keys are treated separately."""
        # Different world_id
        result1 = await dedup_store.check_and_record("world1", "agent1", "req1", 1)
        result2 = await dedup_store.check_and_record("world2", "agent1", "req1", 2)
        assert result1 is None
        assert result2 is None

        # Different agent_id
        result3 = await dedup_store.check_and_record("world1", "agent2", "req1", 3)
        assert result3 is None

        # Different req_id
        result4 = await dedup_store.check_and_record("world1", "agent1", "req2", 4)
        assert result4 is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration of entries."""
        # Create store with very short TTL
        store = InMemoryDedupStore(
            dedup_ttl_minutes=0.01,  # 0.6 seconds
            warmup_duration_minutes=0,
        )
        await store.initialize()

        try:
            # Record entry
            result = await store.check_and_record("world1", "agent1", "req1", 1)
            assert result is None

            # Should still be there immediately
            result = await store.check_and_record("world1", "agent1", "req1", 2)
            assert result == 1

            # Wait for TTL expiration
            await asyncio.sleep(1.0)

            # Should be expired now
            result = await store.check_and_record("world1", "agent1", "req1", 3)
            assert result is None

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        # Create store with short TTL
        store = InMemoryDedupStore(
            dedup_ttl_minutes=0.01,  # 0.6 seconds
            warmup_duration_minutes=0,
        )
        await store.initialize()

        try:
            # Add several entries
            for i in range(5):
                await store.check_and_record("world1", "agent1", f"req{i}", i)

            # All should be active initially
            stats = await store.get_stats()
            assert stats["total_entries"] == 5
            assert stats["active_entries"] == 5

            # Wait for expiration
            await asyncio.sleep(1.0)

            # Cleanup expired entries
            cleaned = await store.cleanup_expired()
            assert cleaned == 5

            # Should be empty now
            stats = await store.get_stats()
            assert stats["total_entries"] == 0

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_cleanup_excess_entries(self):
        """Test cleanup when max entries is exceeded."""
        store = InMemoryDedupStore(
            max_entries=3,
            warmup_duration_minutes=0,
        )
        await store.initialize()

        try:
            # Add entries up to limit
            for i in range(3):
                await store.check_and_record("world1", "agent1", f"req{i}", i)

            stats = await store.get_stats()
            assert stats["total_entries"] == 3

            # Add more entries to exceed limit
            for i in range(3, 6):
                await store.check_and_record("world1", "agent1", f"req{i}", i)

            stats = await store.get_stats()
            assert stats["total_entries"] == 6

            # Cleanup excess entries
            cleaned = await store.cleanup_excess_entries()
            assert cleaned == 3

            # Should be back to limit
            stats = await store.get_stats()
            assert stats["total_entries"] == 3

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_warmup_period(self):
        """Test warmup period with relaxed deduplication."""
        store = InMemoryDedupStore(
            warmup_duration_minutes=0.02,  # 1.2 seconds
        )
        await store.initialize()

        try:
            # During warmup, duplicates should be allowed
            result1 = await store.check_and_record("world1", "agent1", "req1", 1)
            assert result1 is None

            result2 = await store.check_and_record("world1", "agent1", "req1", 2)
            assert result2 is None  # Should be None during warmup

            # Wait for warmup to end
            await asyncio.sleep(1.5)

            # After warmup, deduplication should work normally
            result3 = await store.check_and_record("world1", "agent1", "req1", 3)
            assert result3 == 2  # Should return the last recorded seq

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_concurrent_access(self, dedup_store):
        """Test concurrent access to dedup store."""

        async def worker(worker_id: int):
            results = []
            for i in range(10):
                result = await dedup_store.check_and_record(
                    "world1", f"agent{worker_id}", f"req{i}", worker_id * 100 + i
                )
                results.append(result)
            return results

        # Run multiple workers concurrently
        tasks = [worker(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Each worker should have all None results (no duplicates within worker)
        for worker_results in results:
            assert all(result is None for result in worker_results)

    @pytest.mark.asyncio
    async def test_stats(self, dedup_store):
        """Test statistics reporting."""
        # Initial stats
        stats = await dedup_store.get_stats()
        assert stats["status"] == "initialized"
        assert stats["total_entries"] == 0
        assert stats["active_entries"] == 0

        # Add some entries
        for i in range(5):
            await dedup_store.check_and_record("world1", "agent1", f"req{i}", i)

        stats = await dedup_store.get_stats()
        assert stats["total_entries"] == 5
        assert stats["active_entries"] == 5
        assert not stats["in_warmup"]


class TestSQLiteDedupStore:
    """Test suite for SQLite deduplication store."""

    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        yield db_path

        # Cleanup
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    async def dedup_store(self, temp_db_path):
        """Create SQLite dedup store for testing."""
        store = DedupStore(
            db_path=temp_db_path,
            dedup_ttl_minutes=1,
            max_entries=100,
            cleanup_interval_minutes=1,
            warmup_duration_minutes=0,
        )
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_persistence(self, temp_db_path):
        """Test that data persists across store instances."""
        # Create first store instance
        store1 = DedupStore(
            db_path=temp_db_path,
            warmup_duration_minutes=0,
        )
        await store1.initialize()

        # Record entry
        result = await store1.check_and_record("world1", "agent1", "req1", 1)
        assert result is None

        await store1.close()

        # Create second store instance with same database
        store2 = DedupStore(
            db_path=temp_db_path,
            warmup_duration_minutes=0,
        )
        await store2.initialize()

        # Entry should still exist
        result = await store2.check_and_record("world1", "agent1", "req1", 2)
        assert result == 1

        await store2.close()

    @pytest.mark.asyncio
    async def test_sqlite_basic_operations(self, dedup_store):
        """Test basic SQLite operations."""
        # Same tests as in-memory but with SQLite backend
        result = await dedup_store.check_and_record("world1", "agent1", "req1", 1)
        assert result is None

        result = await dedup_store.check_and_record("world1", "agent1", "req1", 2)
        assert result == 1

    @pytest.mark.asyncio
    async def test_sqlite_cleanup(self, dedup_store):
        """Test SQLite cleanup operations."""
        # Add entries
        for i in range(10):
            await dedup_store.check_and_record("world1", "agent1", f"req{i}", i)

        stats = await dedup_store.get_stats()
        assert stats["total_entries"] == 10

        # Test cleanup (won't clean anything since TTL is 1 minute)
        cleaned = await dedup_store.cleanup_expired()
        assert cleaned == 0

        # Test excess cleanup
        cleaned = await dedup_store.cleanup_excess_entries()
        assert cleaned == 0  # Under limit

    @pytest.mark.asyncio
    async def test_sqlite_concurrent_access(self, dedup_store):
        """Test concurrent access to SQLite store."""

        async def worker(worker_id: int):
            results = []
            for i in range(5):
                result = await dedup_store.check_and_record(
                    "world1", f"agent{worker_id}", f"req{i}", worker_id * 100 + i
                )
                results.append(result)
            return results

        # Run multiple workers concurrently
        tasks = [worker(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        # Each worker should have all None results
        for worker_results in results:
            assert all(result is None for result in worker_results)

    @pytest.mark.asyncio
    async def test_sqlite_wal_mode(self, temp_db_path):
        """Test that SQLite is configured with WAL mode for better concurrency."""
        store = DedupStore(db_path=temp_db_path)
        await store.initialize()

        try:
            # Check that WAL mode is enabled
            if store._db is not None:
                async with store._db.execute("PRAGMA journal_mode") as cursor:
                    row = await cursor.fetchone()
                    assert row is not None and row[0].lower() == "wal"

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_cleanup_loop(self, temp_db_path):
        """Test automatic cleanup loop."""
        # Create store with very short cleanup interval
        store = DedupStore(
            db_path=temp_db_path,
            cleanup_interval_minutes=0.01,  # 0.6 seconds
            warmup_duration_minutes=0,
        )
        await store.initialize()

        try:
            # Add some entries
            for i in range(5):
                await store.check_and_record("world1", "agent1", f"req{i}", i)

            stats = await store.get_stats()
            assert stats["total_entries"] == 5

            # Wait for cleanup loop to run (it should run but not clean anything due to TTL)
            await asyncio.sleep(1.0)

            # Entries should still be there (TTL not expired)
            stats = await store.get_stats()
            assert stats["total_entries"] == 5

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, temp_db_path):
        """Test async context manager interface."""
        async with DedupStore(db_path=temp_db_path, warmup_duration_minutes=0) as store:
            result = await store.check_and_record("world1", "agent1", "req1", 1)
            assert result is None

            result = await store.check_and_record("world1", "agent1", "req1", 2)
            assert result == 1

        # Store should be closed automatically

    @pytest.mark.asyncio
    async def test_error_handling(self, dedup_store):
        """Test error handling for invalid inputs."""
        # Test with empty strings
        with pytest.raises((ValueError, Exception)):
            await dedup_store.check_and_record("", "agent1", "req1", 1)

        with pytest.raises((ValueError, Exception)):
            await dedup_store.check_and_record("world1", "", "req1", 1)

        with pytest.raises((ValueError, Exception)):
            await dedup_store.check_and_record("world1", "agent1", "", 1)

    @pytest.mark.asyncio
    async def test_memory_vs_sqlite_consistency(self, temp_db_path):
        """Test that in-memory and SQLite stores behave consistently."""
        # Test same operations on both stores
        memory_store = InMemoryDedupStore(warmup_duration_minutes=0)
        sqlite_store = DedupStore(db_path=temp_db_path, warmup_duration_minutes=0)

        await memory_store.initialize()
        await sqlite_store.initialize()

        try:
            # Perform same operations on both
            operations = [
                ("world1", "agent1", "req1", 1),
                ("world1", "agent1", "req1", 2),  # Duplicate
                ("world1", "agent2", "req1", 3),  # Different agent
                ("world2", "agent1", "req1", 4),  # Different world
            ]

            memory_results = []
            sqlite_results = []

            for world_id, agent_id, req_id, seq in operations:
                mem_result = await memory_store.check_and_record(
                    world_id, agent_id, req_id, seq
                )
                sql_result = await sqlite_store.check_and_record(
                    world_id, agent_id, req_id, seq
                )

                memory_results.append(mem_result)
                sqlite_results.append(sql_result)

            # Results should be identical
            assert memory_results == sqlite_results
            assert memory_results == [None, 1, None, None]

        finally:
            await memory_store.close()
            await sqlite_store.close()
