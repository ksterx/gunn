"""Unit tests for DedupStore."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from gunn.storage.dedup_store import DedupStore


class TestDedupStore:
    """Test suite for DedupStore."""

    @pytest.mark.asyncio
    async def test_init_and_close(self):
        """Test initialization and cleanup."""
        store = DedupStore()

        # Should not be initialized yet
        with pytest.raises(RuntimeError, match="DedupStore not initialized"):
            await store.check_and_record("world1", "agent1", "req1", 1)

        # Initialize
        await store.initialize()

        # Should work now
        result = await store.check_and_record("world1", "agent1", "req1", 1)
        assert result is None  # New entry

        # Close
        await store.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        async with DedupStore() as store:
            result = await store.check_and_record("world1", "agent1", "req1", 1)
            assert result is None

    @pytest.mark.asyncio
    async def test_deduplication_basic(self):
        """Test basic deduplication functionality."""
        async with DedupStore() as store:
            # First request should be new
            result1 = await store.check_and_record("world1", "agent1", "req1", 100)
            assert result1 is None

            # Same request should be duplicate
            result2 = await store.check_and_record("world1", "agent1", "req1", 200)
            assert result2 == 100  # Returns original global_seq

            # Different req_id should be new
            result3 = await store.check_and_record("world1", "agent1", "req2", 300)
            assert result3 is None

            # Different agent should be new
            result4 = await store.check_and_record("world1", "agent2", "req1", 400)
            assert result4 is None

            # Different world should be new
            result5 = await store.check_and_record("world2", "agent1", "req1", 500)
            assert result5 is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration of entries."""
        # Use very short TTL for testing, disable warmup
        async with DedupStore(
            dedup_ttl_minutes=0.01,  # 0.6 seconds
            warmup_ttl_minutes=0.001,  # Very short warmup
        ) as store:
            # Record entry
            result1 = await store.check_and_record("world1", "agent1", "req1", 100)
            assert result1 is None

            # Should still be duplicate immediately
            result2 = await store.check_and_record("world1", "agent1", "req1", 200)
            assert result2 == 100

            # Wait for warmup to end
            await asyncio.sleep(0.1)

            # Wait for expiration
            await asyncio.sleep(0.8)

            # Should be new again after expiration
            result3 = await store.check_and_record("world1", "agent1", "req1", 300)
            assert result3 is None

    @pytest.mark.asyncio
    async def test_warmup_ttl_guard(self):
        """Test TTL warmup guard for relaxed deduplication after restart."""
        # Use short TTL with warmup
        async with DedupStore(
            dedup_ttl_minutes=0.01,  # 0.6 seconds normal TTL
            warmup_ttl_minutes=0.02,  # 1.2 seconds warmup period
        ) as store:
            # Record entry
            await store.check_and_record("world1", "agent1", "req1", 100)

            # Wait past normal TTL but within warmup period
            await asyncio.sleep(0.8)

            # Should still be duplicate due to warmup guard
            result = await store.check_and_record("world1", "agent1", "req1", 200)
            assert result == 100

            # Wait past warmup period
            await asyncio.sleep(0.6)

            # Should be new now
            result = await store.check_and_record("world1", "agent1", "req1", 300)
            assert result is None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        async with DedupStore(
            dedup_ttl_minutes=0.01,  # 0.6 seconds
            warmup_ttl_minutes=0.001,  # Very short warmup
        ) as store:
            # Add some entries
            await store.check_and_record("world1", "agent1", "req1", 100)
            await store.check_and_record("world1", "agent1", "req2", 200)
            await store.check_and_record("world1", "agent2", "req1", 300)

            # Wait for warmup to end
            await asyncio.sleep(0.1)

            # Wait for expiration
            await asyncio.sleep(0.8)

            # Add one more recent entry
            await store.check_and_record("world1", "agent1", "req3", 400)

            # Cleanup expired entries
            cleaned = await store.cleanup_expired()
            assert cleaned == 3  # Should clean up the 3 expired entries

            # Recent entry should still be duplicate
            result = await store.check_and_record("world1", "agent1", "req3", 500)
            assert result == 400

    @pytest.mark.asyncio
    async def test_cleanup_excess(self):
        """Test cleanup of excess entries beyond max limit."""
        async with DedupStore(max_entries=5) as store:
            # Add more entries than the limit
            for i in range(10):
                await store.check_and_record("world1", "agent1", f"req{i}", i * 100)

            # Cleanup excess entries
            cleaned = await store.cleanup_excess()
            assert cleaned == 5  # Should clean up 5 oldest entries

            # Verify recent entries are still there
            result = await store.check_and_record("world1", "agent1", "req9", 1000)
            assert result == 900  # Should still be duplicate

            # Verify old entries are gone
            result = await store.check_and_record("world1", "agent1", "req0", 1100)
            assert result is None  # Should be new (old entry cleaned up)

    @pytest.mark.asyncio
    async def test_persistent_storage(self):
        """Test persistent storage with file database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            # Create store and add entry
            async with DedupStore(db_path=db_path) as store1:
                result = await store1.check_and_record("world1", "agent1", "req1", 100)
                assert result is None

            # Create new store with same database
            async with DedupStore(db_path=db_path) as store2:
                # Entry should still be there
                result = await store2.check_and_record("world1", "agent1", "req1", 200)
                assert result == 100

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test statistics gathering."""
        async with DedupStore(
            dedup_ttl_minutes=1,
            max_entries=100,
            warmup_ttl_minutes=0.01,  # 0.6 seconds warmup
        ) as store:
            # Initially empty
            stats = await store.get_stats()
            assert stats["status"] == "initialized"
            assert stats["total_entries"] == 0
            assert stats["recent_entries"] == 0
            assert stats["warmup_active"] is True

            # Add some entries
            await store.check_and_record("world1", "agent1", "req1", 100)
            await store.check_and_record("world1", "agent1", "req2", 200)
            await store.check_and_record("world1", "agent2", "req1", 300)

            stats = await store.get_stats()
            assert stats["total_entries"] == 3
            assert stats["recent_entries"] == 3
            assert stats["max_entries"] == 100
            assert stats["dedup_ttl_seconds"] == 60
            assert stats["oldest_entry_age"] is not None
            assert stats["newest_entry_age"] is not None

            # Wait for warmup to end
            await asyncio.sleep(0.8)

            stats = await store.get_stats()
            assert stats["warmup_active"] is False

    @pytest.mark.asyncio
    async def test_cleanup_loop_integration(self):
        """Test that cleanup loop runs automatically."""
        async with DedupStore(
            dedup_ttl_minutes=0.01,  # 0.6 seconds
            cleanup_interval_minutes=0.005,  # 0.3 seconds cleanup interval
            warmup_ttl_minutes=0.001,  # Very short warmup
        ) as store:
            # Add entries
            await store.check_and_record("world1", "agent1", "req1", 100)
            await store.check_and_record("world1", "agent1", "req2", 200)

            # Wait for warmup to end
            await asyncio.sleep(0.1)

            # Wait for entries to expire and cleanup to run
            await asyncio.sleep(1.0)

            # Entries should be cleaned up automatically
            stats = await store.get_stats()
            assert stats["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent access to deduplication store."""
        async with DedupStore(warmup_ttl_minutes=0.001) as store:  # Short warmup
            # Simulate concurrent requests for same req_id
            async def make_request(seq):
                try:
                    return await store.check_and_record("world1", "agent1", "req1", seq)
                except Exception as e:
                    # Should not happen with proper race condition handling
                    pytest.fail(f"Unexpected error in concurrent access: {e}")

            # Run multiple concurrent requests
            tasks = [make_request(i * 100) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # Only one should be None (new), others should return the same global_seq
            new_count = sum(1 for r in results if r is None)
            assert new_count == 1

            # All non-None results should be the same
            non_none_results = [r for r in results if r is not None]
            if non_none_results:
                assert all(r == non_none_results[0] for r in non_none_results)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling scenarios."""
        store = DedupStore()

        # Should raise error when not initialized
        with pytest.raises(RuntimeError, match="DedupStore not initialized"):
            await store.check_and_record("world1", "agent1", "req1", 1)

        with pytest.raises(RuntimeError, match="DedupStore not initialized"):
            await store.cleanup_expired()

        # Stats should work even when not initialized
        stats = await store.get_stats()
        assert stats["status"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_multiple_worlds_isolation(self):
        """Test that different worlds are properly isolated."""
        async with DedupStore() as store:
            # Add same req_id in different worlds
            result1 = await store.check_and_record("world1", "agent1", "req1", 100)
            assert result1 is None

            result2 = await store.check_and_record("world2", "agent1", "req1", 200)
            assert result2 is None  # Should be new in different world

            # Verify isolation
            result3 = await store.check_and_record("world1", "agent1", "req1", 300)
            assert result3 == 100  # Should be duplicate in world1

            result4 = await store.check_and_record("world2", "agent1", "req1", 400)
            assert result4 == 200  # Should be duplicate in world2
