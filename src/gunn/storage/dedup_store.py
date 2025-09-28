"""Deduplication storage using SQLite for intent idempotency.

This module provides persistent storage for intent deduplication with TTL
cleanup and warmup guards for relaxed deduplication after restart.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

import aiosqlite

from gunn.utils.telemetry import get_logger


class DedupStore:
    """SQLite-based deduplication store for intent idempotency.

    Provides persistent storage for tracking processed intents with TTL-based
    cleanup and warmup guards for relaxed deduplication after restart.

    Requirements addressed:
    - 3.3: Two-phase commit with idempotency checking
    - 10.1: Idempotent ingestion where duplicate req_id must not create duplicate effects
    - 10.2: TTL cleanup for expired entries
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        dedup_ttl_minutes: int | float = 60,
        max_entries: int = 10000,
        cleanup_interval_minutes: int | float = 10,
        warmup_duration_minutes: int | float = 5,
    ):
        """Initialize deduplication store.

        Args:
            db_path: Path to SQLite database file (":memory:" for in-memory)
            dedup_ttl_minutes: TTL for deduplication entries in minutes
            max_entries: Maximum number of entries before cleanup
            cleanup_interval_minutes: Interval between cleanup runs
            warmup_duration_minutes: Duration for relaxed deduplication after startup
        """
        self.db_path = str(db_path)
        self.dedup_ttl_seconds = dedup_ttl_minutes * 60
        self.max_entries = max_entries
        self.cleanup_interval_seconds = cleanup_interval_minutes * 60
        self.warmup_duration_seconds = warmup_duration_minutes * 60

        self._db: aiosqlite.Connection | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._startup_time = time.time()
        self._logger = get_logger("gunn.storage.dedup")

        self._logger.info(
            "DedupStore initialized",
            db_path=self.db_path,
            dedup_ttl_minutes=dedup_ttl_minutes,
            max_entries=max_entries,
            cleanup_interval_minutes=cleanup_interval_minutes,
            warmup_duration_minutes=warmup_duration_minutes,
        )

    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        self._db = await aiosqlite.connect(self.db_path)

        # Enable WAL mode for better concurrency
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")

        # Create deduplication table
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS intent_dedup (
                world_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                req_id TEXT NOT NULL,
                global_seq INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (world_id, agent_id, req_id)
            )
        """
        )

        # Create index for TTL cleanup
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intent_dedup_created_at
            ON intent_dedup(created_at)
        """
        )

        await self._db.commit()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._logger.info("DedupStore database initialized")

    async def close(self) -> None:
        """Close database connection and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._db:
            await self._db.close()
            self._db = None

        self._logger.info("DedupStore closed")

    async def check_and_record(
        self, world_id: str, agent_id: str, req_id: str, global_seq: int
    ) -> int | None:
        """Check if intent was already processed and record if not.

        Args:
            world_id: World identifier
            agent_id: Agent identifier
            req_id: Request identifier
            global_seq: Global sequence number for this intent

        Returns:
            None if intent is new (and was recorded)
            Existing global_seq if intent was already processed

        Raises:
            RuntimeError: If database is not initialized
            ValueError: If any identifier is empty
        """
        if not self._db:
            raise RuntimeError("DedupStore not initialized")

        # Validate inputs
        if not world_id.strip():
            raise ValueError("world_id cannot be empty")
        if not agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        if not req_id.strip():
            raise ValueError("req_id cannot be empty")

        # Check if we're in warmup period (relaxed deduplication)
        current_time = time.time()
        in_warmup = (current_time - self._startup_time) < self.warmup_duration_seconds

        if in_warmup:
            self._logger.debug(
                "In warmup period, skipping deduplication check",
                world_id=world_id,
                agent_id=agent_id,
                req_id=req_id,
                warmup_remaining_seconds=self.warmup_duration_seconds
                - (current_time - self._startup_time),
            )
            # Still record the intent for future deduplication
            await self._record_intent(
                world_id, agent_id, req_id, global_seq, current_time
            )
            return None

        # Check for existing entry
        async with self._db.execute(
            "SELECT global_seq FROM intent_dedup WHERE world_id = ? AND agent_id = ? AND req_id = ?",
            (world_id, agent_id, req_id),
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            existing_seq = row[0]
            self._logger.info(
                "Intent already processed (duplicate)",
                world_id=world_id,
                agent_id=agent_id,
                req_id=req_id,
                existing_seq=existing_seq,
                new_seq=global_seq,
            )
            return int(existing_seq)

        # Record new intent
        await self._record_intent(world_id, agent_id, req_id, global_seq, current_time)
        return None

    async def _record_intent(
        self,
        world_id: str,
        agent_id: str,
        req_id: str,
        global_seq: int,
        created_at: float,
    ) -> None:
        """Record intent in deduplication store."""
        if self._db is not None:
            await self._db.execute(
                """
            INSERT OR REPLACE INTO intent_dedup
            (world_id, agent_id, req_id, global_seq, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
                (world_id, agent_id, req_id, global_seq, created_at),
            )
            await self._db.commit()

        self._logger.debug(
            "Intent recorded for deduplication",
            world_id=world_id,
            agent_id=agent_id,
            req_id=req_id,
            global_seq=global_seq,
        )

    async def cleanup_expired(self) -> int:
        """Clean up expired deduplication entries.

        Returns:
            Number of entries cleaned up
        """
        if not self._db:
            return 0

        current_time = time.time()
        cutoff_time = current_time - self.dedup_ttl_seconds

        # Delete expired entries
        async with self._db.execute(
            "DELETE FROM intent_dedup WHERE created_at < ?",
            (cutoff_time,),
        ) as cursor:
            deleted_count = cursor.rowcount

        await self._db.commit()

        if deleted_count > 0:
            self._logger.info(
                "Cleaned up expired deduplication entries",
                deleted_count=deleted_count,
                cutoff_time=cutoff_time,
            )

        return deleted_count

    async def cleanup_excess_entries(self) -> int:
        """Clean up excess entries when max_entries is exceeded.

        Returns:
            Number of entries cleaned up
        """
        if not self._db:
            return 0

        # Count current entries
        async with self._db.execute("SELECT COUNT(*) FROM intent_dedup") as cursor:
            row = await cursor.fetchone()
            current_count = row[0] if row else 0

        if current_count <= self.max_entries:
            return 0

        # Delete oldest entries to get back to max_entries
        entries_to_delete = current_count - self.max_entries

        async with self._db.execute(
            """
            DELETE FROM intent_dedup
            WHERE rowid IN (
                SELECT rowid FROM intent_dedup
                ORDER BY created_at ASC
                LIMIT ?
            )
            """,
            (entries_to_delete,),
        ) as cursor:
            deleted_count = cursor.rowcount

        await self._db.commit()

        if deleted_count > 0:
            self._logger.info(
                "Cleaned up excess deduplication entries",
                deleted_count=deleted_count,
                current_count=current_count,
                max_entries=self.max_entries,
            )

        return deleted_count

    async def get_stats(self) -> dict[str, Any]:
        """Get deduplication store statistics.

        Returns:
            Dictionary with store statistics
        """
        if not self._db:
            return {"status": "not_initialized"}

        current_time = time.time()

        # Count total entries
        async with self._db.execute("SELECT COUNT(*) FROM intent_dedup") as cursor:
            row = await cursor.fetchone()
            total_entries = row[0] if row else 0

        # Count entries by age
        cutoff_time = current_time - self.dedup_ttl_seconds
        async with self._db.execute(
            "SELECT COUNT(*) FROM intent_dedup WHERE created_at >= ?",
            (cutoff_time,),
        ) as cursor:
            row = await cursor.fetchone()
            active_entries = row[0] if row else 0

        expired_entries = total_entries - active_entries

        # Check if in warmup period
        in_warmup = (current_time - self._startup_time) < self.warmup_duration_seconds
        warmup_remaining = max(
            0, self.warmup_duration_seconds - (current_time - self._startup_time)
        )

        return {
            "status": "initialized",
            "total_entries": total_entries,
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "max_entries": self.max_entries,
            "dedup_ttl_seconds": self.dedup_ttl_seconds,
            "in_warmup": in_warmup,
            "warmup_remaining_seconds": warmup_remaining,
            "startup_time": self._startup_time,
        }

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired and excess entries."""
        self._logger.info("Starting deduplication cleanup loop")

        try:
            while True:
                await asyncio.sleep(self.cleanup_interval_seconds)

                try:
                    # Clean up expired entries
                    expired_cleaned = await self.cleanup_expired()

                    # Clean up excess entries
                    excess_cleaned = await self.cleanup_excess_entries()

                    if expired_cleaned > 0 or excess_cleaned > 0:
                        self._logger.info(
                            "Cleanup cycle completed",
                            expired_cleaned=expired_cleaned,
                            excess_cleaned=excess_cleaned,
                        )

                except Exception as e:
                    self._logger.error(
                        "Error during cleanup cycle",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        except asyncio.CancelledError:
            self._logger.info("Cleanup loop cancelled")
            raise
        except Exception as e:
            self._logger.error(
                "Cleanup loop failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    async def __aenter__(self) -> "DedupStore":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()


class InMemoryDedupStore:
    """In-memory deduplication store for testing.

    Provides the same interface as DedupStore but uses in-memory storage
    for faster testing without SQLite overhead.
    """

    def __init__(
        self,
        dedup_ttl_minutes: int | float = 60,
        max_entries: int = 10000,
        warmup_duration_minutes: int | float = 5,
    ):
        """Initialize in-memory deduplication store.

        Args:
            dedup_ttl_minutes: TTL for deduplication entries in minutes
            max_entries: Maximum number of entries before cleanup
            warmup_duration_minutes: Duration for relaxed deduplication after startup
        """
        self.dedup_ttl_seconds = dedup_ttl_minutes * 60
        self.max_entries = max_entries
        self.warmup_duration_seconds = warmup_duration_minutes * 60

        self._entries: dict[
            tuple[str, str, str], tuple[int, float]
        ] = {}  # (world_id, agent_id, req_id) -> (global_seq, created_at)
        self._startup_time = time.time()
        self._logger = get_logger("gunn.storage.dedup.memory")

    async def initialize(self) -> None:
        """Initialize store (no-op for in-memory)."""
        self._logger.info("InMemoryDedupStore initialized")

    async def close(self) -> None:
        """Close store (no-op for in-memory)."""
        self._entries.clear()
        self._logger.info("InMemoryDedupStore closed")

    async def check_and_record(
        self, world_id: str, agent_id: str, req_id: str, global_seq: int
    ) -> int | None:
        """Check if intent was already processed and record if not."""
        # Validate inputs
        if not world_id.strip():
            raise ValueError("world_id cannot be empty")
        if not agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        if not req_id.strip():
            raise ValueError("req_id cannot be empty")

        current_time = time.time()
        key = (world_id, agent_id, req_id)

        # Check if we're in warmup period
        in_warmup = (current_time - self._startup_time) < self.warmup_duration_seconds

        if in_warmup:
            # Still record the intent for future deduplication
            self._entries[key] = (global_seq, current_time)
            return None

        # Check for existing entry
        if key in self._entries:
            existing_seq, created_at = self._entries[key]

            # Check if entry is still valid (not expired)
            if current_time - created_at <= self.dedup_ttl_seconds:
                return existing_seq
            else:
                # Entry expired, remove it
                del self._entries[key]

        # Record new intent
        self._entries[key] = (global_seq, current_time)
        return None

    async def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        current_time = time.time()
        cutoff_time = current_time - self.dedup_ttl_seconds

        expired_keys = [
            key
            for key, (_, created_at) in self._entries.items()
            if created_at < cutoff_time
        ]

        for key in expired_keys:
            del self._entries[key]

        return len(expired_keys)

    async def cleanup_excess_entries(self) -> int:
        """Clean up excess entries when max_entries is exceeded."""
        if len(self._entries) <= self.max_entries:
            return 0

        # Sort by creation time and remove oldest
        sorted_entries = sorted(self._entries.items(), key=lambda x: x[1][1])
        entries_to_delete = len(self._entries) - self.max_entries

        for i in range(entries_to_delete):
            key = sorted_entries[i][0]
            del self._entries[key]

        return entries_to_delete

    async def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        current_time = time.time()
        cutoff_time = current_time - self.dedup_ttl_seconds

        active_entries = sum(
            1 for _, created_at in self._entries.values() if created_at >= cutoff_time
        )

        expired_entries = len(self._entries) - active_entries
        in_warmup = (current_time - self._startup_time) < self.warmup_duration_seconds
        warmup_remaining = max(
            0, self.warmup_duration_seconds - (current_time - self._startup_time)
        )

        return {
            "status": "initialized",
            "total_entries": len(self._entries),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "max_entries": self.max_entries,
            "dedup_ttl_seconds": self.dedup_ttl_seconds,
            "in_warmup": in_warmup,
            "warmup_remaining_seconds": warmup_remaining,
            "startup_time": self._startup_time,
        }

    async def __aenter__(self) -> "InMemoryDedupStore":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
