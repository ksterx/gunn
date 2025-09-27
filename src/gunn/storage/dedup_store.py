"""Deduplication storage using SQLite for intent idempotency.

This module provides persistent storage for intent deduplication with TTL
cleanup and warmup guards for relaxed deduplication after restart.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from gunn.utils.telemetry import get_logger


class DedupStore:
    """SQLite-based deduplication store with TTL cleanup.

    Provides persistent storage for intent deduplication using tuple keys
    (world_id, agent_id, req_id) with configurable TTL and cleanup jobs.

    Requirements addressed:
    - 10.1: Idempotent ingestion where duplicate req_id MUST NOT create duplicate effects
    - 10.2: TTL cleanup for expired entries
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        dedup_ttl_minutes: int = 60,
        max_entries: int = 10000,
        cleanup_interval_minutes: int = 10,
        warmup_ttl_minutes: int = 5,
    ):
        """Initialize deduplication store.

        Args:
            db_path: Path to SQLite database file (":memory:" for in-memory)
            dedup_ttl_minutes: TTL for deduplication entries in minutes
            max_entries: Maximum entries before cleanup (N thousand entries)
            cleanup_interval_minutes: Interval between cleanup jobs
            warmup_ttl_minutes: TTL warmup guard duration after restart
        """
        self.db_path = str(db_path)
        self.dedup_ttl_seconds = dedup_ttl_minutes * 60
        self.max_entries = max_entries
        self.cleanup_interval_seconds = cleanup_interval_minutes * 60
        self.warmup_ttl_seconds = warmup_ttl_minutes * 60

        self._db: Optional[aiosqlite.Connection] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._startup_time = time.time()
        self._logger = get_logger("gunn.storage.dedup")

        self._logger.info(
            "DedupStore initialized",
            db_path=self.db_path,
            dedup_ttl_minutes=dedup_ttl_minutes,
            max_entries=max_entries,
            cleanup_interval_minutes=cleanup_interval_minutes,
            warmup_ttl_minutes=warmup_ttl_minutes,
        )

    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        if self._db is not None:
            return

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

        # Create index for cleanup queries
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
            self._cleanup_task = None

        if self._db:
            await self._db.close()
            self._db = None

        self._logger.info("DedupStore closed")

    async def check_and_record(
        self, world_id: str, agent_id: str, req_id: str, global_seq: int
    ) -> Optional[int]:
        """Check if request exists and record if new.

        Args:
            world_id: World identifier
            agent_id: Agent identifier
            req_id: Request identifier
            global_seq: Global sequence number for this request

        Returns:
            Existing global_seq if duplicate, None if new (and recorded)

        Raises:
            RuntimeError: If database is not initialized
        """
        if not self._db:
            raise RuntimeError("DedupStore not initialized")

        current_time = time.time()

        # Check for existing entry
        async with self._db.execute(
            "SELECT global_seq, created_at FROM intent_dedup WHERE world_id = ? AND agent_id = ? AND req_id = ?",
            (world_id, agent_id, req_id),
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            existing_seq, created_at = row
            age_seconds = current_time - created_at

            # Check if entry is still within TTL
            effective_ttl = self._get_effective_ttl(current_time)
            if age_seconds <= effective_ttl:
                self._logger.debug(
                    "Duplicate request found",
                    world_id=world_id,
                    agent_id=agent_id,
                    req_id=req_id,
                    existing_seq=existing_seq,
                    age_seconds=age_seconds,
                    effective_ttl=effective_ttl,
                )
                return existing_seq
            else:
                # Entry expired, remove it
                await self._db.execute(
                    "DELETE FROM intent_dedup WHERE world_id = ? AND agent_id = ? AND req_id = ?",
                    (world_id, agent_id, req_id),
                )
                self._logger.debug(
                    "Expired duplicate entry removed",
                    world_id=world_id,
                    agent_id=agent_id,
                    req_id=req_id,
                    age_seconds=age_seconds,
                )

        # Record new entry - handle race conditions with INSERT OR IGNORE
        try:
            await self._db.execute(
                "INSERT INTO intent_dedup (world_id, agent_id, req_id, global_seq, created_at) VALUES (?, ?, ?, ?, ?)",
                (world_id, agent_id, req_id, global_seq, current_time),
            )
            await self._db.commit()

            self._logger.debug(
                "New request recorded",
                world_id=world_id,
                agent_id=agent_id,
                req_id=req_id,
                global_seq=global_seq,
            )

            return None

        except Exception as e:
            # Handle race condition - another thread may have inserted the same key
            if "UNIQUE constraint failed" in str(e):
                # Re-check for the entry that was just inserted
                async with self._db.execute(
                    "SELECT global_seq FROM intent_dedup WHERE world_id = ? AND agent_id = ? AND req_id = ?",
                    (world_id, agent_id, req_id),
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        existing_seq = row[0]
                        self._logger.debug(
                            "Race condition detected, returning existing entry",
                            world_id=world_id,
                            agent_id=agent_id,
                            req_id=req_id,
                            existing_seq=existing_seq,
                        )
                        return existing_seq

            # Re-raise other exceptions
            raise

    async def cleanup_expired(self) -> int:
        """Clean up expired entries.

        Returns:
            Number of entries cleaned up
        """
        if not self._db:
            raise RuntimeError("DedupStore not initialized")

        current_time = time.time()
        effective_ttl = self._get_effective_ttl(current_time)
        cutoff_time = current_time - effective_ttl

        # Delete expired entries
        cursor = await self._db.execute(
            "DELETE FROM intent_dedup WHERE created_at < ?", (cutoff_time,)
        )
        deleted_count = cursor.rowcount
        await self._db.commit()

        if deleted_count > 0:
            self._logger.info(
                "Expired entries cleaned up",
                deleted_count=deleted_count,
                cutoff_time=cutoff_time,
                effective_ttl=effective_ttl,
            )

        return deleted_count

    async def cleanup_excess(self) -> int:
        """Clean up excess entries beyond max_entries limit.

        Returns:
            Number of entries cleaned up
        """
        if not self._db:
            return 0

        # Count total entries
        async with self._db.execute("SELECT COUNT(*) FROM intent_dedup") as cursor:
            row = await cursor.fetchone()
            total_count = row[0] if row else 0

        if total_count <= self.max_entries:
            return 0

        # Delete oldest entries beyond limit
        excess_count = total_count - self.max_entries
        cursor = await self._db.execute(
            "DELETE FROM intent_dedup WHERE rowid IN (SELECT rowid FROM intent_dedup ORDER BY created_at ASC LIMIT ?)",
            (excess_count,),
        )
        deleted_count = cursor.rowcount
        await self._db.commit()

        self._logger.info(
            "Excess entries cleaned up",
            total_count=total_count,
            max_entries=self.max_entries,
            deleted_count=deleted_count,
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

        # Get total count
        async with self._db.execute("SELECT COUNT(*) FROM intent_dedup") as cursor:
            row = await cursor.fetchone()
            total_count = row[0] if row else 0

        # Get count by age buckets
        effective_ttl = self._get_effective_ttl(current_time)
        recent_cutoff = current_time - (effective_ttl / 4)  # Last 25% of TTL

        async with self._db.execute(
            "SELECT COUNT(*) FROM intent_dedup WHERE created_at > ?", (recent_cutoff,)
        ) as cursor:
            row = await cursor.fetchone()
            recent_count = row[0] if row else 0

        # Get oldest and newest timestamps
        oldest_ts = None
        newest_ts = None

        async with self._db.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM intent_dedup"
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0] is not None:
                oldest_ts, newest_ts = row

        return {
            "status": "initialized",
            "total_entries": total_count,
            "recent_entries": recent_count,
            "max_entries": self.max_entries,
            "dedup_ttl_seconds": self.dedup_ttl_seconds,
            "effective_ttl_seconds": effective_ttl,
            "warmup_active": self._is_warmup_active(current_time),
            "oldest_entry_age": current_time - oldest_ts if oldest_ts else None,
            "newest_entry_age": current_time - newest_ts if newest_ts else None,
            "startup_time": self._startup_time,
        }

    def _get_effective_ttl(self, current_time: float) -> float:
        """Get effective TTL considering warmup period.

        During warmup period after restart, TTL is relaxed to avoid
        false positives from entries that might have been processed
        before restart.

        Args:
            current_time: Current timestamp

        Returns:
            Effective TTL in seconds
        """
        if self._is_warmup_active(current_time):
            # During warmup, use longer TTL for relaxed deduplication
            return self.dedup_ttl_seconds + self.warmup_ttl_seconds
        else:
            return self.dedup_ttl_seconds

    def _is_warmup_active(self, current_time: float) -> bool:
        """Check if warmup period is still active.

        Args:
            current_time: Current timestamp

        Returns:
            True if warmup is active
        """
        return (current_time - self._startup_time) < self.warmup_ttl_seconds

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        self._logger.info("Cleanup loop started")

        try:
            while True:
                await asyncio.sleep(self.cleanup_interval_seconds)

                try:
                    # Clean up expired entries
                    expired_count = await self.cleanup_expired()

                    # Clean up excess entries
                    excess_count = await self.cleanup_excess()

                    if expired_count > 0 or excess_count > 0:
                        self._logger.debug(
                            "Cleanup completed",
                            expired_count=expired_count,
                            excess_count=excess_count,
                        )

                except Exception as e:
                    self._logger.error("Cleanup failed", error=str(e))

        except asyncio.CancelledError:
            self._logger.info("Cleanup loop cancelled")
            raise
        except Exception as e:
            self._logger.error("Cleanup loop failed", error=str(e))
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
