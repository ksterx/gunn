"""Memory management and compaction utilities for the simulation core.

This module provides memory management capabilities including WorldState snapshots,
log compaction, view cache eviction with LRU policy, and memory usage tracking.
"""

import asyncio
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from gunn.schemas.messages import WorldState
from gunn.utils.telemetry import get_logger, record_queue_depth

if TYPE_CHECKING:
    from gunn.core.event_log import EventLog


class MemoryStats(BaseModel):
    """Memory usage statistics."""

    total_log_entries: int = Field(..., description="Total number of log entries")
    total_snapshots: int = Field(..., description="Total number of snapshots")
    view_cache_size: int = Field(..., description="View cache size")
    estimated_memory_mb: float = Field(..., description="Estimated memory usage in MB")
    oldest_entry_age_seconds: float = Field(
        ..., description="Oldest entry age in seconds"
    )
    newest_entry_age_seconds: float = Field(
        ..., description="Newest entry age in seconds"
    )
    compaction_eligible_entries: int = Field(
        ..., description="Number of compaction eligible entries"
    )


class MemoryConfig(BaseModel):
    """Configuration for memory management."""

    max_log_entries: int = Field(10000, description="Maximum number of log entries")
    view_cache_size: int = Field(1000, description="Maximum size of view cache")
    compaction_threshold: int = Field(
        5000, description="Threshold for triggering log compaction"
    )
    snapshot_interval: int = Field(1000, description="Create snapshot every N events")
    max_snapshots: int = Field(10, description="Maximum number of snapshots to keep")
    memory_check_interval_seconds: float = Field(
        60.0, description="Interval between memory checks"
    )
    auto_compaction_enabled: bool = Field(
        True, description="Enable automatic log compaction"
    )


class WorldStateSnapshot(BaseModel):
    """Snapshot of world state at a specific point in time."""

    global_seq: int = Field(..., description="Global sequence number for this snapshot")
    sim_time: float = Field(..., description="Simulation time for this snapshot")
    wall_time: float = Field(..., description="Wall time for this snapshot")
    world_state: WorldState = Field(..., description="World state for this snapshot")
    checksum: str = Field(..., description="Checksum for this snapshot")


class ViewCache:
    """LRU cache for agent views with configurable size limits."""

    def __init__(self, max_size: int = 1000):
        """Initialize view cache with maximum size.

        Args:
            max_size: Maximum number of cached views
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._access_times: dict[str, float] = {}
        self._logger = get_logger("gunn.memory.view_cache")

    def get(self, key: str) -> Any:
        """Get cached view and mark as recently used.

        Args:
            key: Cache key (typically agent_id:view_seq)

        Returns:
            Cached view or None if not found
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._access_times[key] = time.time()
            return self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Store view in cache with LRU eviction.

        Args:
            key: Cache key
            value: View to cache
        """
        current_time = time.time()

        if key in self._cache:
            # Update existing entry
            self._cache[key] = value
            self._cache.move_to_end(key)
            self._access_times[key] = current_time
        else:
            # Add new entry
            if len(self._cache) >= self.max_size:
                # Evict least recently used
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._access_times[oldest_key]

                self._logger.debug(
                    "Evicted LRU cache entry",
                    evicted_key=oldest_key,
                    cache_size=len(self._cache),
                )

            self._cache[key] = value
            self._access_times[key] = current_time

    def evict_old_entries(self, max_age_seconds: float = 3600.0) -> int:
        """Evict entries older than specified age.

        Args:
            max_age_seconds: Maximum age for cached entries

        Returns:
            Number of entries evicted
        """
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds

        keys_to_evict = []
        for key, access_time in self._access_times.items():
            if access_time < cutoff_time:
                keys_to_evict.append(key)

        for key in keys_to_evict:
            del self._cache[key]
            del self._access_times[key]

        if keys_to_evict:
            self._logger.info(
                "Evicted old cache entries",
                evicted_count=len(keys_to_evict),
                max_age_seconds=max_age_seconds,
                remaining_size=len(self._cache),
            )

        return len(keys_to_evict)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_times.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        access_times = list(self._access_times.values())

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0,
            "oldest_access_age": current_time - min(access_times)
            if access_times
            else 0,
            "newest_access_age": current_time - max(access_times)
            if access_times
            else 0,
        }


class SnapshotManager:
    """Manages WorldState snapshots for faster replay."""

    def __init__(self, config: MemoryConfig):
        """Initialize snapshot manager.

        Args:
            config: Memory configuration
        """
        self.config = config
        self._snapshots: list[WorldStateSnapshot] = []
        self._lock = asyncio.Lock()
        self._logger = get_logger("gunn.memory.snapshots")

    async def create_snapshot(
        self, global_seq: int, world_state: WorldState, sim_time: float = 0.0
    ) -> WorldStateSnapshot:
        """Create a new WorldState snapshot.

        Args:
            global_seq: Global sequence number for this snapshot
            world_state: Current world state to snapshot
            sim_time: Simulation time

        Returns:
            Created snapshot
        """
        import hashlib

        from gunn.utils.hashing import canonical_json

        # Create snapshot with integrity checksum
        snapshot_data = {
            "global_seq": global_seq,
            "world_state": world_state.model_dump(),
            "sim_time": sim_time,
        }

        checksum = hashlib.sha256(canonical_json(snapshot_data)).hexdigest()

        snapshot = WorldStateSnapshot(
            global_seq=global_seq,
            sim_time=sim_time,
            wall_time=time.time(),
            world_state=world_state,
            checksum=checksum,
        )

        async with self._lock:
            self._snapshots.append(snapshot)

            # Maintain maximum number of snapshots
            if len(self._snapshots) > self.config.max_snapshots:
                removed = self._snapshots.pop(0)
                self._logger.debug(
                    "Removed old snapshot",
                    removed_seq=removed.global_seq,
                    remaining_count=len(self._snapshots),
                )

            self._logger.info(
                "Created WorldState snapshot",
                global_seq=global_seq,
                sim_time=sim_time,
                checksum=checksum[:8],
                total_snapshots=len(self._snapshots),
            )

        return snapshot

    def find_nearest_snapshot(self, target_seq: int) -> WorldStateSnapshot | None:
        """Find the snapshot closest to but not exceeding target sequence.

        Args:
            target_seq: Target global sequence number

        Returns:
            Nearest snapshot or None if no suitable snapshot exists
        """
        best_snapshot = None

        for snapshot in self._snapshots:
            if snapshot.global_seq <= target_seq:
                if (
                    best_snapshot is None
                    or snapshot.global_seq > best_snapshot.global_seq
                ):
                    best_snapshot = snapshot

        return best_snapshot

    def get_all_snapshots(self) -> list[WorldStateSnapshot]:
        """Get all snapshots ordered by global_seq.

        Returns:
            List of all snapshots
        """
        return sorted(self._snapshots, key=lambda s: s.global_seq)

    async def validate_snapshot_integrity(self, snapshot: WorldStateSnapshot) -> bool:
        """Validate snapshot integrity using checksum.

        Args:
            snapshot: Snapshot to validate

        Returns:
            True if snapshot is valid
        """
        import hashlib

        from gunn.utils.hashing import canonical_json

        snapshot_data = {
            "global_seq": snapshot.global_seq,
            "world_state": snapshot.world_state.model_dump(),
            "sim_time": snapshot.sim_time,
        }

        expected_checksum = hashlib.sha256(canonical_json(snapshot_data)).hexdigest()
        is_valid = expected_checksum == snapshot.checksum

        if not is_valid:
            self._logger.warning(
                "Snapshot integrity check failed",
                global_seq=snapshot.global_seq,
                expected_checksum=expected_checksum[:8],
                actual_checksum=snapshot.checksum[:8],
            )

        return is_valid

    def get_stats(self) -> dict[str, Any]:
        """Get snapshot statistics.

        Returns:
            Dictionary with snapshot statistics
        """
        if not self._snapshots:
            return {
                "total_snapshots": 0,
                "oldest_seq": None,
                "newest_seq": None,
                "time_range": None,
            }

        oldest = min(self._snapshots, key=lambda s: s.global_seq)
        newest = max(self._snapshots, key=lambda s: s.global_seq)

        return {
            "total_snapshots": len(self._snapshots),
            "oldest_seq": oldest.global_seq,
            "newest_seq": newest.global_seq,
            "seq_range": newest.global_seq - oldest.global_seq,
            "time_range": {
                "oldest_wall_time": oldest.wall_time,
                "newest_wall_time": newest.wall_time,
                "duration_seconds": newest.wall_time - oldest.wall_time,
            },
        }


class MemoryManager:
    """Central memory management for the simulation core.

    Handles WorldState snapshots, log compaction, view cache eviction,
    and memory usage monitoring with configurable limits.

    Requirements addressed:
    - 7.3: WorldState snapshot creation every N events for faster replay
    - 11.4: Memory limits and compaction correctness
    """

    def __init__(self, config: MemoryConfig):
        """Initialize memory manager with configuration.

        Args:
            config: Memory management configuration
        """
        self.config = config
        self.view_cache = ViewCache(config.view_cache_size)
        self.snapshot_manager = SnapshotManager(config)
        self._last_snapshot_seq = 0
        self._last_memory_check = 0.0
        self._compaction_in_progress = False
        self._lock = asyncio.Lock()
        self._logger = get_logger("gunn.memory.manager")

        self._logger.info(
            "MemoryManager initialized",
            max_log_entries=config.max_log_entries,
            view_cache_size=config.view_cache_size,
            compaction_threshold=config.compaction_threshold,
            snapshot_interval=config.snapshot_interval,
        )

    async def check_and_create_snapshot(
        self, current_seq: int, world_state: WorldState, sim_time: float = 0.0
    ) -> WorldStateSnapshot | None:
        """Check if a snapshot should be created and create it if needed.

        Args:
            current_seq: Current global sequence number
            world_state: Current world state
            sim_time: Current simulation time

        Returns:
            Created snapshot or None if no snapshot was needed
        """
        if current_seq - self._last_snapshot_seq >= self.config.snapshot_interval:
            snapshot = await self.snapshot_manager.create_snapshot(
                current_seq, world_state, sim_time
            )
            self._last_snapshot_seq = current_seq
            return snapshot

        return None

    async def compact_log(self, event_log: "EventLog") -> int:
        """Compact old log entries while preserving replay capability.

        Removes old entries from the log while ensuring that replay capability
        is preserved through snapshots. Only compacts if there are sufficient
        snapshots to maintain replay integrity.

        Args:
            event_log: Event log to compact

        Returns:
            Number of entries removed
        """
        if self._compaction_in_progress:
            self._logger.debug("Compaction already in progress, skipping")
            return 0

        async with self._lock:
            self._compaction_in_progress = True

            try:
                current_entries = event_log.get_entry_count()

                if current_entries <= self.config.compaction_threshold:
                    self._logger.debug(
                        "Log compaction not needed",
                        current_entries=current_entries,
                        threshold=self.config.compaction_threshold,
                    )
                    return 0

                # Find the latest snapshot to determine safe compaction point
                snapshots = self.snapshot_manager.get_all_snapshots()
                if not snapshots:
                    self._logger.warning(
                        "Cannot compact log without snapshots - replay capability would be lost"
                    )
                    return 0

                # Keep entries after the most recent snapshot
                latest_snapshot = max(snapshots, key=lambda s: s.global_seq)
                safe_compaction_seq = latest_snapshot.global_seq

                # Ensure we keep some entries after the snapshot for safety
                safety_margin = min(1000, self.config.snapshot_interval // 2)
                keep_entries_after_seq = safe_compaction_seq - safety_margin

                # Calculate how many entries to remove
                all_entries = event_log.get_all_entries()
                entries_to_remove = 0

                for entry in all_entries:
                    if entry.global_seq <= keep_entries_after_seq:
                        entries_to_remove += 1
                    else:
                        break

                if entries_to_remove == 0:
                    self._logger.debug("No entries eligible for compaction")
                    return 0

                # Perform compaction
                removed_count = await event_log.compact(
                    current_entries - entries_to_remove
                )

                self._logger.info(
                    "Log compaction completed",
                    removed_entries=removed_count,
                    remaining_entries=event_log.get_entry_count(),
                    safe_compaction_seq=safe_compaction_seq,
                    latest_snapshot_seq=latest_snapshot.global_seq,
                )

                return removed_count

            finally:
                self._compaction_in_progress = False

    def evict_old_views(self, max_age_seconds: float = 3600.0) -> int:
        """Remove old cached views to free memory.

        Args:
            max_age_seconds: Maximum age for cached views

        Returns:
            Number of views evicted
        """
        return self.view_cache.evict_old_entries(max_age_seconds)

    async def check_memory_limits(self, event_log: "EventLog") -> bool:
        """Check if memory limits are exceeded and trigger compaction if needed.

        Args:
            event_log: Event log to check

        Returns:
            True if memory is within limits
        """
        current_time = time.time()

        # Rate limit memory checks
        if (
            current_time - self._last_memory_check
            < self.config.memory_check_interval_seconds
        ):
            return True

        self._last_memory_check = current_time

        current_entries = event_log.get_entry_count()

        # Check if compaction is needed
        if (
            current_entries > self.config.max_log_entries
            and self.config.auto_compaction_enabled
        ):
            self._logger.info(
                "Memory limit exceeded, triggering compaction",
                current_entries=current_entries,
                max_entries=self.config.max_log_entries,
            )

            removed_count = await self.compact_log(event_log)

            if removed_count == 0:
                self._logger.warning(
                    "Compaction failed to remove entries - memory limit still exceeded"
                )
                return False

        # Update metrics
        record_queue_depth("memory_log_entries", event_log.get_entry_count())
        record_queue_depth("memory_view_cache", len(self.view_cache._cache))
        record_queue_depth("memory_snapshots", len(self.snapshot_manager._snapshots))

        return True

    def estimate_memory_usage(self, event_log: "EventLog") -> MemoryStats:
        """Return current memory usage statistics.

        Args:
            event_log: Event log to analyze

        Returns:
            Memory usage statistics
        """
        current_time = time.time()
        all_entries = event_log.get_all_entries()

        # Calculate ages
        oldest_age = 0.0
        newest_age = 0.0

        if all_entries:
            oldest_entry = min(all_entries, key=lambda e: e.wall_time)
            newest_entry = max(all_entries, key=lambda e: e.wall_time)
            oldest_age = current_time - oldest_entry.wall_time
            newest_age = current_time - newest_entry.wall_time

        # Estimate memory usage (rough calculation)
        # Each entry is approximately 1KB on average
        estimated_memory_mb = (
            len(all_entries) * 1.0  # Event log entries
            + len(self.view_cache._cache) * 0.5  # View cache
            + len(self.snapshot_manager._snapshots) * 10.0  # Snapshots (larger)
        ) / 1024.0

        # Count entries eligible for compaction
        compaction_eligible = 0
        if all_entries:
            snapshots = self.snapshot_manager.get_all_snapshots()
            if snapshots:
                latest_snapshot = max(snapshots, key=lambda s: s.global_seq)
                safety_margin = min(1000, self.config.snapshot_interval // 2)
                cutoff_seq = latest_snapshot.global_seq - safety_margin

                for entry in all_entries:
                    if entry.global_seq <= cutoff_seq:
                        compaction_eligible += 1

        return MemoryStats(
            total_log_entries=len(all_entries),
            total_snapshots=len(self.snapshot_manager._snapshots),
            view_cache_size=len(self.view_cache._cache),
            estimated_memory_mb=estimated_memory_mb,
            oldest_entry_age_seconds=oldest_age,
            newest_entry_age_seconds=newest_age,
            compaction_eligible_entries=compaction_eligible,
        )

    def get_detailed_stats(self, event_log: "EventLog") -> dict[str, Any]:
        """Get comprehensive memory management statistics.

        Args:
            event_log: Event log to analyze

        Returns:
            Detailed statistics dictionary
        """
        memory_stats = self.estimate_memory_usage(event_log)
        view_cache_stats = self.view_cache.get_stats()
        snapshot_stats = self.snapshot_manager.get_stats()

        return {
            "memory_usage": {
                "total_log_entries": memory_stats.total_log_entries,
                "total_snapshots": memory_stats.total_snapshots,
                "view_cache_size": memory_stats.view_cache_size,
                "estimated_memory_mb": memory_stats.estimated_memory_mb,
                "compaction_eligible_entries": memory_stats.compaction_eligible_entries,
            },
            "view_cache": view_cache_stats,
            "snapshots": snapshot_stats,
            "config": {
                "max_log_entries": self.config.max_log_entries,
                "view_cache_size": self.config.view_cache_size,
                "compaction_threshold": self.config.compaction_threshold,
                "snapshot_interval": self.config.snapshot_interval,
                "auto_compaction_enabled": self.config.auto_compaction_enabled,
            },
            "status": {
                "compaction_in_progress": self._compaction_in_progress,
                "last_snapshot_seq": self._last_snapshot_seq,
                "last_memory_check": self._last_memory_check,
            },
        }

    async def cleanup(self) -> None:
        """Cleanup memory manager resources."""
        self.view_cache.clear()
        self._logger.info("MemoryManager cleanup completed")
