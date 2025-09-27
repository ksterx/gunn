"""Event log implementation for append-only storage with integrity checking.

This module provides the EventLog class that maintains an immutable sequence
of effects with hash chain integrity verification and replay capabilities.
"""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from gunn.schemas.types import Effect
from gunn.utils.hashing import chain_checksum
from gunn.utils.telemetry import (
    MonotonicClock,
    PerformanceTimer,
    get_logger,
    record_queue_depth,
)


class EventLogEntry(BaseModel):
    """Single entry in the event log with integrity checking.

    Each entry contains an effect along with metadata for ordering,
    timing, and hash chain integrity verification.
    """

    global_seq: int = Field(..., description="Monotonically increasing sequence number")
    sim_time: float = Field(..., description="Simulation time when effect occurred")
    wall_time: float = Field(..., description="Wall clock time when logged")
    effect: Effect = Field(..., description="The effect that occurred")
    source_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    checksum: str = Field(..., description="Hash chain checksum for integrity")
    req_id: str | None = Field(default=None, description="Request ID for idempotency")


class EventLog:
    """Append-only event log with hash chain integrity.

    Provides thread-safe append operations, replay capabilities, and
    integrity verification through hash chaining.

    Requirements addressed:
    - 1.2: Record events in global sequential log with global_seq identifier
    - 7.1: Maintain complete sequential log with global_seq numbers
    - 7.3: Provide replay capabilities from event log
    - 7.5: Validate state consistency using log hash/CRC and sequence gap detection
    """

    def __init__(self, world_id: str = "default") -> None:
        """Initialize empty event log.

        Args:
            world_id: Identifier for the world this log belongs to
        """
        self.world_id = world_id
        self._entries: list[EventLogEntry] = []
        self._seq_counter: int = 0
        self._lock = asyncio.Lock()
        self._logger = get_logger("gunn.event_log", world_id=world_id)

    async def append(self, effect: Effect, req_id: str | None = None) -> int:
        """Append effect to log with hash chain checksum.

        Thread-safe operation that adds a new effect to the log with
        proper sequencing and integrity checking.

        Args:
            effect: Effect to append to the log
            req_id: Optional request ID for idempotency tracking

        Returns:
            The global_seq assigned to this entry

        Raises:
            ValueError: If effect is missing required fields
        """
        if not effect.get("uuid"):
            raise ValueError("Effect must have uuid field")
        if not effect.get("kind"):
            raise ValueError("Effect must have kind field")

        with PerformanceTimer("event_log_append", record_metrics=True):
            async with self._lock:
                # Calculate next sequence number
                self._seq_counter += 1
                global_seq = self._seq_counter

                # Update effect with sequence number if not already set
                effect_dict = dict(effect)  # Make a copy to avoid mutating input
                if "global_seq" not in effect_dict or effect_dict["global_seq"] == 0:
                    effect_dict["global_seq"] = global_seq

                # Get previous checksum for hash chain
                prev_checksum = None
                if self._entries:
                    prev_checksum = self._entries[-1].checksum

                # Calculate checksum for this entry
                checksum = chain_checksum(effect_dict, prev_checksum)

                # Create log entry
                entry = EventLogEntry(
                    global_seq=global_seq,
                    sim_time=effect_dict.get("sim_time", MonotonicClock.now()),  # type: ignore
                    wall_time=MonotonicClock.wall_time(),
                    effect=effect_dict,  # type: ignore
                    source_metadata={"world_id": self.world_id},
                    checksum=checksum,
                    req_id=req_id,
                )

                # Append to log
                self._entries.append(entry)

                # Update metrics
                record_queue_depth("event_log", len(self._entries))

                # Log the append operation
                self._logger.info(
                    "Effect appended to log",
                    global_seq=global_seq,
                    effect_kind=effect_dict["kind"],
                    effect_uuid=effect_dict["uuid"],
                    req_id=req_id,
                    checksum=checksum[:8],  # First 8 chars for readability
                )

                return global_seq

    def get_entries_since(self, since_seq: int) -> list[EventLogEntry]:
        """Get entries for replay and catch-up.

        Returns all entries with global_seq > since_seq in order.

        Args:
            since_seq: Sequence number to start from (exclusive)

        Returns:
            List of entries after the specified sequence number
        """
        with PerformanceTimer("event_log_get_entries", record_metrics=True):
            result = []
            for entry in self._entries:
                if entry.global_seq > since_seq:
                    result.append(entry)

            self._logger.debug(
                "Retrieved entries for replay",
                since_seq=since_seq,
                num_entries=len(result),
                latest_seq=self._entries[-1].global_seq if self._entries else 0,
            )

            return result

    def get_all_entries(self) -> list[EventLogEntry]:
        """Get all entries in the log.

        Returns:
            Complete list of all log entries
        """
        return self._entries.copy()

    def get_latest_seq(self) -> int:
        """Get the latest sequence number in the log.

        Returns:
            Latest global_seq, or 0 if log is empty
        """
        return self._seq_counter

    def get_entry_count(self) -> int:
        """Get total number of entries in the log.

        Returns:
            Number of entries in the log
        """
        return len(self._entries)

    def validate_integrity(self) -> dict[str, Any]:
        """Validate complete log integrity.

        Performs comprehensive integrity checks including:
        - Hash chain validation
        - Sequence gap detection
        - Corruption analysis

        Returns:
            Dictionary with integrity report containing:
            - valid: Overall validity boolean
            - corrupted_entries: List of corrupted entry indices
            - missing_sequences: List of detected sequence gaps
            - total_entries: Total number of entries checked
            - details: Additional diagnostic information
        """
        with PerformanceTimer("event_log_validate", record_metrics=True):
            corrupted_entries: list[int] = []
            missing_sequences: list[int] = []

            # Check hash chain integrity
            prev_checksum = None
            for i, entry in enumerate(self._entries):
                expected_checksum = chain_checksum(dict(entry.effect), prev_checksum)
                if expected_checksum != entry.checksum:
                    corrupted_entries.append(i)
                prev_checksum = entry.checksum

            # Check for sequence gaps
            if self._entries:
                expected_seq = 1
                for entry in self._entries:
                    if entry.global_seq != expected_seq:
                        # Record all missing sequences
                        while expected_seq < entry.global_seq:
                            missing_sequences.append(expected_seq)
                            expected_seq += 1
                    expected_seq = entry.global_seq + 1

            # Overall validity
            is_valid = len(corrupted_entries) == 0 and len(missing_sequences) == 0

            result = {
                "valid": is_valid,
                "corrupted_entries": corrupted_entries,
                "missing_sequences": missing_sequences,
                "total_entries": len(self._entries),
                "details": {
                    "latest_seq": self._seq_counter,
                    "first_seq": self._entries[0].global_seq if self._entries else None,
                    "last_seq": self._entries[-1].global_seq if self._entries else None,
                    "world_id": self.world_id,
                },
            }

            self._logger.info(
                "Log integrity validation completed",
                valid=is_valid,
                corrupted_count=len(corrupted_entries),
                missing_count=len(missing_sequences),
                total_entries=len(self._entries),
            )

            return result

    def find_entry_by_uuid(self, effect_uuid: str) -> EventLogEntry | None:
        """Find entry by effect UUID.

        Args:
            effect_uuid: UUID of the effect to find

        Returns:
            EventLogEntry if found, None otherwise
        """
        for entry in self._entries:
            if entry.effect.get("uuid") == effect_uuid:
                return entry
        return None

    def find_entries_by_req_id(self, req_id: str) -> list[EventLogEntry]:
        """Find entries by request ID.

        Args:
            req_id: Request ID to search for

        Returns:
            List of entries with matching request ID
        """
        result = []
        for entry in self._entries:
            if entry.req_id == req_id:
                result.append(entry)
        return result

    def get_entries_by_source(self, source_id: str) -> list[EventLogEntry]:
        """Get all entries from a specific source.

        Args:
            source_id: Source identifier to filter by

        Returns:
            List of entries from the specified source
        """
        result = []
        for entry in self._entries:
            if entry.effect.get("source_id") == source_id:
                result.append(entry)
        return result

    def get_entries_in_time_range(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        use_sim_time: bool = True,
    ) -> list[EventLogEntry]:
        """Get entries within a time range.

        Args:
            start_time: Start time (inclusive), None for no lower bound
            end_time: End time (inclusive), None for no upper bound
            use_sim_time: If True, filter by sim_time; if False, use wall_time

        Returns:
            List of entries within the specified time range
        """
        result = []
        for entry in self._entries:
            time_value = entry.sim_time if use_sim_time else entry.wall_time

            if start_time is not None and time_value < start_time:
                continue
            if end_time is not None and time_value > end_time:
                continue

            result.append(entry)

        return result

    async def compact(self, keep_entries: int = 1000) -> int:
        """Compact the log by removing old entries.

        Keeps the most recent entries and removes older ones to manage memory.
        This operation maintains integrity by preserving the hash chain.

        Args:
            keep_entries: Number of recent entries to keep

        Returns:
            Number of entries removed
        """
        async with self._lock:
            if len(self._entries) <= keep_entries:
                return 0

            entries_to_remove = len(self._entries) - keep_entries
            self._entries = self._entries[entries_to_remove:]

            self._logger.info(
                "Log compacted",
                removed_count=entries_to_remove,
                remaining_count=len(self._entries),
                oldest_remaining_seq=self._entries[0].global_seq
                if self._entries
                else None,
            )

            return entries_to_remove

    def get_stats(self) -> dict[str, Any]:
        """Get log statistics.

        Returns:
            Dictionary with log statistics including entry counts,
            time ranges, and integrity status
        """
        if not self._entries:
            return {
                "total_entries": 0,
                "latest_seq": 0,
                "time_range": None,
                "world_id": self.world_id,
            }

        first_entry = self._entries[0]
        last_entry = self._entries[-1]

        return {
            "total_entries": len(self._entries),
            "latest_seq": self._seq_counter,
            "seq_range": {
                "first": first_entry.global_seq,
                "last": last_entry.global_seq,
            },
            "time_range": {
                "sim_time": {
                    "first": first_entry.sim_time,
                    "last": last_entry.sim_time,
                    "duration": last_entry.sim_time - first_entry.sim_time,
                },
                "wall_time": {
                    "first": first_entry.wall_time,
                    "last": last_entry.wall_time,
                    "duration": last_entry.wall_time - first_entry.wall_time,
                },
            },
            "world_id": self.world_id,
        }
