"""Event log implementation with append-only storage and integrity checking."""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from gunn.schemas.messages import EventLogEntry
from gunn.schemas.types import Effect
from gunn.utils.hashing import chain_checksum
from gunn.utils.telemetry import OperationTimer, create_logger

logger = create_logger(__name__)


class EventLog:
    """Append-only event log with hash chain integrity checking.
    
    This class provides thread-safe append operations and maintains a hash chain
    for integrity verification. It supports replay functionality and deterministic
    ordering for reproducible simulations.
    """

    def __init__(self, file_path: Optional[Path] = None, world_seed: Optional[int] = None):
        """Initialize the event log.
        
        Args:
            file_path: Optional file path for persistent storage
            world_seed: Optional seed for deterministic replay
        """
        self._entries: List[EventLogEntry] = []
        self._seq_counter: int = 0
        self._lock = asyncio.Lock()
        self._file_path = file_path
        self._world_seed = world_seed or random.randint(0, 2**32 - 1)
        
        # Set the random seed for deterministic behavior
        random.seed(self._world_seed)
        
        logger.info(
            "EventLog initialized",
            file_path=str(file_path) if file_path else None,
            world_seed=self._world_seed,
        )

    @property
    def world_seed(self) -> int:
        """Get the world seed for deterministic replay."""
        return self._world_seed

    @property
    def entry_count(self) -> int:
        """Get the number of entries in the log."""
        return len(self._entries)

    async def append(self, effect: Effect, req_id: Optional[str] = None) -> int:
        """Append an effect to the log with hash chain checksum.
        
        Args:
            effect: The effect to append
            req_id: Optional request ID for idempotency checking
            
        Returns:
            The global sequence number assigned to this entry
            
        Raises:
            ValueError: If effect is missing required fields
        """
        with OperationTimer("event_log_append"):
            async with self._lock:
                # Validate effect structure
                required_fields = ["kind", "payload", "source_id", "schema_version"]
                for field in required_fields:
                    if field not in effect:
                        raise ValueError(f"Effect must contain '{field}' field")

                # Calculate checksum
                prev_checksum = self._entries[-1].checksum if self._entries else ""
                checksum = chain_checksum(effect, prev_checksum)

                # Create entry
                entry = EventLogEntry(
                    global_seq=effect["global_seq"],
                    sim_time=effect["sim_time"],
                    wall_time=time.time(),
                    effect=effect,
                    source_metadata={"world_seed": self._world_seed},
                    checksum=checksum,
                    req_id=req_id or "",
                )

                self._entries.append(entry)
                
                # Persist to file if configured
                if self._file_path:
                    await self._persist_entry(entry)

                logger.debug(
                    "Effect appended to log",
                    global_seq=effect["global_seq"],
                    effect_kind=effect["kind"],
                    source_id=effect["source_id"],
                    req_id=req_id,
                    checksum=checksum[:16] + "...",  # Truncate for readability
                )

                return effect["global_seq"]

    async def get_entries_since(self, since_seq: int) -> List[EventLogEntry]:
        """Get entries for replay and catch-up.
        
        Args:
            since_seq: Get entries with global_seq > since_seq
            
        Returns:
            List of entries since the specified sequence number
        """
        with OperationTimer("event_log_get_entries"):
            async with self._lock:
                result = [
                    entry for entry in self._entries 
                    if entry.global_seq > since_seq
                ]
                
                logger.debug(
                    "Retrieved entries for replay",
                    since_seq=since_seq,
                    entry_count=len(result),
                )
                
                return result

    async def get_entries_range(self, from_seq: int, to_seq: Optional[int] = None) -> List[EventLogEntry]:
        """Get entries in a specific range.
        
        Args:
            from_seq: Starting sequence number (inclusive)
            to_seq: Ending sequence number (inclusive), or None for latest
            
        Returns:
            List of entries in the specified range
        """
        with OperationTimer("event_log_get_range"):
            async with self._lock:
                if to_seq is None:
                    to_seq = self._entries[-1].global_seq if self._entries else from_seq

                result = [
                    entry for entry in self._entries 
                    if from_seq <= entry.global_seq <= to_seq
                ]
                
                logger.debug(
                    "Retrieved entries for range",
                    from_seq=from_seq,
                    to_seq=to_seq,
                    entry_count=len(result),
                )
                
                return result

    async def validate_integrity(self) -> bool:
        """Validate the integrity of the entire log using hash chains.
        
        Returns:
            True if the log integrity is valid, False otherwise
        """
        with OperationTimer("event_log_validate"):
            async with self._lock:
                if not self._entries:
                    return True

                prev_checksum = ""
                for i, entry in enumerate(self._entries):
                    expected_checksum = chain_checksum(entry.effect, prev_checksum)
                    
                    if entry.checksum != expected_checksum:
                        logger.error(
                            "Log integrity violation detected",
                            entry_index=i,
                            global_seq=entry.global_seq,
                            expected_checksum=expected_checksum[:16] + "...",
                            actual_checksum=entry.checksum[:16] + "...",
                        )
                        return False
                    
                    prev_checksum = entry.checksum

                logger.info(
                    "Log integrity validation passed",
                    entry_count=len(self._entries),
                )
                
                return True

    async def load_from_file(self, file_path: Path) -> None:
        """Load entries from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing log entries
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Log file not found: {file_path}")

        async with self._lock:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract metadata
                if "metadata" in data:
                    self._world_seed = data["metadata"].get("world_seed", self._world_seed)
                    random.seed(self._world_seed)
                
                # Load entries
                entries_data = data.get("entries", [])
                self._entries = [EventLogEntry(**entry_data) for entry_data in entries_data]
                
                logger.info(
                    "Loaded log from file",
                    file_path=str(file_path),
                    entry_count=len(self._entries),
                    world_seed=self._world_seed,
                )
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ValueError(f"Invalid log file format: {e}")

    async def save_to_file(self, file_path: Path) -> None:
        """Save the entire log to a JSON file.
        
        Args:
            file_path: Path where to save the log file
        """
        async with self._lock:
            # Prepare data structure
            data = {
                "metadata": {
                    "world_seed": self._world_seed,
                    "created_at": time.time(),
                    "entry_count": len(self._entries),
                },
                "entries": [entry.model_dump() for entry in self._entries],
            }
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(
                "Saved log to file",
                file_path=str(file_path),
                entry_count=len(self._entries),
            )

    async def _persist_entry(self, entry: EventLogEntry) -> None:
        """Persist a single entry to file (append mode).
        
        Args:
            entry: The entry to persist
        """
        if not self._file_path:
            return
            
        # For simplicity, we'll append to a JSONL file
        jsonl_path = self._file_path.with_suffix('.jsonl')
        
        # Ensure directory exists
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append entry as JSON line
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(entry.model_dump()) + '\n')

    def get_latest_seq(self) -> int:
        """Get the latest global sequence number.
        
        Returns:
            Latest global_seq, or -1 if log is empty
        """
        if not self._entries:
            return -1
        return self._entries[-1].global_seq

    def __len__(self) -> int:
        """Get the number of entries in the log."""
        return len(self._entries)

    def __repr__(self) -> str:
        """String representation of the event log."""
        return (
            f"EventLog(entries={len(self._entries)}, "
            f"latest_seq={self.get_latest_seq()}, "
            f"world_seed={self._world_seed})"
        )