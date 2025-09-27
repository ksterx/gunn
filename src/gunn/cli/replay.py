"""CLI utility for replaying event logs with determinism validation."""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from gunn.core.event_log import EventLog
from gunn.schemas.messages import EventLogEntry, WorldState
from gunn.schemas.types import Effect
from gunn.utils.telemetry import create_logger, setup_logging

logger = create_logger(__name__)


class ReplayEngine:
    """Engine for replaying event logs with determinism validation."""

    def __init__(self, log_file: Path, world_seed: Optional[int] = None):
        """Initialize the replay engine.
        
        Args:
            log_file: Path to the event log file
            world_seed: Optional seed override for determinism testing
        """
        self.log_file = log_file
        self.event_log = EventLog(world_seed=world_seed)
        self.world_state = WorldState()
        self._original_seed: Optional[int] = None
        self._seed_override = world_seed

    async def load_log(self) -> None:
        """Load the event log from file."""
        await self.event_log.load_from_file(self.log_file)
        self._original_seed = self.event_log.world_seed
        
        # If we had a seed override, apply it after loading
        if self._seed_override is not None:
            self.event_log._world_seed = self._seed_override
            random.seed(self._seed_override)
            logger.info(
                "Overriding world seed for replay",
                original_seed=self._original_seed,
                override_seed=self._seed_override,
            )
        
        logger.info(
            "Loaded event log for replay",
            file_path=str(self.log_file),
            entry_count=self.event_log.entry_count,
            world_seed=self.event_log.world_seed,
        )

    async def replay_range(self, from_seq: int = 0, to_seq: Optional[int] = None) -> List[EventLogEntry]:
        """Replay events in a specific range.
        
        Args:
            from_seq: Starting sequence number (inclusive)
            to_seq: Ending sequence number (inclusive), or None for latest
            
        Returns:
            List of replayed entries
        """
        entries = await self.event_log.get_entries_range(from_seq, to_seq)
        
        logger.info(
            "Starting replay",
            from_seq=from_seq,
            to_seq=to_seq or "latest",
            entry_count=len(entries),
        )

        # Reset world state for clean replay
        self.world_state = WorldState()
        
        # Set deterministic seed
        if self._original_seed is not None:
            random.seed(self._original_seed)

        replayed_entries = []
        for entry in entries:
            await self._apply_effect(entry.effect)
            replayed_entries.append(entry)
            
            logger.debug(
                "Replayed effect",
                global_seq=entry.global_seq,
                effect_kind=entry.effect["kind"],
                sim_time=entry.sim_time,
            )

        logger.info(
            "Replay completed",
            replayed_count=len(replayed_entries),
        )

        return replayed_entries

    async def validate_determinism(self, iterations: int = 2) -> bool:
        """Validate that replay produces deterministic results.
        
        Args:
            iterations: Number of replay iterations to compare
            
        Returns:
            True if all iterations produce identical results
        """
        logger.info(
            "Starting determinism validation",
            iterations=iterations,
        )

        # Get all entries for comparison
        all_entries = await self.event_log.get_entries_range(0)
        if not all_entries:
            logger.warning("No entries to validate")
            return True

        # Store results from each iteration
        iteration_results = []

        for i in range(iterations):
            logger.debug(f"Running determinism iteration {i + 1}/{iterations}")
            
            # Reset state and seed
            self.world_state = WorldState()
            if self._original_seed is not None:
                random.seed(self._original_seed)

            # Replay all events and capture state snapshots
            snapshots = []
            for entry in all_entries:
                await self._apply_effect(entry.effect)
                # Capture state snapshot after each effect
                snapshots.append({
                    "global_seq": entry.global_seq,
                    "world_state_hash": hash(str(self.world_state.model_dump())),
                    "random_state": random.getstate(),
                })

            iteration_results.append(snapshots)

        # Compare all iterations
        is_deterministic = True
        for i in range(1, iterations):
            if len(iteration_results[i]) != len(iteration_results[0]):
                logger.error(
                    "Determinism violation: different number of snapshots",
                    iteration_0_count=len(iteration_results[0]),
                    iteration_i_count=len(iteration_results[i]),
                    iteration=i,
                )
                is_deterministic = False
                continue

            for j, (snap_0, snap_i) in enumerate(zip(iteration_results[0], iteration_results[i])):
                if snap_0["world_state_hash"] != snap_i["world_state_hash"]:
                    logger.error(
                        "Determinism violation: world state mismatch",
                        global_seq=snap_0["global_seq"],
                        iteration_0_hash=snap_0["world_state_hash"],
                        iteration_i_hash=snap_i["world_state_hash"],
                        iteration=i,
                        step=j,
                    )
                    is_deterministic = False

                if snap_0["random_state"] != snap_i["random_state"]:
                    logger.error(
                        "Determinism violation: random state mismatch",
                        global_seq=snap_0["global_seq"],
                        iteration=i,
                        step=j,
                    )
                    is_deterministic = False

        if is_deterministic:
            logger.info("Determinism validation passed")
        else:
            logger.error("Determinism validation failed")

        return is_deterministic

    async def validate_integrity(self) -> bool:
        """Validate log integrity using hash chains.
        
        Returns:
            True if integrity is valid
        """
        logger.info("Validating log integrity")
        is_valid = await self.event_log.validate_integrity()
        
        if is_valid:
            logger.info("Log integrity validation passed")
        else:
            logger.error("Log integrity validation failed")
            
        return is_valid

    async def _apply_effect(self, effect: Effect) -> None:
        """Apply an effect to the world state.
        
        This is a simplified implementation for demonstration.
        In a real system, this would delegate to the appropriate handlers.
        
        Args:
            effect: The effect to apply
        """
        # Simple effect application based on kind
        if effect["kind"] == "Move":
            payload = effect["payload"]
            entity_id = effect["source_id"]
            
            if "x" in payload and "y" in payload:
                # Update spatial index
                self.world_state.spatial_index[entity_id] = (
                    float(payload["x"]),
                    float(payload["y"]),
                    payload.get("z", 0.0),
                )
                
        elif effect["kind"] == "Speak":
            payload = effect["payload"]
            entity_id = effect["source_id"]
            
            # Update entity with last message
            if entity_id not in self.world_state.entities:
                self.world_state.entities[entity_id] = {}
            
            self.world_state.entities[entity_id]["last_message"] = payload.get("text", "")
            
        elif effect["kind"] == "Interact":
            payload = effect["payload"]
            entity_id = effect["source_id"]
            target_id = payload.get("target_id")
            
            if target_id:
                # Update relationships
                if entity_id not in self.world_state.relationships:
                    self.world_state.relationships[entity_id] = []
                
                if target_id not in self.world_state.relationships[entity_id]:
                    self.world_state.relationships[entity_id].append(target_id)

        # Update metadata
        self.world_state.metadata["last_effect_seq"] = effect["global_seq"]
        self.world_state.metadata["last_effect_time"] = effect["sim_time"]


async def run_replay_command(args: List[str]) -> int:
    """Run the replay command with parsed arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog="gunn replay",
        description="Replay event logs for debugging and validation",
    )
    
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the event log file",
    )
    
    parser.add_argument(
        "--from",
        dest="from_seq",
        type=int,
        default=0,
        help="Starting sequence number (default: 0)",
    )
    
    parser.add_argument(
        "--to",
        dest="to_seq",
        type=int,
        help="Ending sequence number (default: latest)",
    )
    
    parser.add_argument(
        "--validate-integrity",
        action="store_true",
        help="Validate log integrity using hash chains",
    )
    
    parser.add_argument(
        "--validate-determinism",
        action="store_true",
        help="Validate deterministic replay behavior",
    )
    
    parser.add_argument(
        "--determinism-iterations",
        type=int,
        default=3,
        help="Number of iterations for determinism validation (default: 3)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Override world seed for testing",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Save replay results to file",
    )

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        return e.code or 1

    # Setup logging
    log_level = "DEBUG" if parsed_args.verbose else "INFO"
    setup_logging("gunn-replay", log_level)

    # Validate log file exists
    if not parsed_args.log_file.exists():
        logger.error(f"Log file not found: {parsed_args.log_file}")
        return 1

    try:
        # Initialize replay engine
        engine = ReplayEngine(parsed_args.log_file, parsed_args.seed)
        await engine.load_log()

        # Validate integrity if requested
        if parsed_args.validate_integrity:
            if not await engine.validate_integrity():
                logger.error("Log integrity validation failed")
                return 1

        # Validate determinism if requested
        if parsed_args.validate_determinism:
            if not await engine.validate_determinism(parsed_args.determinism_iterations):
                logger.error("Determinism validation failed")
                return 1

        # Perform replay
        start_time = time.time()
        replayed_entries = await engine.replay_range(
            parsed_args.from_seq,
            parsed_args.to_seq,
        )
        end_time = time.time()

        # Report results
        logger.info(
            "Replay completed successfully",
            entry_count=len(replayed_entries),
            duration_ms=(end_time - start_time) * 1000,
        )

        # Save results if requested
        if parsed_args.output:
            results = {
                "metadata": {
                    "replay_time": time.time(),
                    "from_seq": parsed_args.from_seq,
                    "to_seq": parsed_args.to_seq,
                    "entry_count": len(replayed_entries),
                    "duration_ms": (end_time - start_time) * 1000,
                },
                "final_world_state": engine.world_state.model_dump(),
                "replayed_entries": [entry.model_dump() for entry in replayed_entries],
            }
            
            with open(parsed_args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {parsed_args.output}")

        return 0

    except Exception as e:
        logger.error(f"Replay failed: {e}", exc_info=True)
        return 1


def main(args: List[str]) -> int:
    """Main entry point for replay command.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    return asyncio.run(run_replay_command(args))