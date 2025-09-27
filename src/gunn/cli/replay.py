<<<<<<< HEAD
"""Event log replay utility with determinism support.

This module provides CLI tools for replaying event logs from specified ranges
with deterministic behavior and integrity validation.
"""
=======
"""CLI utility for replaying event logs with determinism validation."""
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7

import argparse
import asyncio
import json
import random
<<<<<<< HEAD
import sys
import time
from pathlib import Path
from typing import Any

from gunn.core.event_log import EventLog, EventLogEntry
from gunn.schemas.types import Effect
from gunn.utils.telemetry import get_logger, setup_logging


class ReplayEngine:
    """Engine for replaying event logs with deterministic behavior.

    Supports replay from specified ranges with fixed random seeds
    for deterministic behavior and integrity validation.
    """

    def __init__(self, world_seed: int | None = None, verbose: bool = False):
        """Initialize replay engine.

        Args:
            world_seed: Random seed for deterministic replay
            verbose: Enable verbose logging
        """
        self.world_seed = world_seed
        self.verbose = verbose
        self.logger = get_logger("gunn.replay")

        # Set up deterministic random state if seed provided
        if world_seed is not None:
            random.seed(world_seed)
            self.logger.info(
                "Random seed set for deterministic replay", seed=world_seed
            )

    async def replay_from_log(
        self,
        log: EventLog,
        from_seq: int = 0,
        to_seq: int | None = None,
        validate_integrity: bool = True,
        output_file: Path | None = None,
    ) -> dict[str, Any]:
        """Replay events from log within specified range.

        Args:
            log: EventLog to replay from
            from_seq: Starting sequence number (inclusive)
            to_seq: Ending sequence number (inclusive), None for latest
            validate_integrity: Whether to validate log integrity first
            output_file: Optional file to write replay results

        Returns:
            Dictionary with replay results and statistics
        """
        start_time = time.perf_counter()

        # Validate integrity if requested
        if validate_integrity:
            self.logger.info("Validating log integrity before replay")
            integrity = log.validate_integrity()
            if not integrity["valid"]:
                self.logger.error(
                    "Log integrity validation failed",
                    corrupted_entries=integrity["corrupted_entries"],
                    missing_sequences=integrity["missing_sequences"],
                )
                return {
                    "success": False,
                    "error": "Log integrity validation failed",
                    "integrity_report": integrity,
                }

        # Determine replay range
        latest_seq = log.get_latest_seq()
        if to_seq is None:
            to_seq = latest_seq

        self.logger.info(
            "Starting replay",
            from_seq=from_seq,
            to_seq=to_seq,
            latest_seq=latest_seq,
            world_seed=self.world_seed,
        )

        # Get entries to replay
        if from_seq == 0:
            entries = [
                entry for entry in log.get_all_entries() if entry.global_seq <= to_seq
            ]
        else:
            entries = [
                entry
                for entry in log.get_entries_since(from_seq - 1)
                if entry.global_seq <= to_seq
            ]

        # Replay entries
        replay_results = []
        for entry in entries:
            result = await self._replay_entry(entry)
            replay_results.append(result)

            if self.verbose:
                self.logger.info(
                    "Replayed entry",
                    global_seq=entry.global_seq,
                    effect_kind=entry.effect["kind"],
                    effect_uuid=entry.effect["uuid"],
                )

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Compile results
        results = {
            "success": True,
            "replay_range": {"from": from_seq, "to": to_seq},
            "entries_replayed": len(replay_results),
            "duration_seconds": duration,
            "world_seed": self.world_seed,
            "replay_results": replay_results,
            "statistics": {
                "total_entries": len(replay_results),
                "effect_types": self._count_effect_types(entries),
                "time_range": {
                    "first_sim_time": entries[0].sim_time if entries else None,
                    "last_sim_time": entries[-1].sim_time if entries else None,
                    "first_wall_time": entries[0].wall_time if entries else None,
                    "last_wall_time": entries[-1].wall_time if entries else None,
                },
            },
        }

        # Write to output file if specified
        if output_file:
            await self._write_results(results, output_file)

        self.logger.info(
            "Replay completed",
            entries_replayed=len(replay_results),
            duration_seconds=duration,
        )

        return results

    async def _replay_entry(self, entry: EventLogEntry) -> dict[str, Any]:
        """Replay a single log entry.

        Args:
            entry: EventLogEntry to replay

        Returns:
            Dictionary with replay result for this entry
        """
        effect = entry.effect

        # Simulate processing the effect
        # In a real implementation, this would apply the effect to world state
        processing_start = time.perf_counter()

        # Simulate some processing time based on effect type
        if effect["kind"] in ["Move", "Interact"]:
            await asyncio.sleep(0.001)  # 1ms for complex effects
        else:
            await asyncio.sleep(0.0001)  # 0.1ms for simple effects

        processing_end = time.perf_counter()
        processing_time = processing_end - processing_start

        return {
            "global_seq": entry.global_seq,
            "effect_uuid": effect["uuid"],
            "effect_kind": effect["kind"],
            "sim_time": entry.sim_time,
            "wall_time": entry.wall_time,
            "processing_time_ms": processing_time * 1000,
            "checksum": entry.checksum,
            "req_id": entry.req_id,
        }

    def _count_effect_types(self, entries: list[EventLogEntry]) -> dict[str, int]:
        """Count effect types in entries.

        Args:
            entries: List of entries to analyze

        Returns:
            Dictionary mapping effect kinds to counts
        """
        counts: dict[str, int] = {}
        for entry in entries:
            kind = entry.effect["kind"]
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    async def _write_results(self, results: dict[str, Any], output_file: Path) -> None:
        """Write replay results to file.

        Args:
            results: Results dictionary to write
            output_file: Path to output file
        """
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(
                "Replay results written to file", output_file=str(output_file)
            )
        except Exception as e:
            self.logger.error(
                "Failed to write results to file",
                error=str(e),
                output_file=str(output_file),
            )


async def create_sample_log(
    num_entries: int = 10, world_seed: int | None = None
) -> EventLog:
    """Create a sample event log for testing replay functionality.

    Args:
        num_entries: Number of entries to create
        world_seed: Random seed for deterministic generation

    Returns:
        EventLog with sample data
    """
    if world_seed is not None:
        random.seed(world_seed)

    log = EventLog("sample_world")

    effect_types = ["AgentJoined", "MessageSent", "Move", "Interact", "AgentLeft"]
    agent_ids = ["agent_1", "agent_2", "agent_3"]

    for i in range(num_entries):
        effect_kind = random.choice(effect_types)
        agent_id = random.choice(agent_ids)

        # Generate appropriate payload based on effect type
        payload: dict[str, Any]
        if effect_kind == "AgentJoined":
            payload = {
                "agent_id": agent_id,
                "position": {"x": random.uniform(0, 100), "y": random.uniform(0, 100)},
            }
        elif effect_kind == "MessageSent":
            payload = {
                "sender": agent_id,
                "text": f"Message {i} from {agent_id}",
                "timestamp": float(i * 100),
            }
        elif effect_kind == "Move":
            payload = {
                "agent_id": agent_id,
                "from_pos": {"x": random.uniform(0, 100), "y": random.uniform(0, 100)},
                "to_pos": {"x": random.uniform(0, 100), "y": random.uniform(0, 100)},
            }
        elif effect_kind == "Interact":
            payload = {
                "agent_id": agent_id,
                "target": random.choice(agent_ids),
                "action": "greet",
            }
        else:  # AgentLeft
            payload = {"agent_id": agent_id, "reason": "disconnected"}

        effect = Effect(
            uuid=f"sample-{i:04d}",
            kind=effect_kind,
            payload=payload,
            global_seq=0,  # Will be set by EventLog
            sim_time=float(i * 5),  # 5 second intervals
            source_id=agent_id,
            schema_version="1.0.0",
        )

        await log.append(effect, req_id=f"sample-req-{i}")

    return log


def parse_replay_args(args: list[str]) -> argparse.Namespace:
    """Parse replay command arguments.
=======
import time
from pathlib import Path
from typing import Optional

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

    async def replay_range(
        self, from_seq: int = 0, to_seq: Optional[int] = None
    ) -> list[EventLogEntry]:
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
                snapshots.append(
                    {
                        "global_seq": entry.global_seq,
                        "world_state_hash": hash(str(self.world_state.model_dump())),
                        "random_state": random.getstate(),
                    }
                )

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

            for j, (snap_0, snap_i) in enumerate(
                zip(iteration_results[0], iteration_results[i])
            ):
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

            self.world_state.entities[entity_id]["last_message"] = payload.get(
                "text", ""
            )

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


async def run_replay_command(args: list[str]) -> int:
    """Run the replay command with parsed arguments.
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7

    Args:
        args: Command line arguments

    Returns:
<<<<<<< HEAD
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="gunn replay", description="Replay event logs for debugging and analysis"
=======
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
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7
    )

    parser.add_argument(
        "--from",
        dest="from_seq",
        type=int,
        default=0,
        help="Starting sequence number (default: 0)",
    )

    parser.add_argument(
<<<<<<< HEAD
        "--to", dest="to_seq", type=int, help="Ending sequence number (default: latest)"
    )

    parser.add_argument("--seed", type=int, help="Random seed for deterministic replay")

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to event log file (if not provided, creates sample log)",
    )

    parser.add_argument(
        "--output", type=Path, help="Output file for replay results (JSON format)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate log integrity before replay (default: True)",
    )

    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip log integrity validation",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--sample-entries",
        type=int,
        default=10,
        help="Number of entries in sample log (default: 10)",
    )

    return parser.parse_args(args)


async def run_replay_command(args: list[str]) -> int:
    """Run the replay command with given arguments.
=======
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
            if not await engine.validate_determinism(
                parsed_args.determinism_iterations
            ):
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

            with open(parsed_args.output, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to {parsed_args.output}")

        return 0

    except Exception as e:
        logger.error(f"Replay failed: {e}", exc_info=True)
        return 1


def main(args: list[str]) -> int:
    """Main entry point for replay command.
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7

    Args:
        args: Command line arguments

    Returns:
<<<<<<< HEAD
        Exit code (0 for success, non-zero for error)
    """
    try:
        parsed_args = parse_replay_args(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1

    # Setup logging
    log_level = "DEBUG" if parsed_args.verbose else "INFO"
    setup_logging(log_level, enable_pii_redaction=False)

    logger = get_logger("gunn.replay.cli")

    try:
        # Create or load event log
        if parsed_args.log_file:
            # TODO: Implement log file loading
            logger.error("Loading from log files not yet implemented")
            return 1
        else:
            logger.info("Creating sample log for replay demonstration")
            log = await create_sample_log(
                num_entries=parsed_args.sample_entries, world_seed=parsed_args.seed
            )

        # Create replay engine
        engine = ReplayEngine(world_seed=parsed_args.seed, verbose=parsed_args.verbose)

        # Run replay
        results = await engine.replay_from_log(
            log=log,
            from_seq=parsed_args.from_seq,
            to_seq=parsed_args.to_seq,
            validate_integrity=parsed_args.validate,
            output_file=parsed_args.output,
        )

        if not results["success"]:
            logger.error("Replay failed", error=results.get("error"))
            return 1

        # Print summary
        stats = results["statistics"]
        print("\nReplay Summary:")
        print(
            f"  Range: {results['replay_range']['from']} to {results['replay_range']['to']}"
        )
        print(f"  Entries replayed: {results['entries_replayed']}")
        print(f"  Duration: {results['duration_seconds']:.3f} seconds")
        print(f"  World seed: {results['world_seed']}")

        if stats["effect_types"]:
            print("  Effect types:")
            for effect_type, count in sorted(stats["effect_types"].items()):
                print(f"    {effect_type}: {count}")

        if parsed_args.output:
            print(f"  Results written to: {parsed_args.output}")

        return 0

    except Exception as e:
        logger.error("Replay command failed", error=str(e))
        if parsed_args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for replay CLI."""
    return asyncio.run(run_replay_command(sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
=======
        Exit code
    """
    return asyncio.run(run_replay_command(args))
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7
