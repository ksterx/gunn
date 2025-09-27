"""CLI utility for replaying event logs with determinism validation."""

import argparse
import asyncio
import json
import random
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

    Args:
        args: Command line arguments

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="gunn replay", description="Replay event logs for debugging and analysis"
    )

    parser.add_argument(
        "--from",
        dest="from_seq",
        type=int,
        default=0,
        help="Starting sequence number (default: 0)",
    )

    parser.add_argument(
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

    Args:
        args: Command line arguments

    Returns:
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
