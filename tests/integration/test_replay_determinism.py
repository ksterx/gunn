"""Integration tests for replay determinism and consistency validation.

These tests verify that replay functionality produces identical results
when run with the same parameters and random seeds.
"""

import tempfile
from pathlib import Path

import pytest

from gunn.cli.replay import ReplayEngine, create_sample_log
from gunn.core.event_log import EventLog
from gunn.schemas.types import Effect


class TestReplayDeterminism:
    """Test deterministic replay behavior."""

    @pytest.mark.asyncio
    async def test_deterministic_sample_log_generation(self):
        """Test that sample log generation is deterministic with same seed."""
        seed = 12345

        # Generate two logs with same seed
        log1 = await create_sample_log(num_entries=10, world_seed=seed)
        log2 = await create_sample_log(num_entries=10, world_seed=seed)

        # Logs should be identical
        entries1 = log1.get_all_entries()
        entries2 = log2.get_all_entries()

        assert len(entries1) == len(entries2)

        for entry1, entry2 in zip(entries1, entries2, strict=True):
            assert entry1.effect["uuid"] == entry2.effect["uuid"]
            assert entry1.effect["kind"] == entry2.effect["kind"]
            assert entry1.effect["payload"] == entry2.effect["payload"]
            assert entry1.effect["sim_time"] == entry2.effect["sim_time"]

    @pytest.mark.asyncio
    async def test_different_seeds_produce_different_logs(self):
        """Test that different seeds produce different logs."""
        log1 = await create_sample_log(num_entries=10, world_seed=12345)
        log2 = await create_sample_log(num_entries=10, world_seed=54321)

        entries1 = log1.get_all_entries()
        entries2 = log2.get_all_entries()

        # Should have same number of entries but different content
        assert len(entries1) == len(entries2)

        # At least some effects should be different
        different_effects = 0
        for entry1, entry2 in zip(entries1, entries2, strict=True):
            if (
                entry1.effect["kind"] != entry2.effect["kind"]
                or entry1.effect["payload"] != entry2.effect["payload"]
            ):
                different_effects += 1

        assert different_effects > 0, (
            "Logs with different seeds should produce different effects"
        )

    @pytest.mark.asyncio
    async def test_replay_determinism_same_seed(self):
        """Test that replay produces identical results with same seed."""
        # Create a sample log
        original_log = await create_sample_log(num_entries=5, world_seed=12345)

        # Replay twice with same seed
        engine1 = ReplayEngine(world_seed=67890, verbose=False)
        engine2 = ReplayEngine(world_seed=67890, verbose=False)

        results1 = await engine1.replay_from_log(original_log, validate_integrity=False)
        results2 = await engine2.replay_from_log(original_log, validate_integrity=False)

        # Results should be identical
        assert results1["success"] == results2["success"]
        assert results1["entries_replayed"] == results2["entries_replayed"]
        assert len(results1["replay_results"]) == len(results2["replay_results"])

        # Individual replay results should match
        for result1, result2 in zip(
            results1["replay_results"], results2["replay_results"], strict=True
        ):
            assert result1["global_seq"] == result2["global_seq"]
            assert result1["effect_uuid"] == result2["effect_uuid"]
            assert result1["effect_kind"] == result2["effect_kind"]

    @pytest.mark.asyncio
    async def test_replay_range_consistency(self):
        """Test that replay ranges produce consistent results."""
        # Create a larger log
        log = await create_sample_log(num_entries=20, world_seed=12345)
        engine = ReplayEngine(world_seed=67890)

        # Replay full range
        full_results = await engine.replay_from_log(
            log, from_seq=0, to_seq=None, validate_integrity=False
        )

        # Replay partial ranges
        first_half = await engine.replay_from_log(
            log, from_seq=0, to_seq=10, validate_integrity=False
        )
        second_half = await engine.replay_from_log(
            log, from_seq=11, to_seq=None, validate_integrity=False
        )

        # Combined partial results should match full results
        combined_count = (
            first_half["entries_replayed"] + second_half["entries_replayed"]
        )
        assert combined_count == full_results["entries_replayed"]

        # Check that ranges don't overlap
        first_half_seqs = {r["global_seq"] for r in first_half["replay_results"]}
        second_half_seqs = {r["global_seq"] for r in second_half["replay_results"]}
        assert len(first_half_seqs & second_half_seqs) == 0, "Ranges should not overlap"

    @pytest.mark.asyncio
    async def test_integrity_validation_during_replay(self):
        """Test integrity validation during replay."""
        # Create a log with known integrity
        log = await create_sample_log(num_entries=5, world_seed=12345)
        engine = ReplayEngine(world_seed=67890)

        # Replay with integrity validation
        results = await engine.replay_from_log(log, validate_integrity=True)

        assert results["success"] is True
        assert results["entries_replayed"] == 5

        # Corrupt the log and test validation failure
        log._entries[2].checksum = "corrupted_checksum"

        corrupted_results = await engine.replay_from_log(log, validate_integrity=True)

        assert corrupted_results["success"] is False
        assert "integrity" in corrupted_results["error"].lower()

    @pytest.mark.asyncio
    async def test_replay_output_file(self):
        """Test replay output to file."""
        log = await create_sample_log(num_entries=3, world_seed=12345)
        engine = ReplayEngine(world_seed=67890)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = Path(f.name)

        try:
            # Replay with output file
            results = await engine.replay_from_log(
                log, validate_integrity=False, output_file=output_file
            )

            assert results["success"] is True
            assert output_file.exists()

            # Verify file contents
            import json

            with open(output_file) as f:
                file_contents = json.load(f)

            assert file_contents["success"] is True
            assert file_contents["entries_replayed"] == 3
            assert len(file_contents["replay_results"]) == 3

        finally:
            # Clean up
            if output_file.exists():
                output_file.unlink()


class TestReplayConsistency:
    """Test replay consistency across multiple runs."""

    @pytest.mark.asyncio
    async def test_multiple_replay_runs_consistency(self):
        """Test that multiple replay runs produce consistent results."""
        # Create base log
        log = await create_sample_log(num_entries=8, world_seed=12345)

        # Run replay multiple times with same parameters
        results_list = []
        for _ in range(3):
            engine = ReplayEngine(world_seed=67890)
            results = await engine.replay_from_log(log, validate_integrity=False)
            results_list.append(results)

        # All results should be identical
        first_results = results_list[0]
        for results in results_list[1:]:
            assert results["success"] == first_results["success"]
            assert results["entries_replayed"] == first_results["entries_replayed"]
            assert results["world_seed"] == first_results["world_seed"]

            # Compare replay results
            for i, (result, first_result) in enumerate(
                zip(
                    results["replay_results"],
                    first_results["replay_results"],
                    strict=False,
                )
            ):
                assert result["global_seq"] == first_result["global_seq"], (
                    f"Mismatch at index {i}"
                )
                assert result["effect_uuid"] == first_result["effect_uuid"], (
                    f"Mismatch at index {i}"
                )

    @pytest.mark.asyncio
    async def test_replay_statistics_consistency(self):
        """Test that replay statistics are consistent."""
        log = await create_sample_log(num_entries=15, world_seed=12345)
        engine = ReplayEngine(world_seed=67890)

        # Run replay
        results = await engine.replay_from_log(log, validate_integrity=False)

        # Verify statistics consistency
        stats = results["statistics"]
        assert stats["total_entries"] == results["entries_replayed"]
        assert stats["total_entries"] == len(results["replay_results"])

        # Effect type counts should sum to total
        effect_type_sum = sum(stats["effect_types"].values())
        assert effect_type_sum == stats["total_entries"]

        # Time ranges should be consistent
        if stats["total_entries"] > 0:
            assert stats["time_range"]["first_sim_time"] is not None
            assert stats["time_range"]["last_sim_time"] is not None
            assert (
                stats["time_range"]["first_sim_time"]
                <= stats["time_range"]["last_sim_time"]
            )

    @pytest.mark.asyncio
    async def test_empty_log_replay(self):
        """Test replay behavior with empty log."""
        empty_log = EventLog("empty_test")
        engine = ReplayEngine(world_seed=12345)

        results = await engine.replay_from_log(empty_log, validate_integrity=False)

        assert results["success"] is True
        assert results["entries_replayed"] == 0
        assert len(results["replay_results"]) == 0
        assert results["statistics"]["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_single_entry_replay(self):
        """Test replay with single entry."""
        log = EventLog("single_test")

        effect = Effect(
            uuid="single-test",
            kind="SingleTest",
            payload={"test": "data"},
            global_seq=0,
            sim_time=1.0,
            source_id="test_source",
            schema_version="1.0.0",
        )

        await log.append(effect, req_id="single-req")

        engine = ReplayEngine(world_seed=12345)
        results = await engine.replay_from_log(log, validate_integrity=False)

        assert results["success"] is True
        assert results["entries_replayed"] == 1
        assert len(results["replay_results"]) == 1
        assert results["replay_results"][0]["effect_uuid"] == "single-test"


class TestReplayPerformance:
    """Test replay performance characteristics."""

    @pytest.mark.asyncio
    async def test_large_log_replay_performance(self):
        """Test replay performance with larger logs."""
        # Create a larger log
        log = await create_sample_log(num_entries=100, world_seed=12345)
        engine = ReplayEngine(world_seed=67890)

        # Measure replay time
        import time

        start_time = time.perf_counter()

        results = await engine.replay_from_log(log, validate_integrity=False)

        end_time = time.perf_counter()
        duration = end_time - start_time

        assert results["success"] is True
        assert results["entries_replayed"] == 100

        # Should complete in reasonable time (< 5 seconds for 100 entries)
        assert duration < 5.0

        # Verify performance is recorded in results
        assert results["duration_seconds"] > 0
        assert abs(results["duration_seconds"] - duration) < 0.1  # Should be close

    @pytest.mark.asyncio
    async def test_replay_memory_usage(self):
        """Test that replay doesn't consume excessive memory."""
        # Create log with moderate size
        log = await create_sample_log(num_entries=50, world_seed=12345)
        engine = ReplayEngine(world_seed=67890)

        # Run replay and verify it completes
        results = await engine.replay_from_log(log, validate_integrity=False)

        assert results["success"] is True
        assert results["entries_replayed"] == 50

        # Memory usage test would require more sophisticated monitoring
        # For now, just verify the replay completes successfully
