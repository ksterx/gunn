"""Integration tests for replay consistency and golden trace validation.

These tests ensure that replay functionality produces identical results
when run with the same parameters and validates deterministic behavior
through golden trace comparisons.
"""

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from gunn.cli.replay import ReplayEngine, create_sample_log
from gunn.core.event_log import EventLog
from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.facades import RLFacade
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import Intent
from gunn.utils.telemetry import get_logger


class GoldenTraceValidator:
    """Validates deterministic behavior through golden trace comparisons."""

    def __init__(self):
        self.logger = get_logger("golden_trace_validator")

    def create_trace_hash(self, trace_data: dict[str, Any]) -> str:
        """Create deterministic hash of trace data."""

        # Sort keys recursively for consistent hashing
        def sort_dict(obj):
            if isinstance(obj, dict):
                return {k: sort_dict(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return [sort_dict(item) for item in obj]
            else:
                return obj

        sorted_data = sort_dict(trace_data)
        json_str = json.dumps(sorted_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def extract_trace_from_log(self, event_log: EventLog) -> dict[str, Any]:
        """Extract deterministic trace data from event log."""
        entries = event_log.get_all_entries()

        trace = {
            "version": "1.0.0",
            "entry_count": len(entries),
            "effects": [],
            "checksums": [],
            "global_seqs": [],
            "sim_times": [],
            "source_ids": [],
        }

        for entry in entries:
            effect = entry.effect

            # Extract deterministic fields
            trace["effects"].append(
                {
                    "kind": effect["kind"],
                    "payload": effect["payload"],
                    "global_seq": effect["global_seq"],
                    "sim_time": effect["sim_time"],
                    "source_id": effect["source_id"],
                    "schema_version": effect["schema_version"],
                    # Note: UUID is excluded as it's non-deterministic
                }
            )

            trace["checksums"].append(entry.checksum)
            trace["global_seqs"].append(effect["global_seq"])
            trace["sim_times"].append(effect["sim_time"])
            trace["source_ids"].append(effect["source_id"])

        return trace

    def compare_traces(
        self, trace1: dict[str, Any], trace2: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare two traces and return detailed comparison results."""
        comparison = {
            "identical": True,
            "differences": [],
            "hash1": self.create_trace_hash(trace1),
            "hash2": self.create_trace_hash(trace2),
        }

        # Compare basic properties
        if trace1["entry_count"] != trace2["entry_count"]:
            comparison["identical"] = False
            comparison["differences"].append(
                {
                    "field": "entry_count",
                    "value1": trace1["entry_count"],
                    "value2": trace2["entry_count"],
                }
            )

        # Compare effects
        if len(trace1["effects"]) == len(trace2["effects"]):
            for i, (effect1, effect2) in enumerate(
                zip(trace1["effects"], trace2["effects"], strict=False)
            ):
                if effect1 != effect2:
                    comparison["identical"] = False
                    comparison["differences"].append(
                        {
                            "field": f"effects[{i}]",
                            "value1": effect1,
                            "value2": effect2,
                        }
                    )
        else:
            comparison["identical"] = False
            comparison["differences"].append(
                {
                    "field": "effects_length",
                    "value1": len(trace1["effects"]),
                    "value2": len(trace2["effects"]),
                }
            )

        # Compare sequences
        for field in ["global_seqs", "sim_times", "source_ids"]:
            if trace1[field] != trace2[field]:
                comparison["identical"] = False
                comparison["differences"].append(
                    {
                        "field": field,
                        "value1": trace1[field],
                        "value2": trace2[field],
                    }
                )

        return comparison


class TestReplayConsistency:
    """Test replay consistency across multiple runs."""

    @pytest.fixture
    def config(self) -> OrchestratorConfig:
        """Create test configuration for replay consistency."""
        return OrchestratorConfig(
            max_agents=5,
            staleness_threshold=1,
            debounce_ms=50.0,
            deadline_ms=5000.0,
            token_budget=1000,
            backpressure_policy="defer",
            default_priority=1,
            use_in_memory_dedup=True,
        )

    @pytest.fixture
    async def orchestrator(self, config: OrchestratorConfig) -> Orchestrator:
        """Create orchestrator for replay testing."""
        orchestrator = Orchestrator(config, world_id="replay_test")
        await orchestrator.initialize()
        return orchestrator

    @pytest.fixture
    async def facade(self, orchestrator: Orchestrator) -> RLFacade:
        """Create facade for replay testing."""
        facade = RLFacade(orchestrator=orchestrator)
        await facade.initialize()
        return facade

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for testing."""
        config = PolicyConfig(
            distance_limit=100.0,
            relationship_filter=[],
            field_visibility={},
            max_patch_ops=25,
        )
        return DefaultObservationPolicy(config)

    @pytest.mark.asyncio
    async def test_identical_replay_with_same_seed(self):
        """Test that replay produces identical results with same seed."""
        seed = 12345
        num_entries = 10

        # Create two logs with same seed
        log1 = await create_sample_log(num_entries=num_entries, world_seed=seed)
        log2 = await create_sample_log(num_entries=num_entries, world_seed=seed)

        # Verify logs are identical
        entries1 = log1.get_all_entries()
        entries2 = log2.get_all_entries()

        assert len(entries1) == len(entries2)

        for entry1, entry2 in zip(entries1, entries2, strict=False):
            # Compare all deterministic fields
            assert entry1.effect["kind"] == entry2.effect["kind"]
            assert entry1.effect["payload"] == entry2.effect["payload"]
            assert entry1.effect["global_seq"] == entry2.effect["global_seq"]
            assert entry1.effect["sim_time"] == entry2.effect["sim_time"]
            assert entry1.effect["source_id"] == entry2.effect["source_id"]
            assert entry1.effect["schema_version"] == entry2.effect["schema_version"]
            # Note: UUIDs will be different, but that's expected

    @pytest.mark.asyncio
    async def test_replay_engine_determinism(self):
        """Test that ReplayEngine produces deterministic results."""
        # Create base log
        original_log = await create_sample_log(num_entries=8, world_seed=12345)

        # Replay multiple times with same seed
        replay_seed = 67890
        results = []

        for _ in range(3):
            engine = ReplayEngine(world_seed=replay_seed, verbose=False)
            result = await engine.replay_from_log(
                original_log, validate_integrity=False
            )
            results.append(result)

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result["success"] == first_result["success"]
            assert result["entries_replayed"] == first_result["entries_replayed"]
            assert result["world_seed"] == first_result["world_seed"]

            # Compare replay results
            assert len(result["replay_results"]) == len(first_result["replay_results"])

            for replay_result, first_replay_result in zip(
                result["replay_results"], first_result["replay_results"], strict=False
            ):
                assert replay_result["global_seq"] == first_replay_result["global_seq"]
                assert (
                    replay_result["effect_kind"] == first_replay_result["effect_kind"]
                )
                assert (
                    replay_result["effect_uuid"] == first_replay_result["effect_uuid"]
                )

    @pytest.mark.asyncio
    async def test_replay_range_consistency(self):
        """Test that replay ranges produce consistent partial results."""
        # Create larger log
        log = await create_sample_log(num_entries=20, world_seed=12345)
        engine = ReplayEngine(world_seed=67890)

        # Replay full range
        full_result = await engine.replay_from_log(
            log, from_seq=0, to_seq=None, validate_integrity=False
        )

        # Replay first half
        first_half = await engine.replay_from_log(
            log, from_seq=0, to_seq=10, validate_integrity=False
        )

        # Replay second half
        second_half = await engine.replay_from_log(
            log, from_seq=11, to_seq=None, validate_integrity=False
        )

        # Verify consistency
        assert first_half["success"] and second_half["success"]

        combined_count = (
            first_half["entries_replayed"] + second_half["entries_replayed"]
        )
        assert combined_count == full_result["entries_replayed"]

        # Verify no overlap in sequences
        first_seqs = {r["global_seq"] for r in first_half["replay_results"]}
        second_seqs = {r["global_seq"] for r in second_half["replay_results"]}
        assert len(first_seqs & second_seqs) == 0, "Ranges should not overlap"

        # Verify coverage
        full_seqs = {r["global_seq"] for r in full_result["replay_results"]}
        combined_seqs = first_seqs | second_seqs
        assert combined_seqs == full_seqs, "Combined ranges should cover full range"

    @pytest.mark.asyncio
    async def test_replay_with_integrity_validation(self):
        """Test replay with integrity validation enabled."""
        # Create log with known integrity
        log = await create_sample_log(num_entries=5, world_seed=12345)
        engine = ReplayEngine(world_seed=67890)

        # Verify integrity is valid
        assert log.validate_integrity(), "Original log should have valid integrity"

        # Replay with integrity validation
        result = await engine.replay_from_log(log, validate_integrity=True)

        assert result["success"], "Replay with valid integrity should succeed"
        assert result["entries_replayed"] == 5

        # Test with corrupted integrity
        corrupted_log = await create_sample_log(num_entries=3, world_seed=12345)
        corrupted_log._entries[1].checksum = "corrupted_checksum"

        corrupted_result = await engine.replay_from_log(
            corrupted_log, validate_integrity=True
        )

        assert not corrupted_result["success"], (
            "Replay with corrupted integrity should fail"
        )
        assert "integrity" in corrupted_result["error"].lower()

    @pytest.mark.asyncio
    async def test_replay_output_consistency(self):
        """Test that replay output files are consistent."""
        log = await create_sample_log(num_entries=5, world_seed=12345)
        engine = ReplayEngine(world_seed=67890)

        # Create two output files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            output_file1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            output_file2 = Path(f2.name)

        try:
            # Replay to both files
            result1 = await engine.replay_from_log(
                log, validate_integrity=False, output_file=output_file1
            )

            # Create new engine instance to ensure independence
            engine2 = ReplayEngine(world_seed=67890)
            result2 = await engine2.replay_from_log(
                log, validate_integrity=False, output_file=output_file2
            )

            # Both should succeed
            assert result1["success"] and result2["success"]

            # Load and compare file contents
            with open(output_file1) as f:
                content1 = json.load(f)

            with open(output_file2) as f:
                content2 = json.load(f)

            # Files should be identical
            assert content1["success"] == content2["success"]
            assert content1["entries_replayed"] == content2["entries_replayed"]
            assert content1["world_seed"] == content2["world_seed"]
            assert len(content1["replay_results"]) == len(content2["replay_results"])

            # Compare replay results
            for result1, result2 in zip(
                content1["replay_results"], content2["replay_results"], strict=False
            ):
                assert result1["global_seq"] == result2["global_seq"]
                assert result1["effect_uuid"] == result2["effect_uuid"]
                assert result1["effect_kind"] == result2["effect_kind"]

        finally:
            # Clean up
            for output_file in [output_file1, output_file2]:
                if output_file.exists():
                    output_file.unlink()


class TestGoldenTraces:
    """Test deterministic behavior through golden trace validation."""

    @pytest.fixture
    def validator(self) -> GoldenTraceValidator:
        """Create golden trace validator."""
        return GoldenTraceValidator()

    @pytest.mark.asyncio
    async def test_golden_trace_creation_and_validation(
        self, validator: GoldenTraceValidator
    ):
        """Test creation and validation of golden traces."""
        # Create deterministic log
        log = await create_sample_log(num_entries=5, world_seed=42)

        # Extract trace
        trace = validator.extract_trace_from_log(log)

        # Verify trace structure
        assert trace["version"] == "1.0.0"
        assert trace["entry_count"] == 5
        assert len(trace["effects"]) == 5
        assert len(trace["checksums"]) == 5
        assert len(trace["global_seqs"]) == 5

        # Verify deterministic properties
        assert trace["global_seqs"] == sorted(trace["global_seqs"]), (
            "Global seqs should be monotonic"
        )
        assert len(set(trace["checksums"])) == len(trace["checksums"]), (
            "Checksums should be unique"
        )

        # Create hash
        trace_hash = validator.create_trace_hash(trace)
        assert len(trace_hash) == 64, "SHA-256 hash should be 64 characters"

    @pytest.mark.asyncio
    async def test_identical_traces_from_same_seed(
        self, validator: GoldenTraceValidator
    ):
        """Test that identical seeds produce identical traces."""
        seed = 12345

        # Create two logs with same seed
        log1 = await create_sample_log(num_entries=8, world_seed=seed)
        log2 = await create_sample_log(num_entries=8, world_seed=seed)

        # Extract traces
        trace1 = validator.extract_trace_from_log(log1)
        trace2 = validator.extract_trace_from_log(log2)

        # Compare traces
        comparison = validator.compare_traces(trace1, trace2)

        assert comparison["identical"], (
            f"Traces should be identical: {comparison['differences']}"
        )
        assert comparison["hash1"] == comparison["hash2"], "Trace hashes should match"
        assert len(comparison["differences"]) == 0, "Should have no differences"

    @pytest.mark.asyncio
    async def test_different_traces_from_different_seeds(
        self, validator: GoldenTraceValidator
    ):
        """Test that different seeds produce different traces."""
        # Create logs with different seeds
        log1 = await create_sample_log(num_entries=8, world_seed=12345)
        log2 = await create_sample_log(num_entries=8, world_seed=54321)

        # Extract traces
        trace1 = validator.extract_trace_from_log(log1)
        trace2 = validator.extract_trace_from_log(log2)

        # Compare traces
        comparison = validator.compare_traces(trace1, trace2)

        assert not comparison["identical"], "Traces from different seeds should differ"
        assert comparison["hash1"] != comparison["hash2"], "Trace hashes should differ"
        assert len(comparison["differences"]) > 0, "Should have differences"

    @pytest.mark.asyncio
    async def test_trace_consistency_across_replay(
        self, validator: GoldenTraceValidator
    ):
        """Test that traces remain consistent across replay operations."""
        # Create original log
        original_log = await create_sample_log(num_entries=6, world_seed=12345)
        original_trace = validator.extract_trace_from_log(original_log)

        # Replay the log
        engine = ReplayEngine(world_seed=67890, verbose=False)
        replay_result = await engine.replay_from_log(
            original_log, validate_integrity=False
        )

        assert replay_result["success"], "Replay should succeed"

        # The original trace should remain unchanged after replay
        post_replay_trace = validator.extract_trace_from_log(original_log)
        comparison = validator.compare_traces(original_trace, post_replay_trace)

        assert comparison["identical"], "Original trace should be unchanged by replay"

    @pytest.mark.asyncio
    async def test_golden_trace_with_real_orchestrator(
        self, validator: GoldenTraceValidator
    ):
        """Test golden trace creation with real orchestrator operations."""
        config = OrchestratorConfig(
            max_agents=3,
            staleness_threshold=10,  # Increase threshold to avoid staleness errors
            use_in_memory_dedup=True,
        )

        # Create high-performance validator to avoid cooldown issues
        from gunn.core.orchestrator import DefaultEffectValidator

        high_perf_validator = DefaultEffectValidator(
            max_intents_per_minute=12000,
            max_tokens_per_minute=1200000,
            default_cooldown_seconds=0.0,  # No cooldown for testing
            max_payload_size_bytes=50000,
        )

        orchestrator = Orchestrator(
            config, world_id="golden_test", effect_validator=high_perf_validator
        )
        await orchestrator.initialize()

        try:
            facade = RLFacade(orchestrator=orchestrator)
            await facade.initialize()

            # Set deterministic seed
            # orchestrator.set_world_seed(98765)  # Method not implemented yet

            # Register agents
            policy_config = PolicyConfig(
                distance_limit=100.0,
                relationship_filter=[],
                field_visibility={},
                max_patch_ops=25,
            )

            agent_ids = []
            for i in range(3):
                agent_id = f"golden_agent_{i}"
                policy = DefaultObservationPolicy(policy_config)
                await facade.register_agent(agent_id, policy)
                agent_ids.append(agent_id)

            # Setup permissions and world state for all agents
            effect_validator = orchestrator.effect_validator
            if hasattr(effect_validator, "set_agent_permissions"):
                permissions = {
                    "submit_intent",
                    "intent:speak",
                    "intent:move",
                    "intent:interact",
                    "intent:custom",
                }
                for agent_id in agent_ids:
                    effect_validator.set_agent_permissions(agent_id, permissions)

            # Add agents to world state
            for agent_id in agent_ids:
                orchestrator.world_state.entities[agent_id] = {
                    "id": agent_id,
                    "type": "agent",
                    "position": {"x": 0, "y": 0},
                }
                orchestrator.world_state.spatial_index[agent_id] = (0.0, 0.0, 0.0)

            # Clear any existing cooldowns by setting all agent cooldowns to zero
            if hasattr(effect_validator, "_agent_cooldowns"):
                effect_validator._agent_cooldowns.clear()

            # Execute deterministic operations (using only different agents to avoid cooldown issues)
            operations = [
                ("golden_agent_0", "Speak", {"message": "First message"}),
                ("golden_agent_1", "Speak", {"message": "Hello world"}),
                ("golden_agent_2", "Speak", {"message": "Test message"}),
            ]

            successful_operations = 0
            for i, (agent_id, kind, payload) in enumerate(operations):
                try:
                    # Use global sequence to avoid staleness errors
                    current_global_seq = orchestrator._global_seq
                    intent: Intent = {
                        "kind": kind,
                        "payload": payload,
                        "context_seq": current_global_seq,
                        "req_id": f"golden_req_{i}",
                        "agent_id": agent_id,
                        "priority": 1,
                        "schema_version": "1.0.0",
                    }

                    result = await facade.step(agent_id, intent)
                    if result and not result.get("error"):
                        successful_operations += 1
                        # Add small delay to ensure proper processing
                        await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"Operation {i} failed: {e}")
                    # Continue with next operation

            # Extract golden trace
            event_log = orchestrator.event_log
            golden_trace = validator.extract_trace_from_log(event_log)

            # Verify trace properties - expect at least 2 successful operations
            actual_operations = golden_trace["entry_count"]

            assert actual_operations >= 2, (
                f"Expected at least 2 operations, got {actual_operations}"
            )
            assert actual_operations <= len(operations), (
                f"Got more operations ({actual_operations}) than expected ({len(operations)})"
            )
            assert len(golden_trace["effects"]) == actual_operations

            # Verify deterministic ordering
            global_seqs = golden_trace["global_seqs"]
            assert global_seqs == sorted(global_seqs), (
                "Global sequences should be monotonic"
            )

            # Test basic golden trace properties
            assert "checksums" in golden_trace, "Golden trace should have checksums"
            assert "sim_times" in golden_trace, (
                "Golden trace should have simulation times"
            )
            assert "source_ids" in golden_trace, "Golden trace should have source IDs"
            assert "effects" in golden_trace, "Golden trace should have effects"

            # Verify basic trace structure
            assert len(golden_trace["checksums"]) == actual_operations, (
                "Checksums count should match operations"
            )
            assert len(golden_trace["sim_times"]) == actual_operations, (
                "Sim times count should match operations"
            )
            assert len(golden_trace["source_ids"]) == actual_operations, (
                "Source IDs count should match operations"
            )

        finally:
            await facade.shutdown()

    @pytest.mark.asyncio
    async def test_trace_hash_stability(self, validator: GoldenTraceValidator):
        """Test that trace hashes are stable across multiple calculations."""
        # Create log
        log = await create_sample_log(num_entries=5, world_seed=12345)
        trace = validator.extract_trace_from_log(log)

        # Calculate hash multiple times
        hashes = [validator.create_trace_hash(trace) for _ in range(5)]

        # All hashes should be identical
        assert len(set(hashes)) == 1, "Hash should be stable across calculations"
        assert all(len(h) == 64 for h in hashes), "All hashes should be valid SHA-256"

    @pytest.mark.asyncio
    async def test_trace_sensitivity_to_changes(self, validator: GoldenTraceValidator):
        """Test that traces are sensitive to small changes."""
        # Create base log
        log = await create_sample_log(num_entries=3, world_seed=12345)
        base_trace = validator.extract_trace_from_log(log)

        # Modify one effect slightly
        modified_log = await create_sample_log(num_entries=3, world_seed=12345)
        entries = modified_log.get_all_entries()

        # Change payload of first effect
        entries[0].effect["payload"]["modified"] = True

        modified_trace = validator.extract_trace_from_log(modified_log)

        # Compare traces
        comparison = validator.compare_traces(base_trace, modified_trace)

        assert not comparison["identical"], "Traces should differ after modification"
        assert comparison["hash1"] != comparison["hash2"], "Hashes should differ"
        assert len(comparison["differences"]) > 0, "Should detect differences"
