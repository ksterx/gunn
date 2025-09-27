"""Integration tests for replay CLI functionality."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from gunn.cli.replay import ReplayEngine, run_replay_command
from gunn.core.event_log import EventLog
from gunn.schemas.types import Effect


class TestReplayEngine:
    """Test ReplayEngine functionality."""

    @pytest.fixture
    async def sample_log_file(self) -> Path:
        """Create a sample log file for testing."""
        # Create sample effects
        effects = []
        for i in range(5):
            effect = {
                "uuid": f"uuid-{i}",
                "kind": "Move" if i % 2 == 0 else "Speak",
                "payload": {"x": i, "y": i} if i % 2 == 0 else {"text": f"Message {i}"},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": f"agent_{i % 2}",  # Two agents
                "schema_version": "1.0.0",
            }
            effects.append(effect)

        # Create event log and add effects
        event_log = EventLog(world_seed=12345)
        for effect in effects:
            await event_log.append(effect, f"req_{effect['global_seq']}")

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        await event_log.save_to_file(temp_path)
        return temp_path

    @pytest.mark.asyncio
    async def test_load_log(self, sample_log_file: Path):
        """Test loading a log file."""
        engine = ReplayEngine(sample_log_file)
        await engine.load_log()
        
        assert engine.event_log.entry_count == 5
        assert engine.event_log.world_seed == 12345

    @pytest.mark.asyncio
    async def test_replay_range_full(self, sample_log_file: Path):
        """Test replaying full range."""
        engine = ReplayEngine(sample_log_file)
        await engine.load_log()
        
        entries = await engine.replay_range(0)
        assert len(entries) == 5
        
        # Verify world state was updated
        assert "last_effect_seq" in engine.world_state.metadata
        assert engine.world_state.metadata["last_effect_seq"] == 4

    @pytest.mark.asyncio
    async def test_replay_range_partial(self, sample_log_file: Path):
        """Test replaying partial range."""
        engine = ReplayEngine(sample_log_file)
        await engine.load_log()
        
        entries = await engine.replay_range(2, 3)
        assert len(entries) == 2
        assert entries[0].global_seq == 2
        assert entries[1].global_seq == 3

    @pytest.mark.asyncio
    async def test_validate_determinism(self, sample_log_file: Path):
        """Test determinism validation."""
        engine = ReplayEngine(sample_log_file)
        await engine.load_log()
        
        is_deterministic = await engine.validate_determinism(iterations=3)
        assert is_deterministic is True

    @pytest.mark.asyncio
    async def test_validate_integrity(self, sample_log_file: Path):
        """Test integrity validation."""
        engine = ReplayEngine(sample_log_file)
        await engine.load_log()
        
        is_valid = await engine.validate_integrity()
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_effect_application_move(self, sample_log_file: Path):
        """Test Move effect application."""
        engine = ReplayEngine(sample_log_file)
        await engine.load_log()
        
        # Apply a Move effect
        move_effect = {
            "uuid": "test-uuid",
            "kind": "Move",
            "payload": {"x": 100, "y": 200, "z": 50},
            "global_seq": 999,
            "sim_time": 999.0,
            "source_id": "test_agent",
            "schema_version": "1.0.0",
        }
        
        await engine._apply_effect(move_effect)
        
        # Verify spatial index was updated
        assert "test_agent" in engine.world_state.spatial_index
        assert engine.world_state.spatial_index["test_agent"] == (100.0, 200.0, 50.0)

    @pytest.mark.asyncio
    async def test_effect_application_speak(self, sample_log_file: Path):
        """Test Speak effect application."""
        engine = ReplayEngine(sample_log_file)
        await engine.load_log()
        
        # Apply a Speak effect
        speak_effect = {
            "uuid": "test-uuid",
            "kind": "Speak",
            "payload": {"text": "Hello, world!"},
            "global_seq": 999,
            "sim_time": 999.0,
            "source_id": "test_agent",
            "schema_version": "1.0.0",
        }
        
        await engine._apply_effect(speak_effect)
        
        # Verify entity was updated
        assert "test_agent" in engine.world_state.entities
        assert engine.world_state.entities["test_agent"]["last_message"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_effect_application_interact(self, sample_log_file: Path):
        """Test Interact effect application."""
        engine = ReplayEngine(sample_log_file)
        await engine.load_log()
        
        # Apply an Interact effect
        interact_effect = {
            "uuid": "test-uuid",
            "kind": "Interact",
            "payload": {"target_id": "target_agent"},
            "global_seq": 999,
            "sim_time": 999.0,
            "source_id": "test_agent",
            "schema_version": "1.0.0",
        }
        
        await engine._apply_effect(interact_effect)
        
        # Verify relationship was created
        assert "test_agent" in engine.world_state.relationships
        assert "target_agent" in engine.world_state.relationships["test_agent"]


class TestReplayCommand:
    """Test replay command functionality."""

    @pytest.fixture
    async def sample_log_file(self) -> Path:
        """Create a sample log file for testing."""
        # Create sample effects with deterministic behavior
        effects = []
        for i in range(10):
            effect = {
                "uuid": f"uuid-{i:03d}",  # Padded for consistent ordering
                "kind": "Move",
                "payload": {"x": i * 10, "y": i * 20},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": f"agent_{i % 3}",  # Three agents
                "schema_version": "1.0.0",
            }
            effects.append(effect)

        # Create event log and add effects
        event_log = EventLog(world_seed=54321)
        for effect in effects:
            await event_log.append(effect, f"req_{effect['global_seq']}")

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        await event_log.save_to_file(temp_path)
        return temp_path

    @pytest.mark.asyncio
    async def test_replay_command_basic(self, sample_log_file: Path):
        """Test basic replay command."""
        args = [str(sample_log_file)]
        result = await run_replay_command(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_replay_command_with_range(self, sample_log_file: Path):
        """Test replay command with range."""
        args = [str(sample_log_file), "--from", "2", "--to", "5"]
        result = await run_replay_command(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_replay_command_validate_integrity(self, sample_log_file: Path):
        """Test replay command with integrity validation."""
        args = [str(sample_log_file), "--validate-integrity"]
        result = await run_replay_command(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_replay_command_validate_determinism(self, sample_log_file: Path):
        """Test replay command with determinism validation."""
        args = [str(sample_log_file), "--validate-determinism", "--determinism-iterations", "2"]
        result = await run_replay_command(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_replay_command_with_output(self, sample_log_file: Path):
        """Test replay command with output file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)

        try:
            args = [str(sample_log_file), "--output", str(output_path)]
            result = await run_replay_command(args)
            assert result == 0
            
            # Verify output file was created and contains expected data
            assert output_path.exists()
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            
            assert "metadata" in output_data
            assert "final_world_state" in output_data
            assert "replayed_entries" in output_data
            assert output_data["metadata"]["entry_count"] == 10

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_replay_command_nonexistent_file(self):
        """Test replay command with nonexistent file."""
        args = ["/nonexistent/file.json"]
        result = await run_replay_command(args)
        assert result == 1

    @pytest.mark.asyncio
    async def test_replay_command_verbose(self, sample_log_file: Path):
        """Test replay command with verbose output."""
        args = [str(sample_log_file), "--verbose"]
        result = await run_replay_command(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_replay_command_custom_seed(self, sample_log_file: Path):
        """Test replay command with custom seed."""
        args = [str(sample_log_file), "--seed", "99999"]
        result = await run_replay_command(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_replay_command_help(self):
        """Test replay command help."""
        args = ["--help"]
        result = await run_replay_command(args)
        assert result in [0, 1]  # Help can return either depending on implementation


class TestReplayDeterminism:
    """Test deterministic replay behavior."""

    @pytest.mark.asyncio
    async def test_deterministic_replay_consistency(self):
        """Test that multiple replays produce identical results."""
        # Create a log with random elements
        event_log = EventLog(world_seed=11111)
        
        # Add effects that use random numbers
        import random
        random.seed(11111)  # Set seed for consistent test
        
        for i in range(20):
            # Simulate some randomness in effect creation
            random_x = random.randint(0, 100)
            random_y = random.randint(0, 100)
            
            effect = {
                "uuid": f"uuid-{i:03d}",
                "kind": "Move",
                "payload": {"x": random_x, "y": random_y},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": f"agent_{i % 2}",
                "schema_version": "1.0.0",
            }
            await event_log.append(effect)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            await event_log.save_to_file(temp_path)

            # Perform multiple replays and compare results
            results = []
            for iteration in range(3):
                engine = ReplayEngine(temp_path)
                await engine.load_log()
                
                # Replay all events
                await engine.replay_range(0)
                
                # Capture final state
                final_state = engine.world_state.model_dump()
                results.append(final_state)

            # All results should be identical
            for i in range(1, len(results)):
                assert results[i] == results[0], f"Iteration {i} produced different results"

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_seed_override_determinism(self):
        """Test that seed override produces consistent results."""
        # Create a log
        event_log = EventLog(world_seed=22222)
        
        for i in range(5):
            effect = {
                "uuid": f"uuid-{i}",
                "kind": "Move",
                "payload": {"x": i, "y": i},
                "global_seq": i,
                "sim_time": float(i),
                "source_id": "agent_1",
                "schema_version": "1.0.0",
            }
            await event_log.append(effect)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            await event_log.save_to_file(temp_path)

            # Test with different seed overrides
            override_seed = 99999
            
            results = []
            for iteration in range(2):
                engine = ReplayEngine(temp_path, world_seed=override_seed)
                await engine.load_log()
                
                # Verify seed was overridden
                assert engine.event_log.world_seed == override_seed
                
                await engine.replay_range(0)
                final_state = engine.world_state.model_dump()
                results.append(final_state)

            # Results should be identical with same override seed
            assert results[1] == results[0]

        finally:
            temp_path.unlink()