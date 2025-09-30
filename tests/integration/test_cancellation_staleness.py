"""Unit tests for cancellation and staleness detection functionality.

Tests the enhanced cancellation system including automatic cancellation
when context becomes outdated, debounce logic, and interrupt policies.
"""

import asyncio
import time
from unittest.mock import Mock

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.types import CancelToken, Effect, EffectDraft


class TestCancellationAndStaleness:
    """Test suite for cancellation and staleness detection."""

    @pytest.fixture
    def config(self):
        """Create test configuration with short debounce time."""
        return OrchestratorConfig(
            staleness_threshold=0,  # Any staleness triggers cancellation
            debounce_ms=50.0,  # Short debounce for testing
            use_in_memory_dedup=True,
        )

    @pytest.fixture
    async def orchestrator(self, config):
        """Create and initialize orchestrator for testing."""
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        return orchestrator

    @pytest.fixture
    def mock_policy(self):
        """Create mock observation policy."""
        policy = Mock(spec=ObservationPolicy)
        policy.should_observe_event.return_value = True
        policy.filter_world_state.return_value = Mock(
            view_seq=0,
            visible_entities={},
            visible_relationships={},
            context_digest="test_digest",
        )

        # Create mock latency model
        latency_model = Mock()
        latency_model.calculate_delay.return_value = 0.001  # 1ms delay
        policy.latency_model = latency_model

        return policy

    @pytest.mark.asyncio
    async def test_issue_cancel_token_basic(self, orchestrator):
        """Test basic cancel token issuance."""
        token = orchestrator.issue_cancel_token("agent_1", "req_1")

        assert isinstance(token, CancelToken)
        assert token.req_id == "req_1"
        assert token.agent_id == "agent_1"
        assert not token.cancelled
        assert token.reason is None

    @pytest.mark.asyncio
    async def test_issue_cancel_token_with_context(self, orchestrator):
        """Test cancel token issuance with context digest."""
        token = orchestrator.issue_cancel_token("agent_1", "req_1", "context_123")

        assert isinstance(token, CancelToken)
        assert token.req_id == "req_1"
        assert token.agent_id == "agent_1"
        assert not token.cancelled

    @pytest.mark.asyncio
    async def test_issue_cancel_token_validation(self, orchestrator):
        """Test cancel token issuance validation."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            orchestrator.issue_cancel_token("", "req_1")

        with pytest.raises(ValueError, match="req_id cannot be empty"):
            orchestrator.issue_cancel_token("agent_1", "")

    @pytest.mark.asyncio
    async def test_issue_cancel_token_replacement(self, orchestrator):
        """Test that new tokens replace old ones for the same key."""
        # Issue first token
        token1 = orchestrator.issue_cancel_token("agent_1", "req_1")
        assert not token1.cancelled

        # Issue second token with same key
        token2 = orchestrator.issue_cancel_token("agent_1", "req_1")

        # First token should be cancelled
        assert token1.cancelled
        assert token1.reason == "replaced_by_new_token"

        # Second token should be active
        assert not token2.cancelled

    @pytest.mark.asyncio
    async def test_cancel_if_stale_basic(self, orchestrator, mock_policy):
        """Test basic staleness detection and cancellation."""
        # Register agent
        agent_handle = await orchestrator.register_agent("agent_1", mock_policy)
        agent_handle.view_seq = 5  # Set current view sequence

        # Issue cancel token
        token = orchestrator.issue_cancel_token("agent_1", "req_1")

        # Test staleness with threshold exceeded
        was_cancelled = await orchestrator.cancel_if_stale("agent_1", "req_1", 10)

        assert was_cancelled
        assert token.cancelled
        assert "stale_due_to_seq_gap_5_threshold_0" in token.reason

    @pytest.mark.asyncio
    async def test_cancel_if_stale_within_threshold(self, orchestrator, mock_policy):
        """Test that tokens are not cancelled when within staleness threshold."""
        # Register agent
        agent_handle = await orchestrator.register_agent("agent_1", mock_policy)
        agent_handle.view_seq = 5

        # Issue cancel token
        token = orchestrator.issue_cancel_token("agent_1", "req_1")

        # Test staleness equal to threshold (should not cancel since staleness must be > threshold)
        # With threshold=0, staleness=0 should not cancel
        was_cancelled = await orchestrator.cancel_if_stale(
            "agent_1", "req_1", 5
        )  # Same as view_seq

        assert not was_cancelled
        assert not token.cancelled

    @pytest.mark.asyncio
    async def test_cancel_if_stale_debounce(self, orchestrator, mock_policy):
        """Test debounce logic prevents rapid successive cancellations."""
        # Register agent
        agent_handle = await orchestrator.register_agent("agent_1", mock_policy)
        agent_handle.view_seq = 5

        # Issue first token and cancel it
        token1 = orchestrator.issue_cancel_token("agent_1", "req_1")
        was_cancelled1 = await orchestrator.cancel_if_stale("agent_1", "req_1", 10)
        assert was_cancelled1
        assert token1.cancelled

        # Immediately issue second token and try to cancel
        token2 = orchestrator.issue_cancel_token("agent_1", "req_2")
        was_cancelled2 = await orchestrator.cancel_if_stale("agent_1", "req_2", 15)

        # Should be suppressed by debounce
        assert not was_cancelled2
        assert not token2.cancelled

        # Wait for debounce period to pass
        await asyncio.sleep(0.06)  # 60ms > 50ms debounce

        # Now cancellation should work
        was_cancelled3 = await orchestrator.cancel_if_stale("agent_1", "req_2", 20)
        assert was_cancelled3
        assert token2.cancelled

    @pytest.mark.asyncio
    async def test_cancel_if_stale_nonexistent_token(self, orchestrator):
        """Test cancel_if_stale with non-existent token."""
        was_cancelled = await orchestrator.cancel_if_stale("agent_1", "nonexistent", 10)
        assert not was_cancelled

    @pytest.mark.asyncio
    async def test_cancel_if_stale_already_cancelled(self, orchestrator, mock_policy):
        """Test cancel_if_stale with already cancelled token."""
        # Register agent
        await orchestrator.register_agent("agent_1", mock_policy)

        # Issue and manually cancel token
        token = orchestrator.issue_cancel_token("agent_1", "req_1")
        token.cancel("manual_cancellation")

        # Try to cancel again
        was_cancelled = await orchestrator.cancel_if_stale("agent_1", "req_1", 10)
        assert was_cancelled  # Returns True because token is already cancelled
        assert token.reason == "manual_cancellation"  # Reason unchanged

    @pytest.mark.asyncio
    async def test_set_agent_interrupt_policy(self, orchestrator):
        """Test setting agent interrupt policies."""
        # Test valid policies
        orchestrator.set_agent_interrupt_policy("agent_1", "always")
        assert orchestrator.get_agent_interrupt_policy("agent_1") == "always"

        orchestrator.set_agent_interrupt_policy("agent_1", "only_conflict")
        assert orchestrator.get_agent_interrupt_policy("agent_1") == "only_conflict"

        # Test invalid policy
        with pytest.raises(ValueError, match="Invalid interrupt policy"):
            orchestrator.set_agent_interrupt_policy("agent_1", "invalid")

    @pytest.mark.asyncio
    async def test_get_agent_interrupt_policy_default(self, orchestrator):
        """Test default interrupt policy."""
        # Should default to "always"
        assert orchestrator.get_agent_interrupt_policy("nonexistent_agent") == "always"

    @pytest.mark.asyncio
    async def test_check_and_cancel_stale_tokens_always_policy(
        self, orchestrator, mock_policy
    ):
        """Test automatic cancellation with 'always' interrupt policy."""
        # Register agent with 'always' policy
        agent_handle = await orchestrator.register_agent("agent_1", mock_policy)
        agent_handle.view_seq = 5
        orchestrator.set_agent_interrupt_policy("agent_1", "always")

        # Issue cancel token
        token = orchestrator.issue_cancel_token("agent_1", "req_1")

        # Create effect from different agent
        effect = Effect(
            uuid="test_uuid",
            kind="TestEffect",
            payload={},
            global_seq=10,
            sim_time=1.0,
            source_id="agent_2",  # Different agent
            schema_version="1.0.0",
        )

        # Check for stale tokens
        cancelled_req_ids = await orchestrator.check_and_cancel_stale_tokens(effect)

        assert "req_1" in cancelled_req_ids
        assert token.cancelled

    @pytest.mark.asyncio
    async def test_check_and_cancel_stale_tokens_only_conflict_policy(
        self, orchestrator, mock_policy
    ):
        """Test automatic cancellation with 'only_conflict' interrupt policy."""
        # Register agent with 'only_conflict' policy
        agent_handle = await orchestrator.register_agent("agent_1", mock_policy)
        agent_handle.view_seq = 5
        orchestrator.set_agent_interrupt_policy("agent_1", "only_conflict")

        # Issue cancel token
        token = orchestrator.issue_cancel_token("agent_1", "req_1")

        # Create effect from same agent (should not trigger cancellation)
        effect_same_agent = Effect(
            uuid="test_uuid_1",
            kind="TestEffect",
            payload={},
            global_seq=10,
            sim_time=1.0,
            source_id="agent_1",  # Same agent
            schema_version="1.0.0",
        )

        cancelled_req_ids = await orchestrator.check_and_cancel_stale_tokens(
            effect_same_agent
        )
        assert "req_1" not in cancelled_req_ids
        assert not token.cancelled

        # Create effect from different agent (should trigger cancellation)
        effect_different_agent = Effect(
            uuid="test_uuid_2",
            kind="TestEffect",
            payload={},
            global_seq=15,
            sim_time=2.0,
            source_id="agent_2",  # Different agent
            schema_version="1.0.0",
        )

        cancelled_req_ids = await orchestrator.check_and_cancel_stale_tokens(
            effect_different_agent
        )
        assert "req_1" in cancelled_req_ids
        assert token.cancelled

    @pytest.mark.asyncio
    async def test_cleanup_cancelled_tokens(self, orchestrator):
        """Test cleanup of cancelled tokens."""
        # Issue several tokens
        token1 = orchestrator.issue_cancel_token("agent_1", "req_1")
        token2 = orchestrator.issue_cancel_token("agent_1", "req_2")
        _ = orchestrator.issue_cancel_token("agent_2", "req_3")

        # Cancel some tokens
        token1.cancel("test_reason")
        token2.cancel("test_reason")

        # Cleanup should remove cancelled tokens
        cleaned_count = orchestrator.cleanup_cancelled_tokens()

        assert cleaned_count == 2
        assert len(orchestrator._cancel_tokens) == 1

        # Verify only active token remains
        active_tokens = orchestrator.get_active_cancel_tokens()
        assert "agent_2" in active_tokens
        assert "req_3" in active_tokens["agent_2"]
        assert "agent_1" not in active_tokens

    @pytest.mark.asyncio
    async def test_get_active_cancel_tokens(self, orchestrator):
        """Test getting active cancel tokens grouped by agent."""
        # Issue tokens for multiple agents
        _ = orchestrator.issue_cancel_token("agent_1", "req_1")
        token2 = orchestrator.issue_cancel_token("agent_1", "req_2")
        _ = orchestrator.issue_cancel_token("agent_2", "req_3")

        # Cancel one token
        token2.cancel("test_reason")

        active_tokens = orchestrator.get_active_cancel_tokens()

        assert "agent_1" in active_tokens
        assert "req_1" in active_tokens["agent_1"]
        assert "req_2" not in active_tokens["agent_1"]

        assert "agent_2" in active_tokens
        assert "req_3" in active_tokens["agent_2"]

    @pytest.mark.asyncio
    async def test_automatic_cancellation_during_observation_distribution(
        self, orchestrator, mock_policy
    ):
        """Test that observation distribution triggers automatic cancellation."""
        # Register agent
        agent_handle = await orchestrator.register_agent("agent_1", mock_policy)
        agent_handle.view_seq = 0  # Start with view_seq 0

        # Issue cancel token
        token = orchestrator.issue_cancel_token("agent_1", "req_1")

        # Broadcast event that should trigger cancellation
        # This will create an effect with global_seq = 1
        effect_draft = EffectDraft(
            kind="TestEvent",
            payload={"test": "data"},
            source_id="agent_2",
            schema_version="1.0.0",
        )

        await orchestrator.broadcast_event(effect_draft)

        # Token should be cancelled due to staleness
        # Staleness = new_global_seq (1) - agent_view_seq (0) = 1
        # Since staleness (1) > threshold (1), it should be cancelled
        # Wait a moment for async processing
        await asyncio.sleep(0.01)

        assert token.cancelled
        assert "stale_due_to_seq_gap" in token.reason

    @pytest.mark.asyncio
    async def test_cancellation_timing_accuracy(self, orchestrator, mock_policy):
        """Test cancellation timing meets requirements."""
        # Register agent
        agent_handle = await orchestrator.register_agent("agent_1", mock_policy)
        agent_handle.view_seq = 5

        # Issue cancel token
        token = orchestrator.issue_cancel_token("agent_1", "req_1")

        # Measure cancellation timing
        start_time = time.perf_counter()
        was_cancelled = await orchestrator.cancel_if_stale("agent_1", "req_1", 10)
        end_time = time.perf_counter()

        cancellation_time_ms = (end_time - start_time) * 1000

        assert was_cancelled
        assert token.cancelled
        # Cancellation should be very fast (< 5ms for in-memory operations)
        assert cancellation_time_ms < 5.0

    @pytest.mark.asyncio
    async def test_multiple_agents_cancellation(self, orchestrator, mock_policy):
        """Test cancellation works correctly with multiple agents."""
        # Register multiple agents
        agent1_handle = await orchestrator.register_agent("agent_1", mock_policy)
        agent2_handle = await orchestrator.register_agent("agent_2", mock_policy)
        agent1_handle.view_seq = 5
        agent2_handle.view_seq = 8

        # Issue tokens for both agents
        token1 = orchestrator.issue_cancel_token("agent_1", "req_1")
        token2 = orchestrator.issue_cancel_token("agent_2", "req_2")

        # Create effect that should trigger cancellation for both
        effect = Effect(
            uuid="test_uuid",
            kind="TestEffect",
            payload={},
            global_seq=15,
            sim_time=1.0,
            source_id="agent_3",
            schema_version="1.0.0",
        )

        cancelled_req_ids = await orchestrator.check_and_cancel_stale_tokens(effect)

        # Both tokens should be cancelled due to staleness
        assert "req_1" in cancelled_req_ids
        assert "req_2" in cancelled_req_ids
        assert token1.cancelled
        assert token2.cancelled

    @pytest.mark.asyncio
    async def test_debounce_per_agent(self, orchestrator, mock_policy):
        """Test that debounce logic is applied per agent."""
        # Register multiple agents
        agent1_handle = await orchestrator.register_agent("agent_1", mock_policy)
        agent2_handle = await orchestrator.register_agent("agent_2", mock_policy)
        agent1_handle.view_seq = 5
        agent2_handle.view_seq = 5

        # Cancel token for agent_1 to start debounce timer
        token1 = orchestrator.issue_cancel_token("agent_1", "req_1")
        await orchestrator.cancel_if_stale("agent_1", "req_1", 10)
        assert token1.cancelled

        # Immediately try to cancel token for agent_2 (should work - different agent)
        token2 = orchestrator.issue_cancel_token("agent_2", "req_2")
        was_cancelled = await orchestrator.cancel_if_stale("agent_2", "req_2", 10)

        assert was_cancelled
        assert token2.cancelled

        # Try to cancel another token for agent_1 (should be debounced)
        token3 = orchestrator.issue_cancel_token("agent_1", "req_3")
        was_cancelled = await orchestrator.cancel_if_stale("agent_1", "req_3", 15)

        assert not was_cancelled
        assert not token3.cancelled
